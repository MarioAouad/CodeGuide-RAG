"""Capture the real local application; --live authorizes Gemini test questions."""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_settings  # noqa: E402
from rag.generation import INSUFFICIENT_INFORMATION, valid_citations  # noqa: E402
from scripts.evaluate import CASES, UNSUPPORTED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--workflow", action="store_true", help="Also capture upload and reset states."
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to authorize sending test questions to configured Gemini.")
    OUT.mkdir(exist_ok=True)
    report_path = OUT / "verification_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(url=args.url, recorded_at_utc=datetime.now(UTC).isoformat())
    report["generation_model"] = load_settings().gemini_model
    report["browser"] = "Microsoft Edge (Playwright), 1600 x 1400 viewport"
    report.setdefault("questions", {})

    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(args.url + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("The local Streamlit server is not ready.") from None
        time.sleep(0.25)

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1400}, device_scale_factor=1)
        page.set_default_timeout(30000)
        page.goto(args.url, wait_until="domcontentloaded")
        expect(page.get_by_role("heading", name="CodeGuide", exact=True)).to_be_visible()

        def wait_idle():
            expect(page.get_by_test_id("stApp")).to_have_attribute(
                "data-test-script-state", "notRunning", timeout=180000
            )

        def screenshot(name):
            wait_idle()
            # Expanders animate after the script becomes idle. Wait for their
            # content before capturing; otherwise an open card can look empty.
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / name), full_page=True, animations="disabled")
            print("Saved:", name, flush=True)

        screenshot("01_start_screen.png")
        expect(page.get_by_role("button", name="Ask", exact=True)).to_be_disabled()
        report["initial_state_pass"] = True
        page.get_by_role("button", name="Load sample notebook", exact=True).click()
        page.get_by_role("button", name="Index files", exact=True).click()
        try:
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled(timeout=180000)
        except Exception:
            screenshot("02_indexing_error.png")
            print(page.locator("body").inner_text(), flush=True)
            raise
        report["sample_indexing_pass"] = True
        screenshot("02_notebook_indexed.png")
        print(
            "Text element samples:", page.get_by_test_id("stText").all_text_contents(), flush=True
        )
        cases = CASES + [(UNSUPPORTED, set())]
        slugs = [
            "loss",
            "gradient",
            "gradient_descent",
            "learning_rate",
            "hessian",
            "autograd",
            "unsupported_question",
        ]

        for number in range(args.start, min(args.start + args.limit, len(cases))):
            question, targets = cases[number]
            previous = page.get_by_test_id("stText").all_text_contents()
            question_field = page.get_by_role("textbox", name="Your question", exact=True)
            question_field.fill(question)
            page.get_by_role("button", name="Ask", exact=True).click()
            try:
                wait_idle()
                page.wait_for_function(
                    """previous => {
                      const texts = [...document.querySelectorAll('[data-testid="stText"]')]
                        .map(e=>e.textContent);
                      const hasAnswer = texts.some(
                        t => t !== 'lab01.ipynb' && !previous.includes(t));
                      const alerts = [...document.querySelectorAll('[data-testid="stAlert"]')];
                      const failure = 'quota|rejected|unavailable|Could not reach|request failed'
                        + '|invalid citations|Set GEMINI_API_KEY';
                      const error = alerts.some(e => new RegExp(failure).test(e.textContent));
                      return hasAnswer || error;
                    }""",
                    arg=previous,
                    timeout=90000,
                )
                expect(page.get_by_role("heading", name="Answer", exact=True)).to_be_visible(
                    timeout=1000
                )
                wait_idle()
            except Exception:
                screenshot(f"{number + 3:02d}_{slugs[number]}_error.png")
                body = page.locator("body").inner_text()
                report["questions"][slugs[number]] = {
                    "question": question,
                    "pass": False,
                    "visible_error_state": body,
                }
                save_report()
                print("Live answer failed:", body, flush=True)
                raise

            answers = [
                t for t in page.get_by_test_id("stText").all_text_contents() if t != "lab01.ipynb"
            ]
            answer = max(answers, key=len)
            source_labels = page.locator("summary").all_text_contents()
            source_labels = [label for label in source_labels if re.search(r"\[S\d+\]", label)]
            cells = sorted(
                {int(n) for label in source_labels for n in re.findall(r"cell (\d+)", label)}
            )
            citation_pass = valid_citations(answer, len(source_labels))
            retrieval_pass = bool(targets.intersection(cells)) if targets else None
            refusal_pass = answer.strip() == INSUFFICIENT_INFORMATION if not targets else None
            record = {
                "question": question,
                "answer": answer,
                "source_labels": source_labels,
                "retrieved_cells": cells,
                "citation_pass": citation_pass,
                "retrieval_pass": retrieval_pass,
                "unsupported_refusal_pass": refusal_pass,
                "pass": citation_pass and retrieval_pass is not False and refusal_pass is not False,
            }
            report["questions"][slugs[number]] = record
            save_report()
            page.get_by_role("heading", name="CodeGuide", exact=True).scroll_into_view_if_needed()
            screenshot(f"{number + 3:02d}_{slugs[number]}_answer.png")

            if targets:
                summaries = page.locator("summary")
                for i in range(summaries.count()):
                    label = summaries.nth(i).inner_text()
                    matched_cells = {int(n) for n in re.findall(r"cell (\d+)", label)}
                    if "[S" in label and targets.intersection(matched_cells):
                        summaries.nth(i).click()
                        expect(
                            summaries.nth(i).locator("..").get_by_test_id("stCode")
                        ).to_be_visible()
                        summaries.nth(i).scroll_into_view_if_needed()
                        screenshot(f"{number + 3:02d}_{slugs[number]}_source.png")
                        summaries.nth(i).click()
                        break
            print(json.dumps(record, ensure_ascii=True), flush=True)

        if args.workflow:
            page.get_by_text("Example questions", exact=True).click()
            screenshot("10_example_questions.png")
            page.get_by_role("button", name="What does loss(w, b) calculate?", exact=True).click()
            wait_idle()
            expect(page.get_by_role("textbox", name="Your question")).to_have_value(
                "What does loss(w, b) calculate?"
            )
            page.get_by_text("Example questions", exact=True).click()
            page.get_by_role("textbox", name="Your question").fill("")
            page.get_by_role("button", name="Ask", exact=True).click()
            wait_idle()
            expect(page.get_by_text("Enter a question first.", exact=True)).to_be_visible()
            screenshot("11_empty_question_validation.png")

            upload = page.locator('input[type="file"]')
            upload.set_input_files(
                {"name": "invalid.ipynb", "mimeType": "application/json", "buffer": b"{broken"}
            )
            wait_idle()
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_disabled()
            screenshot("12_changed_selection.png")
            page.get_by_role("button", name="Index files", exact=True).click()
            wait_idle()
            expect(page.get_by_text("invalid.ipynb: malformed notebook JSON.")).to_be_visible()
            screenshot("13_invalid_notebook.png")

            page.get_by_role("button", name="Load sample notebook", exact=True).click()
            wait_idle()
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled()
            page.get_by_role("button", name="Index files", exact=True).click()
            wait_idle()
            expect(
                page.get_by_text("Reused saved embeddings: 25 sources, 49 chunks.")
            ).to_be_visible()
            screenshot("14_saved_index_reused.png")

            upload = page.locator('input[type="file"]')
            upload.set_input_files(
                [
                    {
                        "name": "lab01.ipynb",
                        "mimeType": "application/json",
                        "buffer": (ROOT / "Knowledge_Source" / "lab01.ipynb").read_bytes(),
                    },
                    {
                        "name": "demo.py",
                        "mimeType": "text/plain",
                        "buffer": b'def greet(name):\n    return f"Hello, {name}"\n',
                    },
                    {
                        "name": "notes.md",
                        "mimeType": "text/markdown",
                        "buffer": b"# Greeting example\ngreet returns a greeting string.\n",
                    },
                    {
                        "name": "readme.txt",
                        "mimeType": "text/plain",
                        "buffer": b"These four files demonstrate supported source formats.\n",
                    },
                ]
            )
            wait_idle()
            page.get_by_role("button", name="Index files", exact=True).click()
            wait_idle()
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled()
            for name in ("lab01.ipynb", "demo.py", "notes.md", "readme.txt"):
                expect(page.get_by_test_id("stText").filter(has_text=name)).to_be_visible()
            screenshot("15_all_supported_uploads.png")
            page.get_by_role("button", name="Clear active index", exact=True).click()
            wait_idle()
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_disabled()
            expect(page.get_by_role("button", name="Index files", exact=True)).to_be_disabled()
            expect(page.get_by_role("heading", name="Answer", exact=True)).to_have_count(0)
            screenshot("16_index_cleared.png")
            report["workflow"] = {
                "examples_fill_question": True,
                "empty_question_validation": True,
                "changed_selection_blocks_questions": True,
                "invalid_notebook_rejected": True,
                "previous_index_preserved": True,
                "saved_embeddings_reused": True,
                "all_four_upload_formats": True,
                "clear_index_resets_session": True,
            }

        save_report()
        browser.close()
    selected = slugs[args.start : min(args.start + args.limit, len(cases))]
    if not all(report["questions"][slug]["pass"] for slug in selected):
        raise SystemExit("One or more live checks failed; see verification_results.json.")


if __name__ == "__main__":
    main()
