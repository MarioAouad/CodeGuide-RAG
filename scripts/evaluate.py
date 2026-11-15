"""Verify real sample retrieval; add --live to send seven questions to Gemini."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import ROOT, load_settings  # noqa: E402
from rag.generation import INSUFFICIENT_INFORMATION, answer_question  # noqa: E402
from rag.retrieval import LocalEmbedder, build_index  # noqa: E402

CASES = [
    ("What does loss(w, b) calculate?", {3, 4}),
    ("How is the gradient calculated and checked?", {6, 7, 8}),
    ("What does gd return, and does it include the starting point?", {9, 10, 11}),
    ("Why does the notebook say a learning rate of 0.04 diverges?", {12, 13}),
    ("How does classify decide whether a stationary point is a saddle?", {17, 18, 19}),
    ("How does the optional PyTorch section obtain gradients?", {24, 25}),
]
UNSUPPORTED = "Which database stores this application's user passwords?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Send questions to configured Gemini API"
    )
    args = parser.parse_args()
    settings = load_settings()
    if args.live and not settings.api_key:
        parser.error("Set GEMINI_API_KEY locally before using --live.")
    embedder = LocalEmbedder(settings.embedding_model)
    index = build_index(
        {"lab01.ipynb": (ROOT / "Knowledge_Source" / "lab01.ipynb").read_bytes()},
        embedder,
        settings,
    )
    failures = 0
    records = []
    for question, targets in CASES + [(UNSUPPORTED, set())]:
        hits = index.search(question, embedder, settings)
        cells = sorted({hit.chunk.cell_index for hit in hits})
        passed = bool(targets.intersection(cells)) if targets else None
        failures += passed is False
        record = {"question": question, "cells": cells, "retrieval_pass": passed}
        if args.live:
            answer = answer_question(question, hits, settings)
            record["answer"] = answer
            if not targets:
                refused = answer == INSUFFICIENT_INFORMATION
                record["unsupported_refusal_pass"] = refused
                failures += not refused
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    output = ROOT / "data" / "evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {output}. Live grounding still needs human review.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
