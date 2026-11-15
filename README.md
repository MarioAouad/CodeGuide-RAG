# CodeGuide: a small code-explanation RAG

CodeGuide answers questions about Python code and notebooks. It retrieves
relevant excerpts locally, sends the question and those excerpts to Gemini,
and shows the answer beside the original source evidence.

This is the portable submission folder. Installed packages, downloaded models,
generated indexes, API keys, logs, and presentation build files are excluded.
Extract the ZIP before installing or running the application.

## Start here

- [Project PowerPoint](presentation/CodeGuide_RAG_Presentation.pptx): 27 slides with speaker notes and real screenshots.
- [Screenshot evidence](screenshots/README.md): 22 screenshots covering answers, sources, uploads, validation, and reset.
- [Recorded questions and answers](screenshots/questions_and_answers.md).
- [Architecture](docs/architecture.md) and [validation record](docs/validation.md).

## Install and run

Use Python 3.11 or 3.12. The recorded test environment used Python 3.12.7.
Open PowerShell inside the extracted `code_sample_RAG` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit the new `.env` locally and set `GEMINI_API_KEY`. Use a Gemini project with
free-tier access. The configured answer model is `gemini-3.5-flash-lite`.
If `.env` already exists, keep your local settings instead of copying over it.
Environment variables take precedence over `.env`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open **http://127.0.0.1:8502**. On macOS/Linux, use `.venv/bin/python` instead
of `.\.venv\Scripts\python.exe`. If Windows opens the Microsoft Store for
`python`, use the full path of your installed Python interpreter.

The first indexing run downloads `sentence-transformers/all-MiniLM-L6-v2`.
The model then embeds text locally on the CPU. Download time and disk space
depend on the installed packages and model cache. Generation requires internet
access and available Gemini quota. The app does not enable provider billing or
switch to a fallback model. Free-tier availability depends on the account.

For the exact previously tested package snapshot, install `requirements-lock.txt`
instead of `requirements.txt`. Developer and browser tools are optional.

## Try the demonstration

1. Click **Load sample notebook**, then **Index files**.
2. Enter `What does loss(w, b) calculate?` and click **Ask**.
3. Expand `[S1]` and compare the explanation with the original code.
4. Try `What does gd return, and does it include the starting point?`.
5. Ask `Which database stores this application's user passwords?`.
   The expected response is `I could not find enough information in the provided files.`

Upload `.ipynb`, `.py`, `.md`, or `.txt` files to replace the sample, then click
**Index files** again. The limit is 10 MiB per file, 30 MiB per corpus, and
50 files. Text must be UTF-8. Changed selections disable questions until indexed.
Invalid replacements preserve the previous index. **Clear active index** resets
the session. Each question is independent.

The original `Knowledge_Source/lab01.ipynb` is included unchanged. The app reads
source text and Markdown without executing the notebook or reading saved plots
and outputs. Its 25 source cells produced 49 chunks in the recorded demonstration.

## Folder guide

| Location | Purpose |
| --- | --- |
| `app.py` | Streamlit application entry point |
| `rag/` | Configuration, loaders, chunking, retrieval, and generation |
| `Knowledge_Source/` | Original demonstration notebook |
| `tests/` | Offline unit and interface checks |
| `scripts/evaluate.py` | Real-model retrieval evaluation and optional live generation |
| `scripts/capture_evidence.py` | Optional browser checks and screenshot capture |
| `screenshots/` | Real application evidence and verification report |
| `presentation/` | PowerPoint and its presentation guide |
| `docs/` | Architecture |
| `.env.example` | Configuration template containing no API key |
| `.streamlit/config.toml` | Local-only server and upload settings |
| `requirements*.txt` | Runtime, development, browser, and locked dependencies |

The app creates `data/models/` and `data/index/` when used. They are generated
runtime files and are not included in the submission. Indexes contain vectors,
exact excerpts, and manifests. Unchanged corpora reuse saved embeddings.
Clearing the active session does not erase these disk caches.

## RAG defaults

| Setting | Default |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Answer model | `gemini-3.5-flash-lite` |
| Chunk budget / overlap | 200 / up to 30 embedding-tokenizer tokens |
| Initial retrieval | 4 distinct chunks |
| Maximum final context | 8 chunks and 1,800 embedding-tokenizer tokens |
| Output limit | 1,200 tokens |

The search uses cosine similarity with a small exact-name ranking preference.
Adjacent notebook context can supplement the initial results. The app assigns
source labels and allows at most one citation repair request. Similarity and
valid labels do not prove that every generated claim is correct. Details are in
the [architecture notes](docs/architecture.md).

## Tests and evaluation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

The recorded verification passed **41 automated tests**, **7 live question
checks**, and **8 browser workflow checks**. The live checks cover target-cell
retrieval, source-label validity, and an exact unsupported refusal. Supported
answers also received manual source review. These counts describe the selected
demonstration, not a general answer-accuracy benchmark.

To evaluate the actual local embedding model:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py
```

Add `--live` to send seven questions to Gemini. This uses the configured API key
and quota, with up to two requests per question if citation repair is needed.
Results are written to `data/evaluation.json`.

To repeat browser evidence capture, install `requirements-browser.txt`, run the
app on port 8502, and use an installed Microsoft Edge browser:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-browser.txt
.\.venv\Scripts\python.exe scripts/capture_evidence.py --live --workflow
```

This replaces screenshots and makes live API requests. Add `--limit 0` to run
only the upload and reset checks. The evidence included in this submission
already records the completed demonstration.

## Privacy and limits

The API key belongs only in a local `.env` file. Asking sends the question and
retrieved excerpts to Google. Google's free-tier terms may allow product
improvement use of that content. Keep private material out of the demonstration.

This is a local, single-user application without authentication or conversation
history. It explains source as written, rather than verifying runtime behavior.
Review source cards when assessing an answer. Missing-key, unavailable-model,
quota, and network errors produce readable messages while preserving the index.

## Planning documents

The original [planning PDF](docs/implementation_plan.pdf) is an archival draft
that preserves its earlier folder names. The [editable brief](docs/implementation_plan.md)
uses the portable paths. Follow this README for current setup instructions.

## Official references

- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Gemini API pricing and terms](https://ai.google.dev/gemini-api/docs/pricing)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Streamlit file uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
