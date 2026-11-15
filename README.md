# CodeGuide — code explanation with RAG

CodeGuide answers questions about uploaded Python code, notebooks, and notes. It creates embeddings and retrieves excerpts locally, then sends the question and selected excerpts to Gemini to generate an answer with source labels.

## Repository contents

~~~text
CodeGuide-RAG/
├── app.py                    Streamlit application
├── rag/                      Configuration, loading, chunking, retrieval, generation
├── tests/                    Offline tests with synthetic data and test doubles
├── scripts/evaluate.py       Optional course-notebook evaluation
├── scripts/capture_evidence.py  Optional course-notebook browser capture
├── docs/architecture.md      Implementation notes
├── .env.example              Configuration template with an empty API key
├── .streamlit/config.toml    Local server and upload settings
├── requirements.txt          Runtime dependency ranges
├── requirements-dev.txt      Test and lint dependencies
├── requirements-browser.txt  Optional browser automation dependencies
├── requirements-lock.txt     Recorded dependency snapshot
└── pyproject.toml            Test and formatting configuration
~~~

Presentations, screenshots, recorded answers, planning PDFs, and a validation report are not included in this public code repository.

The original course notebook `Knowledge_Source/lab01.ipynb` is also excluded. Use your own uploads for the demonstration below. The **Load sample notebook** button and notebook-specific example questions in the application refer to that original notebook; they are not the starting workflow for this public repository.

## Install and run

Use Python 3.11 or 3.12. On Windows, clone the repository and open PowerShell in the folder containing `app.py`:

~~~powershell
git clone https://github.com/MarioAouad/CodeGuide-RAG.git
cd CodeGuide-RAG
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Alternatively, use GitHub's **Code → Download ZIP**, extract it, and start with the environment creation command inside the extracted folder. Use the full path to a Python interpreter if `python` opens the Microsoft Store.

Edit the local `.env` and set `GEMINI_API_KEY`. If `.env` already exists, keep your settings instead of overwriting it. The configured model is `gemini-3.5-flash-lite`; change `GEMINI_MODEL` if your Google project requires a different supported model. Model access, quota, and billing depend on that project. Environment variables take precedence over `.env`.

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
~~~

Open <http://127.0.0.1:8502>. On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

The first indexing run downloads `sentence-transformers/all-MiniLM-L6-v2`. Embeddings then run locally on the CPU. Generating answers requires internet access, a valid Gemini key, and available quota. Indexing and retrieval can operate without a generation key.

## Demonstrate with your own file

Create a file named `greeting.py` containing:

~~~python
def greet(name):
    return f"Hello, {name}"
~~~

1. Upload `greeting.py` using **Upload code or notes**.
2. Click **Index files** and wait for the ready message.
3. Enter **What does greet(name) return?** and click **Ask**.
4. Expand a retrieved source such as **[S1]** and compare the answer with the code.
5. Upload a changed file and index it again, or click **Clear active index** to reset the session.

You can upload `.ipynb`, `.py`, `.md`, and `.txt` files. Limits are 10 MiB per file, 30 MiB per corpus, and 50 files. Text must be UTF-8, and filenames must be unique. The notebook loader reads code and Markdown cells without executing them or using saved outputs.

Changed selections block new questions until indexed. Invalid replacements preserve the previous in-memory index. Each question is independent; there is no conversation history.

## RAG defaults

| Setting | Default |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Answer model | `gemini-3.5-flash-lite` |
| Chunk budget / overlap | 200 / up to 30 embedding-tokenizer tokens |
| Initial retrieval | Up to 4 distinct chunks |
| Maximum final context | 8 chunks and 1,800 embedding-tokenizer tokens |
| Output limit | 1,200 tokens |

Retrieval uses cosine similarity with a small exact-name ranking preference. Adjacent notebook cells can supplement results. The generation adapter validates source labels and permits one citation repair attempt. Valid citations do not prove that an explanation is correct; review the actual source excerpts. See [implementation notes](docs/architecture.md).

## Run offline tests

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
~~~

The tests create their own small notebook fixture and use deterministic embedding doubles and mocked generation requests. They do not need the excluded course notebook, an API key, a downloaded embedding model, or live Gemini calls. Passing tests verify the covered behavior, not general answer accuracy.

`requirements-lock.txt` is a recorded dependency snapshot. It can be installed instead of `requirements.txt` to reproduce those package versions, subject to platform availability.

## Optional course-specific scripts

`scripts/evaluate.py` and `scripts/capture_evidence.py` preserve the original course demonstration workflow. Both require a local copy of `Knowledge_Source/lab01.ipynb`, which is not provided here. Their questions, target cell numbers, and capture expectations are specific to that notebook and do not apply to arbitrary uploads.

The evaluation script downloads/loads the real embedding model. Adding `--live` sends its questions to Gemini and consumes quota. It writes `data/evaluation.json`.

The browser capture script requires `requirements-browser.txt`, Microsoft Edge, a running app, and `--live`. It can send questions to Gemini and writes new images and a report under `screenshots/`. Those generated artifacts are not included in this repository. Use the upload demonstration above for a fresh public checkout.

## Local data and privacy

Runtime model files are stored in `data/models/`; indexes under `data/index/` contain vectors, source excerpts, and manifests. Identical corpora can reuse saved embeddings. **Clear active index** resets the session but does not delete disk caches. Stop the application and delete `data/index/` if you want to remove cached source excerpts; delete `data/models/` separately if you also want to remove downloaded models.

Keep the API key in a local `.env` file. Asking sends the question and retrieved excerpts to Google; review the terms applicable to your account before using private files. Missing-key, model, quota, and connection failures are displayed as readable errors.

This is a local, single-user application without authentication. Keep it on localhost. It explains source text and does not execute or verify uploaded programs.

## References

- [Implementation notes](docs/architecture.md)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
