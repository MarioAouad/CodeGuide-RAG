"""Run with `streamlit run app.py` from the project directory."""

import streamlit as st

from rag.config import ROOT, load_settings
from rag.generation import GenerationError, answer_question
from rag.loaders import SourceError, load_corpus
from rag.retrieval import LocalEmbedder, RetrievalError, build_index, corpus_manifest

EXAMPLES = [
    "What does loss(w, b) calculate?",
    "How is the gradient calculated and checked?",
    "What does gd return, and does it include the starting point?",
    "Why does the notebook say a learning rate of 0.04 diverges?",
    "How does classify decide whether a stationary point is a saddle?",
    "How does the optional PyTorch section obtain gradients?",
    "Which database stores this application's user passwords?",
]


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> LocalEmbedder:
    """Streamlit reruns share the CPU model, not users' questions or active corpora."""
    return LocalEmbedder(model_name)


def main() -> None:
    st.set_page_config(page_title="CodeGuide", page_icon="📖", layout="wide")
    st.title("CodeGuide")
    st.caption("Ask about your code. Read the evidence behind each answer.")
    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    state = st.session_state
    state.setdefault("index", None)
    state.setdefault("sample_selected", False)
    state.setdefault("result", None)
    state.setdefault("uploader_version", 0)

    with st.sidebar:
        st.header("Knowledge sources")
        if st.button("Load sample notebook", width="stretch"):
            state.sample_selected = True
            state.uploader_version += 1
            state.result = None
        uploaded = st.file_uploader(
            "Upload code or notes",
            type=["ipynb", "py", "md", "txt"],
            accept_multiple_files=True,
            key=f"uploads_{state.uploader_version}",
            help="UTF-8 files, up to 10 MiB each and 30 MiB in total. Uploads replace the sample.",
        )
        files = {}
        selection_error = None
        if uploaded:
            state.sample_selected = False
            names = [item.name for item in uploaded]
            if len(names) != len(set(names)):
                selection_error = "Each uploaded file must have a unique filename."
            else:
                files = {item.name: item.getvalue() for item in uploaded}
        elif state.sample_selected:
            try:
                files = {"lab01.ipynb": (ROOT / "Knowledge_Source" / "lab01.ipynb").read_bytes()}
                st.caption("Selected: lab01.ipynb")
            except OSError:
                selection_error = "The sample notebook could not be read. Upload a file instead."

        if selection_error:
            st.error(selection_error)
        index_clicked = st.button(
            "Index files", disabled=not files or bool(selection_error), width="stretch"
        )
        if index_clicked:
            try:
                # Validate before downloading the embedding model or changing the active index.
                sources = load_corpus(files)
                with st.spinner(
                    "Indexing locally (the first model download may take a few minutes)…"
                ):
                    embedder = get_embedder(settings.embedding_model)
                    new_index = build_index(files, embedder, settings)
                state.index = new_index
                state.result = None
                reuse = "Reused saved embeddings" if new_index.reused else "Created embeddings"
                st.success(f"{reuse}: {len(sources)} sources, {len(new_index.chunks)} chunks.")
            except (SourceError, RetrievalError) as exc:
                st.error(str(exc))
            except OSError:
                st.error(
                    "Could not read the model or write the index. Check access and disk space."
                )
            except Exception:
                # Provider/model exceptions may contain raw source or credentials.
                st.error(
                    "Indexing failed. Check the model name, connection, and setup instructions."
                )

        if st.button("Clear active index", disabled=state.index is None):
            state.index = None
            state.result = None
            state.sample_selected = False
            state.uploader_version += 1
            st.rerun()
        if state.index is not None:
            st.subheader("Indexed files")
            for item in state.index.manifest["files"]:
                st.text(item["name"])
        st.divider()
        st.caption(
            "Indexing stays local. Asking sends the question and retrieved excerpts to Google. "
            "Free-tier content may be used to improve Google's products. "
            "Use a free-tier project without paid billing."
        )

    active = state.index
    pending = bool(selection_error) or (
        active is not None and corpus_manifest(files, settings) != active.manifest
    )
    ready = active is not None and not pending
    if active is None:
        st.info("Load the sample or upload files, then click Index files to begin.")
    elif pending:
        state.result = None
        st.warning("The selected files changed. Index them before asking about this selection.")
    else:
        st.success(f"Ready · {len(active.chunks)} searchable chunks")
    if not settings.api_key:
        st.info(
            "Retrieval works without an API key. Set GEMINI_API_KEY locally to generate answers."
        )

    with st.expander("Example questions"):
        for example in EXAMPLES:
            if st.button(example, disabled=not ready):
                state.question = example
    with st.form("question_form"):
        question = st.text_area("Your question", key="question", disabled=not ready, max_chars=4000)
        submitted = st.form_submit_button("Ask", disabled=not ready)
    if submitted and ready:
        state.result = None
        hits = []
        try:
            with st.spinner("Retrieving evidence and preparing an answer…"):
                embedder = get_embedder(settings.embedding_model)
                hits = active.search(question, embedder, settings)
                answer = answer_question(question, hits, settings)
            state.result = (answer, hits, None)
        except (GenerationError, RetrievalError) as exc:
            state.result = (None, hits, str(exc))
        except Exception:
            state.result = (None, hits, "The request failed. Check your setup and try again.")

    if state.result and ready:
        answer, hits, error = state.result
        if error:
            st.error(error)
        elif answer:
            st.subheader("Answer")
            # Plain text keeps model-generated HTML, external images and links inert.
            st.text(answer)
        if hits:
            st.subheader("Retrieved sources")
            st.caption("Similarity ranks excerpts; it does not measure answer correctness.")
            for number, hit in enumerate(hits, start=1):
                with st.expander(f"[S{number}] {hit.chunk.location}"):
                    st.caption(f"Cosine similarity: {hit.score:.3f}")
                    language = (
                        "python"
                        if (hit.chunk.cell_type == "code" or hit.chunk.file_name.endswith(".py"))
                        else "text"
                    )
                    st.code(hit.chunk.text, language=language, wrap_lines=True)


if __name__ == "__main__":
    main()
