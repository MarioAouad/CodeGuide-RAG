from rag.chunking import chunk_sources, token_count
from rag.loaders import load_file
from rag.models import Source


def test_sample_chunk_locations_and_token_budget(embedder, sample_notebook):
    sources = load_file(sample_notebook.name, sample_notebook.read_bytes())
    chunks = chunk_sources(sources, embedder.tokenizer)
    originals = {source.cell_index: source.text for source in sources}
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.cell_index for chunk in chunks} == set(range(1, 5))
    for chunk in chunks:
        text = originals[chunk.cell_index]
        assert chunk.text == text[chunk.start_offset : chunk.end_offset]
        assert chunk.start_line == text.count("\n", 0, chunk.start_offset) + 1
        assert chunk.end_line == text.count("\n", 0, chunk.end_offset - 1) + 1
        assert token_count(embedder.tokenizer, chunk.embedding_text) <= 200
        assert chunk.text in "".join(
            text.splitlines(keepends=True)[chunk.start_line - 1 : chunk.end_line]
        )


def test_long_lines_are_not_silently_truncated(embedder):
    text = " ".join(f"token_{i}" for i in range(900))
    source = Source("long.txt", "hash", text, heading="heading " * 1000)
    chunks = chunk_sources([source], embedder.tokenizer, 100, 20)
    covered = set()
    for chunk in chunks:
        assert chunk.start_line == chunk.end_line == 1
        assert token_count(embedder.tokenizer, chunk.embedding_text) <= 100
        covered.update(range(chunk.start_offset, chunk.end_offset))
    assert covered == set(range(len(text)))


def test_long_file_keeps_function_boundaries(embedder):
    text = "def first():\n" + "    x = 1\n" * 100 + "\ndef second():\n    return 2\n"
    chunks = chunk_sources([Source("code.py", "hash", text)], embedder.tokenizer, 100, 20)
    second = [chunk for chunk in chunks if chunk.function_name == "second"]
    assert second and "def second" in second[0].text
    assert all("def second" not in chunk.text for chunk in chunks if chunk.function_name == "first")
