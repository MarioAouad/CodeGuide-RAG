from dataclasses import replace

import numpy as np
import pytest

from rag.loaders import SourceError
from rag.retrieval import RetrievalError, build_index


def test_cache_reuse_replacement_and_removal(embedder, settings):
    original = {"a.py": b"def apple():\n    return 'fruit'\n", "b.txt": b"oranges"}
    first = build_index(original, embedder, settings)
    count = len(embedder.calls)
    second = build_index(original, embedder, settings)
    assert second.reused and len(embedder.calls) == count
    np.testing.assert_allclose(second.vectors, first.vectors)
    changed = build_index({"a.py": b"def banana():\n    return 'yellow'\n"}, embedder, settings)
    assert not changed.reused
    assert all(chunk.file_name == "a.py" and "apple" not in chunk.text for chunk in changed.chunks)
    changed.search("banana", embedder, settings)
    assert len(embedder.calls[-1]) == 1  # only the question, never the corpus
    other = build_index(original, embedder, replace(settings, chunk_tokens=180))
    assert not other.reused


def test_bad_replacement_preserves_existing_cache(embedder, settings):
    files = {"good.py": b"def ok():\n    return 1"}
    first = build_index(files, embedder, settings)
    with pytest.raises(SourceError):
        build_index({"bad.ipynb": b"{"}, embedder, settings)
    assert first.search("ok", embedder, settings)
    assert build_index(files, embedder, settings).reused


def test_corrupted_cache_is_rebuilt_without_pickle(embedder, settings):
    files = {"a.txt": b"useful text"}
    build_index(files, embedder, settings)
    vector_file = next(settings.index_dir.glob("*/vectors.npy"))
    vector_file.write_bytes(b"corrupt")
    assert not build_index(files, embedder, settings).reused
    np.load(vector_file, allow_pickle=False)


def test_search_bounds_and_validation(embedder, settings):
    files = {f"{i}.txt": ("gradient " * 150).encode() for i in range(12)}
    index = build_index(files, embedder, settings)
    hits = index.search("gradient", embedder, settings)
    assert 1 <= len(hits) <= 8
    assert len({hit.chunk.chunk_id for hit in hits}) == len(hits)
    assert sum(len(embedder.tokenizer.encode(hit.chunk.embedding_text)) for hit in hits) <= 1800
    with pytest.raises(RetrievalError, match="question"):
        index.search(" ", embedder, settings)
    with pytest.raises(RetrievalError, match="Shorten"):
        index.search("hello " * 300, embedder, settings)
