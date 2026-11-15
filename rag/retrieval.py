"""Local semantic search with a small, content-addressed NumPy cache."""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag.chunking import chunk_sources, token_count
from rag.config import ROOT, Settings
from rag.loaders import load_corpus, safe_name
from rag.models import Chunk, Hit

CACHE_VERSION = 1


class RetrievalError(ValueError):
    """A readable indexing or retrieval error."""


class LocalEmbedder:
    """Load the model once. Subsequent encoding uses local CPU inference."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=str(ROOT / "data" / "models"),
            trust_remote_code=False,
        )
        self.tokenizer = self.model.tokenizer
        self.max_tokens = self.model.max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )


def _normalize(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise RetrievalError("The embedding model returned invalid vectors.")
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise RetrievalError("The embedding model returned an empty vector.")
    return array / lengths


def corpus_manifest(files: dict[str, bytes], settings: Settings) -> dict:
    """All inputs that change chunking or vectors participate in the cache key."""
    return {
        "version": CACHE_VERSION,
        "files": sorted(
            [
                {"name": safe_name(name), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()
            ],
            key=lambda item: item["name"],
        ),
        "embedding_model": settings.embedding_model,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
    }


@dataclass
class VectorIndex:
    """An active corpus, independent of newer selections or failed replacements."""

    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict
    reused: bool = False

    def search(self, question: str, embedder: Any, settings: Settings) -> list[Hit]:
        """Rank distinct excerpts, then add adjacent notebook context if it fits."""
        question = question.strip()
        if not question:
            raise RetrievalError("Enter a question first.")
        if token_count(embedder.tokenizer, question) > embedder.max_tokens:
            raise RetrievalError(f"Shorten the question to {embedder.max_tokens} embedding tokens.")
        query = _normalize(embedder.encode([question]))
        if query.shape[1] != self.vectors.shape[1]:
            raise RetrievalError("Embedding dimensions changed. Index the files again.")
        scores = self.vectors @ query[0]
        ranked = scores.copy()
        # A small exact identifier preference helps short code names such as gd.
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", question))
        for i, chunk in enumerate(self.chunks):
            names = set(re.findall(r"\b(?:def|class)\s+(\w+)", chunk.text))
            if names & identifiers:
                ranked[i] += 0.08
        order = np.argsort(-ranked, kind="stable")
        selected: list[int] = []
        seen_text: set[tuple[str, int | None, str]] = set()
        budget = settings.context_tokens

        def add(i: int) -> bool:
            nonlocal budget
            chunk = self.chunks[i]
            identity = (chunk.file_name, chunk.cell_index, chunk.text)
            cost = token_count(embedder.tokenizer, chunk.embedding_text)
            if i in selected or identity in seen_text or cost > budget or len(selected) >= 8:
                return False
            selected.append(i)
            seen_text.add(identity)
            budget -= cost
            return True

        for i in order:
            add(int(i))
            if len(selected) >= settings.top_k:
                break
        # Prefer an immediately adjacent cell of the opposite type: prose explains code.
        for i in list(selected):
            chunk = self.chunks[i]
            if chunk.cell_index is None:
                continue
            for neighbor in (chunk.cell_index - 1, chunk.cell_index + 1):
                candidates = [
                    j
                    for j, other in enumerate(self.chunks)
                    if other.file_name == chunk.file_name
                    and other.cell_index == neighbor
                    and other.cell_type != chunk.cell_type
                ]
                if candidates:
                    add(max(candidates, key=lambda j: float(ranked[j])))
        return [Hit(self.chunks[i], float(scores[i])) for i in selected]


def _load_cache(directory: Path, expected: dict) -> VectorIndex | None:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["inputs"] != expected:
            return None
        metadata = (directory / "chunks.json").read_bytes()
        vector_bytes = (directory / "vectors.npy").read_bytes()
        if (
            hashlib.sha256(metadata).hexdigest() != manifest["chunks_sha256"]
            or hashlib.sha256(vector_bytes).hexdigest() != manifest["vectors_sha256"]
        ):
            return None
        chunks = [Chunk(**item) for item in json.loads(metadata)]
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            return None
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            return None
        return VectorIndex(chunks, _normalize(vectors), expected, reused=True)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError):
        return None


def build_index(files: dict[str, bytes], embedder: Any, settings: Settings) -> VectorIndex:
    """Build or reuse a complete replacement; callers commit only on success."""
    sources = load_corpus(files)
    if settings.chunk_tokens > embedder.max_tokens:
        raise RetrievalError("CHUNK_TOKENS exceeds this embedding model's input limit.")
    inputs = corpus_manifest(files, settings)
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    directory = settings.index_dir / fingerprint
    cached = _load_cache(directory, inputs)
    if cached is not None:
        return cached
    chunks = chunk_sources(
        sources, embedder.tokenizer, settings.chunk_tokens, settings.chunk_overlap
    )
    if not chunks:
        raise RetrievalError("No searchable source text was found.")
    vectors = _normalize(embedder.encode([chunk.embedding_text for chunk in chunks]))
    if vectors.shape[0] != len(chunks):
        raise RetrievalError("The embedding model returned the wrong number of vectors.")
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    # Stage all files, then publish the manifest last. Interrupted writes fail checksum
    # validation on the next load; no partially built index reaches the UI.
    with tempfile.TemporaryDirectory(dir=settings.index_dir) as temporary:
        stage = Path(temporary)
        metadata = json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False).encode()
        (stage / "chunks.json").write_bytes(metadata)
        np.save(stage / "vectors.npy", vectors, allow_pickle=False)
        manifest = {
            "inputs": inputs,
            "chunks_sha256": hashlib.sha256(metadata).hexdigest(),
            "vectors_sha256": hashlib.sha256((stage / "vectors.npy").read_bytes()).hexdigest(),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        directory.mkdir(exist_ok=True)
        for name in ("chunks.json", "vectors.npy", "manifest.json"):
            os.replace(stage / name, directory / name)
    return VectorIndex(chunks, vectors, inputs)
