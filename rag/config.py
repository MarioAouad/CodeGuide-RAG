"""Environment-based configuration. Secrets never enter persisted indexes."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    chunk_tokens: int = 200
    chunk_overlap: int = 30
    context_tokens: int = 1800
    max_output_tokens: int = 1200
    index_dir: Path = ROOT / "data" / "index"

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= 8:
            raise ValueError("TOP_K must be between 1 and 8.")
        if not 64 <= self.chunk_tokens <= 220:
            raise ValueError("CHUNK_TOKENS must be between 64 and 220.")
        if not 0 <= self.chunk_overlap < self.chunk_tokens // 2:
            raise ValueError("CHUNK_OVERLAP must be less than half CHUNK_TOKENS.")
        if self.context_tokens < self.chunk_tokens or self.max_output_tokens < 64:
            raise ValueError("Context and output token budgets are too small.")
        if not self.gemini_model.strip() or not self.embedding_model.strip():
            raise ValueError("Model names cannot be empty.")


def load_settings() -> Settings:
    """Load a local .env without overriding environment variables."""
    load_dotenv(ROOT / ".env", override=False)
    try:
        return Settings(
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=int(os.getenv("TOP_K", "4")),
            chunk_tokens=int(os.getenv("CHUNK_TOKENS", "200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
            context_tokens=int(os.getenv("CONTEXT_TOKENS", "1800")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1200")),
        )
    except ValueError as exc:
        raise ValueError("Invalid configuration. Check numeric limits in .env.") from exc
