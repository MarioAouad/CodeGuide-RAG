"""Small immutable records shared by the RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """One notebook cell or one plain file, with its original source text."""

    file_name: str
    file_hash: str
    text: str
    cell_index: int | None = None
    cell_id: str | None = None
    cell_type: str | None = None
    heading: str | None = None


@dataclass(frozen=True)
class Chunk:
    """An exact source substring; offsets are zero-based, lines are one-based."""

    chunk_id: str
    file_name: str
    file_hash: str
    cell_index: int | None
    cell_id: str | None
    cell_type: str | None
    heading: str | None
    start_line: int
    end_line: int
    text: str
    embedding_text: str
    start_offset: int
    end_offset: int
    function_name: str | None = None

    @property
    def location(self) -> str:
        """Human-readable location, without confusing cell positions and run counts."""
        parts = [self.file_name]
        if self.cell_index is not None:
            parts.append(f"cell {self.cell_index}")
        if self.function_name:
            parts.append(self.function_name)
        parts.append(f"lines {self.start_line}-{self.end_line}")
        return " - ".join(parts)


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its cosine similarity (not a confidence probability)."""

    chunk: Chunk
    score: float
