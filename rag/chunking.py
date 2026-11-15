"""Token-aware, overlapping chunks that retain exact original source spans."""

import ast
import hashlib
import re
from typing import Any

from rag.models import Chunk, Source


def token_count(tokenizer: Any, text: str) -> int:
    """Count actual embedding tokens, including the model's special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def _sections(source: Source) -> list[tuple[int, int, str | None]]:
    """Prefer top-level function boundaries when valid Python can be parsed."""
    text = source.text
    if source.cell_type != "code" and not source.file_name.lower().endswith(".py"):
        return [(0, len(text), None)]
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return [(0, len(text), None)]
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    boundaries = [(0, None)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            line = min([node.lineno] + [item.lineno for item in node.decorator_list])
            start = offsets[line - 1]
            if start == 0:
                boundaries[0] = (0, node.name)
            else:
                boundaries.append((start, node.name))
    return [
        (start, boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text), name)
        for i, (start, name) in enumerate(boundaries)
    ]


def chunk_sources(
    sources: list[Source], tokenizer: Any, max_tokens: int = 200, overlap: int = 30
) -> list[Chunk]:
    """Split at lines when possible, and characters for exceptionally long lines.

    The size limit includes heading/function context and special tokens. Offsets
    make even fragments of a single long line independently verifiable.
    """
    if max_tokens < 64 or not 0 <= overlap < max_tokens // 2:
        raise ValueError("Invalid chunk size or overlap.")
    chunks = []
    for source in sources:
        sections = _sections(source)
        # Preserve a whole short notebook cell, even if it defines several functions.
        short_prefix = f"{source.heading or ''}\n"
        if token_count(tokenizer, short_prefix + source.text) <= max_tokens:
            names = [name for _, _, name in sections if name]
            sections = [(0, len(source.text), ", ".join(names) or None)]
        for section_start, section_end, function_name in sections:
            heading = source.heading or ""
            prefix = f"{heading}\n{function_name or ''}\n"
            # Very long source headings must not consume the entire token budget.
            while token_count(tokenizer, prefix) > max_tokens // 4:
                prefix = prefix[: len(prefix) // 2]
            start = section_start
            while start < section_end:
                low, high = start + 1, section_end
                end = start
                while low <= high:
                    middle = (low + high) // 2
                    if token_count(tokenizer, prefix + source.text[start:middle]) <= max_tokens:
                        end, low = middle, middle + 1
                    else:
                        high = middle - 1
                if end == start:
                    raise ValueError("Chunk budget cannot hold one source character.")
                if end < section_end:
                    newline = source.text.rfind("\n", start, end)
                    if newline >= start:
                        end = newline + 1
                text = source.text[start:end]
                if text.strip():
                    identity = (
                        f"{source.file_name}:{source.file_hash}:{source.cell_index}:{start}:{end}"
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=hashlib.sha256(identity.encode()).hexdigest(),
                            file_name=source.file_name,
                            file_hash=source.file_hash,
                            cell_index=source.cell_index,
                            cell_id=source.cell_id,
                            cell_type=source.cell_type,
                            heading=source.heading,
                            start_line=source.text.count("\n", 0, start) + 1,
                            end_line=source.text.count("\n", 0, max(start, end - 1)) + 1,
                            text=text,
                            embedding_text=prefix + text,
                            start_offset=start,
                            end_offset=end,
                            function_name=function_name,
                        )
                    )
                if end == section_end:
                    break
                next_start = end
                if overlap:
                    # Walk back whole lines while the overlap stays within budget.
                    for match in reversed(list(re.finditer(r"(?m)^.+", text))):
                        candidate = start + match.start()
                        if candidate <= start:
                            break
                        if token_count(tokenizer, source.text[candidate:end]) > overlap:
                            break
                        next_start = candidate
                start = max(start + 1, next_start)
    return chunks
