"""Parse uploads as UTF-8 data. Never import or execute their contents."""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath

from rag.models import Source

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_CORPUS_BYTES = 30 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".ipynb", ".py", ".md", ".txt"}


class SourceError(ValueError):
    """A source could not be read safely; the active index may still be used."""


def heading_in(text: str, previous: str | None = None) -> str | None:
    """Find the latest Markdown heading outside fenced code blocks."""
    fence = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            kind = marker[1][0]
            fence = None if fence == kind else kind if fence is None else fence
        elif fence is None:
            match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
            if match:
                previous = match[1]
    return previous


def safe_name(name: str) -> str:
    """Use a display basename only; upload names never become cache paths."""
    result = PurePosixPath(name.replace("\\", "/")).name
    if not result or any(ord(char) < 32 for char in result):
        raise SourceError("A file has an invalid name.")
    return result


def load_file(name: str, content: bytes) -> list[Source]:
    """Return nonempty source cells, ignoring notebook outputs and attachments."""
    name = safe_name(name)
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise SourceError(f"{name}: use .ipynb, .py, .md, or .txt files.")
    if len(content) > MAX_FILE_BYTES:
        raise SourceError(f"{name}: the file exceeds the 10 MiB limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceError(f"{name}: save the file as UTF-8 text.") from exc
    digest = hashlib.sha256(content).hexdigest()
    if suffix != ".ipynb":
        return [Source(name, digest, text)] if text.strip() else []
    try:
        notebook = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise SourceError(f"{name}: malformed notebook JSON.") from exc
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise SourceError(f"{name}: a notebook must contain a cells array.")
    sources = []
    heading = None
    for index, cell in enumerate(notebook["cells"], start=1):
        if not isinstance(cell, dict):
            raise SourceError(f"{name}: cell {index} must be an object.")
        cell_type = cell.get("cell_type")
        if not isinstance(cell_type, str):
            raise SourceError(f"{name}: cell {index} has an invalid cell type.")
        if cell_type not in {"code", "markdown"}:
            continue
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise SourceError(f"{name}: cell {index} has invalid source text.")
        cell_id = cell.get("id")
        if cell_id is not None and not isinstance(cell_id, str):
            raise SourceError(f"{name}: cell {index} has an invalid ID.")
        if cell_type == "markdown":
            heading = heading_in(source, heading)
        if source.strip():
            sources.append(Source(name, digest, source, index, cell_id, cell_type, heading))
    return sources


def load_corpus(files: Mapping[str, bytes]) -> list[Source]:
    """Validate the whole replacement corpus before any index is changed."""
    if not files:
        raise SourceError("Choose at least one file before indexing.")
    if len(files) > 50 or sum(map(len, files.values())) > MAX_CORPUS_BYTES:
        raise SourceError("Use at most 50 files and 30 MiB per corpus.")
    names = [safe_name(name).casefold() for name in files]
    if len(set(names)) != len(names):
        raise SourceError("Each file must have a unique filename.")
    sources = [source for name, content in files.items() for source in load_file(name, content)]
    if not sources:
        raise SourceError("The selected files contain no nonempty code or text cells.")
    return sources
