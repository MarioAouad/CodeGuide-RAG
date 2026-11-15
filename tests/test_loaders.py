import hashlib
import json

import pytest

from rag.loaders import SourceError, heading_in, load_corpus, load_file


def test_sample_inventory_and_unchanged_bytes(sample_notebook):
    path = sample_notebook
    before = path.read_bytes()
    sources = load_file(path.name, before)
    assert len(sources) == 4
    assert sum(item.cell_type == "code" for item in sources) == 2
    assert sum(item.cell_type == "markdown" for item in sources) == 2
    assert [item.cell_index for item in sources] == list(range(1, 5))
    assert "def greet(name):" in sources[1].text
    assert sources[1].heading == "Greeting example"
    assert path.read_bytes() == before
    assert all(item.file_hash == hashlib.sha256(before).hexdigest() for item in sources)


def test_notebook_sources_only_with_true_cell_positions():
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["## Heading\n", "Some explanation"]},
            {"cell_type": "raw", "source": "RAW_SENTINEL"},
            {"cell_type": "code", "source": "\n"},
            {
                "cell_type": "code",
                "id": "stable",
                "execution_count": 999,
                "source": ["dangerous_call()\n"],
                "outputs": [{"text": "OUTPUT_SENTINEL", "data": {"image/png": "IMAGE_SENTINEL"}}],
            },
        ]
    }
    sources = load_file("../demo.ipynb", json.dumps(notebook).encode())
    assert [item.cell_index for item in sources] == [1, 4]
    assert sources[1].cell_id == "stable"
    assert sources[1].heading == "Heading"
    assert sources[1].text == "dangerous_call()\n"
    assert "SENTINEL" not in "".join(item.text for item in sources)


@pytest.mark.parametrize(
    "content",
    [
        b"{",
        b"[]",
        b'{"cells":null}',
        b'{"cells":[3]}',
        b'{"cells":[{"cell_type":"code","source":[4]}]}',
        b"\xff",
    ],
)
def test_malformed_notebook_is_readable_error(content):
    with pytest.raises(SourceError):
        load_file("broken.ipynb", content)


def test_plain_utf8_and_no_execution(tmp_path):
    marker = tmp_path / "executed.txt"
    text = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
    source = load_file("sample.py", text.encode())[0]
    assert source.text == text
    assert source.cell_index is None and source.cell_type is None
    assert not marker.exists()


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"a.exe": b"hello"},
        {"a.txt": b" "},
        {"dir/a.py": b"x=1", "a.py": b"x=2"},
        {"a.txt": b"x" * (10 * 1024 * 1024 + 1)},
    ],
)
def test_corpus_validation(files):
    with pytest.raises(SourceError):
        load_corpus(files)


def test_heading_ignores_fenced_code():
    assert heading_in("## Real\n```python\n# Comment\n```\ntext") == "Real"
