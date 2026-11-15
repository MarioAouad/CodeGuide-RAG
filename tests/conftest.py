"""Deterministic test doubles; unit tests never download a model or call an API."""

import hashlib
import json
import re

import numpy as np
import pytest

from rag.config import Settings


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens + (["CLS", "SEP"] if add_special_tokens else [])


class FakeEmbedder:
    tokenizer = TinyTokenizer()
    max_tokens = 256

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                position = int(hashlib.sha256(word.encode()).hexdigest(), 16) % 64
                vectors[i, position] += 1
            vectors[i, 0] += 0.01
        return vectors


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path):
    return Settings(index_dir=tmp_path / "index")


@pytest.fixture
def sample_notebook(tmp_path):
    """An original test fixture; no private course notebook is needed."""
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": "# Greeting example\nA small function."},
            {
                "cell_type": "code",
                "source": "def greet(name):\n    return f'Hello, {name}'\n",
            },
            {"cell_type": "markdown", "source": "## Example call\nCall the greeting function."},
            {"cell_type": "code", "source": "greet('reader')\n"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "Knowledge_Source" / "lab01.ipynb"
    path.parent.mkdir()
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path
