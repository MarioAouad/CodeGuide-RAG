from streamlit.testing.v1 import AppTest

from rag.config import ROOT


def test_initial_ui_requires_index(monkeypatch, settings):
    import rag.config

    monkeypatch.setattr(rag.config, "load_settings", lambda: settings)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "CodeGuide"
    assert app.text_area[0].disabled
    assert next(button for button in app.button if button.label == "Ask").disabled
    assert next(button for button in app.button if button.label == "Index files").disabled


def test_sample_index_question_sources_and_clear(monkeypatch, embedder, settings, sample_notebook):
    import rag.config
    import rag.retrieval

    monkeypatch.setattr(rag.config, "load_settings", lambda: settings)
    monkeypatch.setattr(rag.config, "ROOT", sample_notebook.parent.parent)
    monkeypatch.setattr(rag.retrieval, "LocalEmbedder", lambda *_: embedder)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=10).run()

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    click("Load sample notebook")
    click("Index files")
    assert not app.text_area[0].disabled
    app.text_area[0].input("What does greet(name) return?")
    click("Ask")
    assert app.error and "GEMINI_API_KEY" in app.error[0].value
    assert app.code  # Retrieval remains useful even without the generation credential.
    click("Clear active index")
    assert app.text_area[0].disabled
    assert not app.code
