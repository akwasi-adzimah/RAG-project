"""Smoke test: the Streamlit app renders against the real (cached) index.
Only the initial page is exercised, so no API calls are made."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "rag_system" / "app.py"


def test_app_renders_library_and_examples():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception

    assert at.title[0].value == "Ask your documents"
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "state-of-code-developer-survey-report.pdf" in sidebar_text
    assert [r.value for r in at.sidebar.radio] == ["hybrid"]
    example_buttons = [b for b in at.button if b.key and b.key.startswith("ex_")]
    assert len(example_buttons) == 4
