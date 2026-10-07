"""Run every Streamlit page with Streamlit's own AppTest runner.

Skipped automatically when Streamlit or Plotly is not installed.
"""

from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]


def test_every_page_runs_without_errors():
    for page in PAGES:
        app = testing.AppTest.from_file(str(page), default_timeout=180)
        app.run()
        assert not app.exception, f"{page.name}: {app.exception}"
