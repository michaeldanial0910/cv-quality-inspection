"""Smoke tests: every page renders on the real outputs, and the controls change what is shown.

Needs models/ and the test images in data/raw/ (skipped otherwise).

    pytest -q tests/test_app.py
"""
import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODELS, REPORTS

pytestmark = pytest.mark.skipif(not (MODELS / "model.onnx").exists() or not (REPORTS / "summary.json").exists(),
                                reason="needs the trained model and reports")
PAGES = ["../app_pages/inspect.py", "../app_pages/results.py", "../app_pages/review_queue.py",
         "../app_pages/limits.py", "../app_pages/data_story.py"]


def html(at):
    return "".join(h.proto.body for h in at.get("html"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(page, default_timeout=120).run()
    assert not at.exception, at.exception


def test_presets_route_to_different_zones():
    at = AppTest.from_file("../app_pages/inspect.py", default_timeout=120).run()
    assert "Auto-reject" in html(at)
    at.segmented_control(key="source").set_value("Clear pass").run()
    assert "Auto-pass" in html(at)
    at.segmented_control(key="source").set_value("Borderline").run()
    assert "Send to a person" in html(at)


def test_drift_warning_appears():
    at = AppTest.from_file("../app_pages/inspect.py", default_timeout=120).run()
    assert "You changed the photo" not in html(at)
    at.slider(key="blur").set_value(3.0).run()
    assert "You changed the photo" in html(at)


def test_threshold_slider_trades_misses_for_scrap():
    at = AppTest.from_file("../app_pages/results.py", default_timeout=120).run()
    at.slider(key="thr").set_value(0.99).run()
    high = html(at)
    at.slider(key="thr").set_value(0.0).run()
    low = html(at)
    assert high != low and "good parts that would be scrapped" in low


def test_review_queue_filter():
    at = AppTest.from_file("../app_pages/review_queue.py", default_timeout=120).run()
    n_all = html(at).count('class="tile')
    at.toggle(key="only_def").set_value(True).run()
    n_def = html(at).count('class="tile')
    assert n_all > n_def > 0
