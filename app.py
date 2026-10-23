"""Streamlit app: casting quality inspection (acceptable vs defective) with a human-review zone.

    streamlit run app.py        ->  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Casting inspection", page_icon="⚙️", layout="wide")

from app_lib.ui import setup  # noqa: E402  (set_page_config must run first)

setup()
pages = [
    st.Page("app_pages/inspect.py", title="Inspect a part", url_path="inspect", default=True),
    st.Page("app_pages/results.py", title="Results", url_path="results"),
    st.Page("app_pages/review_queue.py", title="Review queue", url_path="review"),
    st.Page("app_pages/limits.py", title="Limits", url_path="limits"),
    st.Page("app_pages/data_story.py", title="The data", url_path="data"),
]
st.navigation(pages, position="top").run()
