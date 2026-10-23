"""Shared look, cached data and small HTML components for the Streamlit app."""
import base64
import html
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import EXTRACTED, MANIFEST, MODELS, REPORTS

# ---- design tokens (same family as the churn / fraud / sentiment / recommender apps) ----------
INK = "#1C2333"
INK_SOFT = "#5A6478"
RULE = "#E3E6EC"
PAPER = "#F7F8FA"
ZONE_COLOR = {"auto-pass": "#13976A", "human review": "#B07D00", "auto-reject": "#D4562E"}
ZONE_TEXT = {"auto-pass": "Auto-pass", "human review": "Send to a person", "auto-reject": "Auto-reject"}
MODEL_COLOR = {"hog_lr": "#8A94A6", "linear_probe": "#2A6FD6", "finetuned": "#13976A"}
MODEL_NAME = {"hog_lr": "A · HOG + logistic regression", "linear_probe": "B · frozen EfficientNet-B0",
              "finetuned": "C · fine-tuned EfficientNet-B0"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, td, th, input, select,
button p, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 4.2rem; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.3rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.4rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.08rem; }}
p, li {{ line-height: 1.6; }}
.lede {{ color: {INK_SOFT}; font-size: 1.06rem; max-width: 74ch; margin-top: -0.4rem; }}
.muted {{ color: {INK_SOFT}; font-size: 0.9rem; }}

.verdict {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: 18px 20px; }}
.verdict .zone {{ display: inline-block; font-weight: 600; font-size: 1.05rem; color: #fff; border-radius: 999px;
                  padding: 5px 16px; }}
.verdict .p {{ font-size: 2.6rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; margin-top: 14px; }}
.verdict .pl {{ color: {INK_SOFT}; font-size: 0.88rem; }}
.verdict .row {{ display: flex; justify-content: space-between; border-top: 1px solid {RULE}; padding: 8px 0 0;
                 margin-top: 10px; font-size: 0.9rem; font-variant-numeric: tabular-nums; }}
.verdict .row span:first-child {{ color: {INK_SOFT}; }}
.scale {{ position: relative; height: 34px; margin: 16px 0 6px; }}
.scale .band {{ position: absolute; top: 10px; height: 10px; }}
.scale .pin {{ position: absolute; top: 2px; width: 3px; height: 26px; background: {INK}; border-radius: 2px; }}
.scale .tick {{ position: absolute; top: 24px; font-size: 0.72rem; color: {INK_SOFT}; transform: translateX(-50%); }}
.truth {{ font-size: 0.85rem; padding: 3px 10px; border-radius: 999px; background: #EEF0F4; color: {INK}; display: inline-block; }}
.truth.wrong {{ background: #FBE7E1; color: #9C3418; }}

.tile {{ background: #fff; border: 1px solid {RULE}; border-radius: 10px; padding: 8px; margin-bottom: 12px; }}
.tile img {{ width: 100%; border-radius: 6px; display: block; }}
.tile .cap {{ font-size: 0.8rem; color: {INK_SOFT}; margin-top: 6px; line-height: 1.4; }}
.tile .cap b {{ color: {INK}; font-weight: 600; }}
.tile.def {{ box-shadow: inset 0 3px 0 #D4562E; }}

.figure-row {{ display: flex; gap: 28px; flex-wrap: wrap; margin: 10px 0 4px; }}
.figure {{ min-width: 150px; }}
.figure .n {{ font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.figure .l {{ font-size: 0.86rem; color: {INK_SOFT}; max-width: 24ch; }}
.callout {{ border-left: 4px solid {INK}; background: #fff; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 12px 0; }}
.callout.warn {{ border-left-color: #B07D00; }}
.callout.bad {{ border-left-color: #D4562E; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 5px; vertical-align: 0; }}
.legend {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.76rem; white-space: nowrap; color: {INK_SOFT}; margin: 2px 0 10px; }}
@media (max-width: 640px) {{ h1 {{ font-size: 1.8rem; }} .figure .n {{ font-size: 1.6rem; }} }}
</style>
"""


def setup():
    st.html(CSS)


def esc(s) -> str:
    return html.escape(str(s))


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def figures(items) -> str:
    cells = "".join(f'<div class="figure"><div class="n">{esc(n)}</div><div class="l">{esc(l)}</div></div>'
                    for n, l in items)
    return f'<div class="figure-row">{cells}</div>'


def zone_legend() -> str:
    return '<div class="legend">' + "".join(f"<span>{dot(c)}{ZONE_TEXT[z]}</span>" for z, c in ZONE_COLOR.items()) + "</div>"


def img_uri(img: Image.Image, size: int = 320) -> str:
    b = io.BytesIO()
    im = img.copy()
    im.thumbnail((size, size))
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def tile(img: Image.Image, caption_html: str, defect: bool = False) -> str:
    return (f'<div class="tile{" def" if defect else ""}"><img src="{img_uri(img)}" alt="casting"/>'
            f'<div class="cap">{caption_html}</div></div>')


def scale(prob: float, review_from: float, reject_from: float) -> str:
    """Probability bar with the three zones drawn to a square-root scale (the review band is narrow)."""
    def x(p):
        return 100 * float(np.sqrt(np.clip(p, 0, 1)))
    bands = [(0, review_from, "auto-pass"), (review_from, reject_from, "human review"), (reject_from, 1, "auto-reject")]
    out = "".join(f'<div class="band" style="left:{x(a):.2f}%;width:{x(b) - x(a):.2f}%;background:{ZONE_COLOR[z]};'
                  f'opacity:.85"></div>' for a, b, z in bands)
    ticks = "".join(f'<div class="tick" style="left:{x(t):.2f}%">{t:g}</div>' for t in [0, 0.05, 0.25, 0.5, 1])
    return f'<div class="scale">{out}<div class="pin" style="left:calc({x(prob):.2f}% - 1px)"></div>{ticks}</div>'


# ---- data ----------------------------------------------------------------------------------
@st.cache_resource
def inspector():
    from src.infer import Inspector
    return Inspector.load() if (MODELS / "model.onnx").exists() else None


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def policy() -> dict:
    p = MODELS / "policy.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def test_images() -> pd.DataFrame:
    """Test-split images that exist on this machine, with the served model's calibrated scores."""
    m = pd.read_csv(MANIFEST)
    t = m[m["split"] == "test"][["path", "label"]]
    pred = load_csv("predictions/finetuned_calibrated.csv")
    if pred is not None:
        t = t.merge(pred.loc[pred["split"] == "test", ["path", "prob"]], on="path", how="left")
    return t[[(EXTRACTED / p).exists() for p in t["path"]]].reset_index(drop=True)


def load_image(path: str) -> Image.Image:
    return Image.open(EXTRACTED / path).convert("L")


@st.cache_data(show_spinner=False)
def heat_overlay(path: str) -> tuple[Image.Image, float, str]:
    from src.infer import overlay
    ins = inspector()
    r = ins.inspect(EXTRACTED / path)
    return overlay(EXTRACTED / path, r.heatmap), r.probability, r.zone


def need(what: str):
    st.info(f"This view needs {what}. See README, *How to run*.")


def altair_theme():
    font = "IBM Plex Sans, system-ui, -apple-system, Helvetica, Arial, sans-serif"
    return {"config": {
        "font": font, "background": "transparent", "view": {"stroke": None},
        "axis": {"labelColor": INK_SOFT, "titleColor": INK_SOFT, "gridColor": RULE, "domainColor": RULE,
                 "tickColor": RULE, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500},
        "legend": {"labelColor": INK, "titleColor": INK_SOFT, "labelFontSize": 12, "titleFontWeight": 500},
        "header": {"labelColor": INK, "labelFontSize": 13, "labelFontWeight": 500, "titleColor": INK_SOFT},
        "title": {"color": INK, "fontWeight": 600, "fontSize": 14, "anchor": "start"}}}


def use_altair_theme():
    alt.theme.register("plex", enable=True)(altair_theme)
