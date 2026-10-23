import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

from app_lib.ui import ZONE_COLOR, ZONE_TEXT, esc, inspector, need, scale, test_images, zone_legend
from src.config import EXTRACTED
from src.infer import overlay

st.title("Should this casting ship?")
st.html('<p class="lede">A camera photographs each pump impeller from the top. The model gives a defect probability '
        'and routes the part: <b>auto-pass</b>, <b>auto-reject</b>, or <b>send to a person</b> when it is not sure. '
        'The heatmap shows which part of the photo pushed the score towards "defective".</p>')

ins = inspector()
imgs = test_images()
if ins is None or imgs.empty:
    need("the trained model in models/ and the test images in data/raw/")
    st.stop()

# deterministic demo picks from the test split (never used for training or any choice)
ok, de = imgs[imgs.label == 0].sort_values("prob"), imgs[imgs.label == 1].sort_values("prob")
pol = ins.policy
review = imgs[(imgs.prob >= pol.review_from) & (imgs.prob < pol.reject_from) & (imgs.label == 1)]
PRESETS = {"Clear defect": de.iloc[len(de) // 2]["path"],
           "Clear pass": ok.iloc[len(ok) // 3]["path"],
           "Borderline": (review.iloc[0] if len(review) else de.iloc[0])["path"],
           "Random test part": None,
           "Upload a photo": None}

c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
with c1:
    choice = st.segmented_control("Part", list(PRESETS), default="Clear defect", key="source") or "Clear defect"
with c2:
    if choice == "Random test part" and st.button("Another part", key="another"):
        st.session_state["seed"] = st.session_state.get("seed", 0) + 1

truth = None
if choice == "Upload a photo":
    up = st.file_uploader("Top-view photo of a casting (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="upload")
    if up is None:
        st.html('<p class="muted">Anything that is not a top-view casting photo should be routed to a person: '
                'try a random photo to see the out-of-distribution guard.</p>')
        st.stop()
    img = Image.open(up).convert("L")
    name = up.name
else:
    if choice == "Random test part":
        row = imgs.sample(1, random_state=st.session_state.get("seed", 0)).iloc[0]
    else:
        row = imgs[imgs.path == PRESETS[choice]].iloc[0]
    img = Image.open(EXTRACTED / row["path"]).convert("L")
    truth, name = int(row["label"]), row["path"].split("/")[-1]

with st.expander("Simulate camera drift (lighting, focus)", expanded=st.session_state.get("drift_open", False)):
    d1, d2 = st.columns(2)
    light = d1.slider("Lighting", 0.5, 1.5, 1.0, 0.05, key="light", help="1.0 = as photographed")
    blur = d2.slider("Blur (pixels)", 0.0, 4.0, 0.0, 0.5, key="blur")
shown = ImageEnhance.Brightness(img).enhance(light)
if blur:
    shown = shown.filter(ImageFilter.GaussianBlur(blur))

r = ins.inspect(shown)
a, b, c = st.columns([1, 1, 1.05], gap="medium")
a.image(shown.resize((420, 420)), caption=f"Photo · {name}", width="stretch")
b.image(overlay(shown, r.heatmap).resize((420, 420)), caption="Heatmap: evidence for 'defective'", width="stretch")

truth_html = ""
if truth is not None:
    wrong = (r.zone == "auto-pass" and truth == 1) or (r.zone == "auto-reject" and truth == 0)
    tname = "defective" if truth else "acceptable"
    truth_html = (f'<div class="row"><span>Inspector\'s label (hidden from the model)</span>'
                  f'<span class="truth{" wrong" if wrong else ""}">{tname}{" · wrong call" if wrong else ""}</span></div>')
ood = ("yes: routed to a person" if r.out_of_distribution else "no")
c.html(f'''<div class="verdict">
  <span class="zone" style="background:{ZONE_COLOR[r.zone]}">{ZONE_TEXT[r.zone]}</span>
  <div class="p">{r.probability:.1%}</div><div class="pl">probability the part is defective (calibrated)</div>
  {scale(r.probability, pol.review_from, pol.reject_from)}
  {zone_legend()}
  <div class="row"><span>Zone edges (set on validation)</span><span>{pol.review_from:.3f} · {pol.reject_from:.3f}</span></div>
  <div class="row"><span>Unfamiliar image?</span><span>{esc(ood)} (distance {r.ood_distance:.2f}, limit {pol.ood_distance:.2f})</span></div>
  <div class="row"><span>Model time on this CPU</span><span>{r.ms:.0f} ms</span></div>
  {truth_html}
</div>''')

if light != 1.0 or blur:
    st.html('<div class="callout warn">You changed the photo. The thresholds were set on photos taken under the '
            'original lighting. On the <b>Limits</b> page: under drift the model still catches defects, but it starts '
            'sending many good parts to review or rejecting them.</div>')
st.html('<p class="muted">The heatmap is a class activation map: the last convolutional features weighted by the '
        'final layer\'s weights. For this architecture it shows exactly which 7×7 regions added to the defect score. '
        'Deleting the hottest 10% of a defective photo lowers the score twice as much as deleting a random patch '
        '(Review queue page).</p>')
