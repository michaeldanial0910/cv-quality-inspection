import streamlit as st

from app_lib.ui import esc, figures, heat_overlay, inspector, load_csv, load_json, need, tile

st.title("What lands on the inspector's desk")
st.html('<p class="lede">On the 195 test parts, the automatic zones made no mistakes. The parts the model was unsure '
        'about went to a person. These are those parts, with what the model looked at and why each one was hard.</p>')

err = load_csv("errors_test.csv")
ex = load_json("explain_summary.json")
if err is None or inspector() is None:
    need("reports/errors_test.csv and the model (python -m src.explain)")
    st.stop()

n_def = int((err.label == 1).sum())
dt = ex.get("deletion_test", {})
st.html(figures([(len(err), "test parts sent to a person"), (n_def, "of them truly defective"),
                 (len(err) - n_def, "good parts a person would clear")]))

only = st.toggle("Only the defective ones", key="only_def")
show = err[err.label == 1] if only else err
cols = st.columns(4)
for i, r in enumerate(show.itertuples()):
    try:
        img, p, zone = heat_overlay(r.path)
    except FileNotFoundError:
        continue
    truth = "defective" if r.label == 1 else "acceptable"
    cap = (f'<b>{truth}</b> · P(defect) {p:.3f}<br>{esc(r.tag) if isinstance(r.tag, str) else ""}'
           f'<br>{esc(r.path.split("/")[-1])}')
    cols[i % 4].html(tile(img, cap, defect=r.label == 1))

st.subheader("Why these were hard")
st.html('<ul>'
        f'<li><b>The {n_def} defective parts</b> all have a <b>tiny pinhole or dent on the outer rim</b> and were photographed '
        'against the <b>brighter background</b> setup. Small defect plus different lighting means weak evidence. The model '
        'still scored them far above every passed part, so the policy did its job by sending them to a person.</li>'
        f'<li><b>The {len(err) - n_def} good parts</b> show <b>machining marks or surface texture on the rim</b>, with no '
        'visible defect. Their heatmaps are faint and scattered: the model is unsure, not wrong about a specific spot.</li>'
        '</ul>')

if dt:
    st.subheader("Do the heatmaps mean anything? The deletion test")
    st.html(f'<div class="callout">For the {dt["n"]} defective test parts the model flagged, blurring the <b>hottest 10%</b> '
            f'of the heatmap lowered the defect score (logit) by <b>{dt["mean_logit_drop_heatmap"]:.1f}</b> on average. '
            f'Blurring a <b>random</b> patch of the same size lowered it by <b>{dt["mean_logit_drop_random"]:.1f}</b>. '
            f'Difference 95% CI {dt["difference_ci"][0]:.1f}–{dt["difference_ci"][1]:.1f}; the hot region mattered '
            f'more in {dt["share_heatmap_bigger"]:.0%} of parts. The heatmaps point at evidence the model actually uses.</div>')
