import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import INK, figures, load_csv, load_json, need, use_altair_theme

use_altair_theme()
st.title("Where this model would break on a real line")
st.html('<p class="lede">Every test photo was re-scored after simulating a changed camera or lighting, with the '
        'thresholds frozen as they were set on validation. This is what happens if conditions drift and nobody '
        'recalibrates.</p>')

rb = load_csv("robustness.csv")
ood = load_csv("ood_check.csv")
perf = load_json("performance.json")
if rb is None:
    need("reports/robustness.csv (python -m src.robustness)")
    st.stop()

clean = rb[rb.perturbation == "none"].iloc[0]
d = rb[rb.perturbation != "none"].copy()
d["Change"] = d.apply(lambda r: f"{r.perturbation} {r.severity:g}", axis=1)
worst = d.sort_values("false_reject_rate").iloc[-1]
st.html(figures([(f"{int(d.missed_defects.max())}", "most defects missed under any of the 16 changes"),
                 (f"{clean.false_reject_rate:.0%}", "good parts flagged on clean photos"),
                 (f"{worst.false_reject_rate:.0%}", f"good parts flagged under {worst.Change}")]))

st.subheader("It fails safe: defects are still caught, good parts get flagged")
order = d["Change"].tolist()
base = alt.Chart(d).encode(y=alt.Y("Change:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12)),
                           tooltip=["Change", alt.Tooltip("false_reject_rate:Q", format=".0%", title="good parts flagged"),
                                    alt.Tooltip("missed_defects:Q", title="defects missed"),
                                    alt.Tooltip("pr_auc:Q", format=".3f", title="PR-AUC")])
bars = base.mark_bar(height=14, color="#B07D00", cornerRadiusEnd=2).encode(
    x=alt.X("false_reject_rate:Q", title="Good parts flagged (review or reject)", scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=5)))
rule = alt.Chart(pd.DataFrame({"x": [clean.false_reject_rate]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="x:Q")
st.altair_chart((bars + rule).properties(height=22 * len(d)), theme=None, width="stretch")
st.html('<p class="muted">Dashed line: clean photos. Defect recall stayed at 99–100% under every change, and ranking '
        '(PR-AUC) barely moved, so the model still <i>orders</i> parts well. What breaks is the <b>threshold</b>: '
        'darker, blurrier or more compressed photos all look "more defective". A lighting change would flood the review '
        'queue rather than ship bad parts. In production: watch the review rate, and re-set thresholds after any camera '
        'or lighting change.</p>')

if ood is not None:
    st.subheader("Photos that are not a casting at all")
    o = ood.rename(columns={"input": "Input", "ood_distance": "Distance", "flagged": "Routed to a person",
                            "defect_prob_without_guard": "P(defect) if no guard"})
    st.dataframe(o.style.format({"Distance": "{:.2f}", "P(defect) if no guard": "{:.1%}"}), hide_index=True,
                 width="stretch")
    st.html('<p class="muted">The model only knows two answers. Without the guard, random noise would be called 100% '
            'defective. The guard measures how far an image\'s features are from the 5th-closest training photo; the '
            'limit is the 99th percentile on validation photos (1% of real parts would be sent to review by it).</p>')

st.subheader("Known limitations")
st.html(f'''<ul>
<li><b>One product, one view, one camera.</b> All photos are top views of the same impeller, under at least two lighting
setups (grey and bright backgrounds). A different part, angle or camera needs new data and new thresholds.</li>
<li><b>Small test set.</b> 195 photos (117 defective). "0 missed defects" has a 95% upper bound of about 2.5% on the true
miss rate (rule of three: 3/117). Not enough to promise 99.9%.</li>
<li><b>Binary labels.</b> No defect type, size or location labels, so no per-defect-type error rates.</li>
<li><b>Unknown labelling process.</b> The dataset does not say who labelled the parts or how disagreements were settled.</li>
<li><b>Threshold drift</b> (above): the decision thresholds depend on lighting.</li>
<li><b>Licence:</b> CC BY-NC-ND 4.0: research and education only, images not redistributed here.</li>
<li><b>Speed is not a constraint:</b> {perf.get("cpu_ms_per_image_median", 0):.0f} ms per part on {perf.get("cpu_threads", 4)}
CPU threads, {perf.get("onnx_mb", 16):.0f} MB model, no GPU needed.</li>
</ul>''')
