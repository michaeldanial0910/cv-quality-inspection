import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (INK, MODEL_COLOR, MODEL_NAME, ZONE_COLOR, ZONE_TEXT, figures, load_csv, load_json, need,
                        policy, use_altair_theme)
from src.metrics import at_threshold

use_altair_theme()
st.title("Does transfer learning beat a hand-made baseline?")
st.html('<p class="lede">Three models, the same 195 test photos (117 defective) from parts never seen in training. '
        'Each model\'s threshold was set on validation to catch at least 99% of defects. The question is how many '
        'good parts each one throws away to get there. Intervals are 95% bootstrap over test images.</p>')

s = load_json("summary.json")
m = load_csv("metrics.csv")
if not s or m is None:
    need("reports/summary.json (python -m src.evaluate)")
    st.stop()
M = s["models"]
z = s["zones_test"]
n_test = sum(v["n"] for v in z.values())
st.html(figures([
    (f"{M['finetuned']['pr_auc']:.3f}", "PR-AUC of the fine-tuned model on test (HOG baseline: "
                                        f"{M['hog_lr']['pr_auc']:.3f})"),
    (f"{M['finetuned']['false_reject_rate']:.0%}", "of good parts rejected at ≥99% defect recall "
                                                   f"(HOG: {M['hog_lr']['false_reject_rate']:.0%})"),
    ("0", "wrong calls in the automatic zones on test"),
    (f"{z['human review']['n'] / n_test:.0%}", f"of parts sent to a person ({z['human review']['n']} of {n_test})"),
]))

st.subheader("Good parts thrown away to catch ≥99% of defects")
d = pd.DataFrame([{"model": k, "Model": MODEL_NAME[k], "frr": v["false_reject_rate"], "lo": v["frr_ci"][0],
                   "hi": v["frr_ci"][1], "pr_auc": v["pr_auc"]} for k, v in M.items()])
color = alt.Color("Model:N", scale=alt.Scale(domain=d["Model"].tolist(), range=[MODEL_COLOR[k] for k in d["model"]]),
                  legend=None)
base = alt.Chart(d).encode(y=alt.Y("Model:N", sort=d["Model"].tolist(), title=None,
                                   axis=alt.Axis(labelFontSize=13, labelLimit=280)), color=color,
                           tooltip=["Model", alt.Tooltip("frr:Q", format=".1%", title="false-reject rate"),
                                    alt.Tooltip("lo:Q", format=".1%", title="95% low"),
                                    alt.Tooltip("hi:Q", format=".1%", title="95% high")])
bars = base.mark_bar(height=24, cornerRadiusEnd=3).encode(
    x=alt.X("frr:Q", title="Share of good parts rejected (test)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)))
err = base.mark_rule(color=INK, strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q")
txt = base.mark_text(align="left", dx=6, color=INK, fontSize=12).encode(x="hi:Q", text=alt.Text("frr:Q", format=".0%"))
st.altair_chart((bars + err + txt).properties(height=150), theme=None, width="stretch")

t = m.set_index("model")
tab = pd.DataFrame({
    "Model": [MODEL_NAME[k] for k in t.index],
    "PR-AUC": [f"{M[k]['pr_auc']:.3f} ({M[k]['pr_auc_ci'][0]:.3f}–{min(M[k]['pr_auc_ci'][1], 1):.3f})" for k in t.index],
    "ROC-AUC": [f"{t.loc[k, 'roc_auc']:.3f}" for k in t.index],
    "Defects missed": [int(t.loc[k, "missed_defects"]) for k in t.index],
    "Good parts rejected": [f"{t.loc[k, 'false_reject_rate']:.1%}" for k in t.index],
    "Accuracy at 0.5": [f"{t.loc[k, 'acc_at_0.5']:.1%}" for k in t.index]})
st.dataframe(tab, hide_index=True, width="stretch")
pp = s["paired_pr_auc"]
fh, fl = pp["finetuned - hog_lr"], pp["finetuned - linear_probe"]
st.html(f'<p class="muted">Paired on the same images, fine-tuned minus HOG PR-AUC is <b>+{fh[0]:.3f}</b> '
        f'(95% CI +{fh[1]:.3f} to +{fh[2]:.3f}): transfer learning clearly helps. Fine-tuned minus frozen is '
        f'<b>{fl[0]:+.3f}</b>: both rank this test set perfectly, so it cannot tell them apart. The fine-tuned model '
        f'is served because it was best on <i>validation</i> (AP 0.998), and it rejects slightly fewer good parts.</p>')

st.subheader("The three-zone policy on test")
pol = policy()
zz = pd.DataFrame([{"Zone": ZONE_TEXT[k], "zone": k, "Truth": t_, "n": v[c]} for k, v in z.items()
                   for t_, c in [("defective", "defective"), ("acceptable", "ok")]])
ch = alt.Chart(zz).mark_bar().encode(
    y=alt.Y("Zone:N", sort=[ZONE_TEXT[k] for k in ZONE_COLOR], title=None, axis=alt.Axis(labelFontSize=13)),
    x=alt.X("n:Q", title="Test parts"),
    color=alt.Color("Truth:N", scale=alt.Scale(domain=["defective", "acceptable"], range=["#D4562E", "#9AA3B5"]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["Zone", "Truth", "n"])
st.altair_chart(ch.properties(height=170), theme=None, width="stretch")
st.html(f'<p class="muted">Auto-pass below {pol.get("review_from", 0):.3f}: the edge that kept 99.5% of validation defects '
        f'above it. Auto-reject from {pol.get("reject_from", 0):.3f}: the point where validation precision reached 99%. '
        f'Probabilities are temperature-scaled (T = {pol.get("temperature", 1):.2f}, fitted on validation).</p>')

st.subheader("Move the threshold yourself")
p = load_csv("predictions/finetuned_calibrated.csv")
pt = p[p["split"] == "test"]
thr = st.slider("Reject if P(defect) ≥", 0.0, 1.0, 0.5, 0.01, key="thr")
r = at_threshold(pt["label"], pt["prob"], thr)
st.html(figures([(r["missed_defects"], "defective parts that would ship"),
                 (r["fp"], "good parts that would be scrapped"),
                 (f"{r['defect_recall']:.1%}", "of defects caught")]))
st.html('<p class="muted">A single threshold forces a trade-off. Missing a defect ships a bad pump; a false reject '
        'scraps a good one. The middle "send to a person" zone avoids choosing: only uncertain parts cost labour.</p>')

st.subheader("Fine-tuning curve")
log = load_csv("train_log.csv")
if log is not None:
    log["step"] = np.arange(1, len(log) + 1)
    lf = log.melt(id_vars=["step", "phase"], value_vars=["train_loss", "val_loss"], var_name="curve", value_name="loss")
    lines = alt.Chart(lf).mark_line(point=True).encode(
        x=alt.X("step:Q", title="Epoch (1–4: head only · 5+: last blocks unfrozen)"),
        y=alt.Y("loss:Q", title="Loss"), color=alt.Color("curve:N", legend=alt.Legend(orient="top", title=None),
                                                       scale=alt.Scale(range=["#2A6FD6", "#D4562E"])))
    best = log[log.phase == "fine-tune"].val_ap.idxmax()
    rule = alt.Chart(pd.DataFrame({"step": [best + 1]})).mark_rule(strokeDash=[4, 3], color=INK).encode(x="step:Q")
    st.altair_chart((lines + rule).properties(height=260), theme=None, width="stretch")
    st.html('<p class="muted">Dashed line: the kept checkpoint (best validation PR-AUC, fine-tune epoch 11). Validation '
            'loss was lowest earlier (epoch 6); the rule was fixed before training and is reported, not changed after.</p>')
