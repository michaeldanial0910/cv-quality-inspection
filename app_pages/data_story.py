import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import figures, load_json, need, use_altair_theme
from src.config import FIGURES

use_altair_theme()
st.title("The data, and the leak in its official split")
st.html('<p class="lede">Kaggle\'s "casting product" dataset: top-view photos of submersible pump impellers, labelled '
        'acceptable or defective (casting defects such as blow holes, pinholes, burrs and shrinkage). It ships twice: 1,300 original 512×512 '
        'photos, and 7,348 300×300 copies with a ready-made train/test split.</p>')

dup, sp = load_json("duplicates_summary.json"), load_json("split_summary.json")
if not dup:
    need("reports/duplicates_summary.json (python -m src.dedupe)")
    st.stop()

st.subheader("The ready-made split grades models on parts they trained on")
st.html(figures([(f"{dup['kaggle300_test_with_near_copy_in_train']} / {dup['kaggle300_test_images']}",
                  "test images with a near-copy in Kaggle's train split"),
                 (dup["kaggle300_test_exact_file_copy_in_train"], "test files that are byte-identical to a train file"),
                 (f"{dup['originals_with_a_near_copy']}", "of the 1,300 originals have a re-shot or rotated twin")]))
h = dup["kaggle300_test_nn_train_similarity_hist"]
hd = pd.DataFrame({"bin": list(h), "n": list(h.values())})
ch = alt.Chart(hd).mark_bar(color="#D4562E").encode(
    x=alt.X("bin:N", sort=list(h), title="Similarity of each Kaggle test image to its closest Kaggle train image",
            axis=alt.Axis(labelAngle=0, labelFontSize=11)),
    y=alt.Y("n:Q", title="Test images"), tooltip=["bin", "n"])
st.altair_chart(ch.properties(height=220), theme=None, width="stretch")
st.html('<div class="callout bad">The 7,348-image version is the originals augmented many times (rotated, flipped, '
        'brightened), and copies of the same photo landed on both sides of its split. A model scored on it is '
        'partly tested on photos it memorised, so published 99%+ accuracies on this dataset are inflated. '
        'This project ignores that split and uses the 1,300 originals.</div>')

st.subheader("How duplicates were found")
st.html('<p>Each part is centred on its edges and unwrapped into polar coordinates (distance from centre × angle). '
        'A rotated photo of the same part then becomes the same strip shifted sideways, so the best correlation over all '
        'shifts (and the mirror image) measures "same part, any rotation". Pairs at ≥ 0.97 are treated as one part. '
        'The threshold was set by eye from pairs like these, never from a model score.</p>')
if (FIGURES / "duplicate_pairs.png").exists():
    st.image(str(FIGURES / "duplicate_pairs.png"), caption="Pairs across the similarity range: same part above 0.97, "
                                                           "different parts below", width=430)

if sp:
    st.subheader("The split used here")
    t = pd.DataFrame({"Split": ["train", "val", "test"],
                      "Photos": [sp["counts"][s] for s in ["train", "val", "test"]],
                      "Distinct parts (groups)": [sp["groups"][s] for s in ["train", "val", "test"]],
                      "Defective": [f"{sp['defect_rate'][s]:.0%}" for s in ["train", "val", "test"]],
                      "Used for": ["fitting", "every choice: epochs, thresholds, temperature, OOD limit",
                                   "scored once, at the end"]})
    st.dataframe(t, hide_index=True, width="stretch")
    st.html(f'<p class="muted">Every near-duplicate group sits in one split (checked by tests), and no group mixes the two '
            f'labels. The manifest (<code>data/split_manifest.csv</code>, SHA-256 {sp["manifest_sha256"][:12]}…) is embedded in '
            'the Colab notebook, which checks it and every image\'s hash before training.</p>')
