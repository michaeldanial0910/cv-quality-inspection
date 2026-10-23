# Walkthrough video: notes (target 4–5 min)

Record the Streamlit app (`streamlit run app.py`, browser at full screen) instead of the code. One sentence per idea;
the numbers are on screen, so point at them rather than reading them all.

1. **The problem (20 s).** A foundry photographs every cast impeller from the top. Missing a defect ships a bad pump;
   rejecting a good one wastes it. So the goal is: catch essentially every defect, scrap as few good parts as possible,
   and let a person decide when the model is unsure.

2. **Inspect a part (70 s).** *Clear defect:* auto-reject, 99%+, point at the heatmap sitting on the rim defect.
   *Clear pass:* auto-pass. *Borderline:* a real defective part with a tiny rim pinhole, scored 15%, sent to a
   person: the policy working as intended. Open *Simulate camera drift*, drag lighting to 0.6: the same part jumps
   to auto-reject. Say: "the thresholds were set under the original lighting; I come back to this on Limits."
   Optional: *Upload a photo* of anything that is not a casting: it is routed to a person (OOD guard).

3. **The data (50 s).** *The data* page. Kaggle's ready-made split: 707 of 715 test images have a near-copy in train,
   64 are identical files, so accuracies reported on it are inflated. I used the 1,300 originals instead, found repeat
   photos of the same part with a rotation-invariant comparison (polar unwrap: a rotation becomes a shift), and kept
   each part in one split. Every choice was made on validation; test was scored once.

4. **Results (60 s).** The bar chart: to catch ≥99% of defects, HOG rejects 87% of good parts, the CNNs 12–15%.
   Transfer learning wins; frozen vs fine-tuned can't be separated on this test set (both rank it perfectly), and I
   say so. The three-zone outcome: 0 wrong automatic calls, 14 parts (7%) to a person. Drag the threshold slider:
   one threshold always trades shipped defects for scrapped parts; the review zone avoids that.

5. **Review queue (40 s).** The 14 parts. The 3 defective ones: tiny rim pinholes under bright lighting. The 11 good
   ones: machining marks. Then the deletion test: blurring where the heatmap points lowers the score twice as much as
   blurring a random patch, so the heatmaps reflect what the model uses.

6. **Limits and next steps (40 s).** *Limits* page. It fails safe: under lighting, blur or JPEG changes it still catches
   the defects but flags many more good parts, so a camera change would flood review. Monitor the review rate and
   re-set thresholds after changes. Other limits: one product and view, 195 test photos, binary labels. Next: defect
   type labels, a bigger and harder test set, and training with stronger lighting/blur augmentation.

Close (10 s): 36 ms per part on a laptop CPU, 16 MB model, FastAPI at `/docs`, everything reproducible from the README.
