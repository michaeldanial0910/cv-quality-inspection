"""Generate notebooks/02_train_colab.ipynb: models B (linear probe) and C (fine-tuned), run on a Colab T4.

The notebook is self-contained: the split manifest and src/preprocess.py are embedded, the
manifest's SHA-256 and every image's SHA-256 are checked before training, so the Colab run
trains and tests on exactly the files the local pipeline uses.

    python notebooks/make_colab_notebook.py
"""
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = (ROOT / "data" / "split_manifest.csv").read_bytes()
MANIFEST_SHA = hashlib.sha256(manifest).hexdigest()
MANIFEST_B64 = base64.b64encode(gzip.compress(manifest, mtime=0)).decode()
PREPROCESS = (ROOT / "src" / "preprocess.py").read_text()

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})


md("""
# CV quality inspection: transfer-learning models (Colab, T4 GPU)

**Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. About 15–20 minutes.
At the end a file `cv_outputs.zip` downloads. Unzip it in the repo root: it fills `models/` and
`reports/predictions/`.

* **B · linear probe:** frozen ImageNet EfficientNet-B0, logistic regression on its 1,280 pooled features.
* **C · fine-tuned:** EfficientNet-B0, new 1-logit head; head first, then the last blocks unfrozen; train-only augmentation.

Model choice and early stopping use **validation only**. Test predictions are written once, at the end.
""")

code("""
!pip -q install kagglehub onnx onnxruntime
import base64, gzip, hashlib, io, json, os, random, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image
from torchvision import transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(DEV, torch.__version__, torchvision.__version__)
OUT = Path("cv_outputs"); (OUT / "models").mkdir(parents=True, exist_ok=True); (OUT / "reports/predictions").mkdir(parents=True, exist_ok=True)
""")

md("## 1 · The split manifest (embedded) and its checksum")
code(f"""
MANIFEST_SHA = "{MANIFEST_SHA}"
raw = gzip.decompress(base64.b64decode("{MANIFEST_B64}"))
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "manifest changed"
m = pd.read_csv(io.BytesIO(raw))
print(m.groupby(["split", "label"]).size().unstack())
""")

md("## 2 · Dataset from Kaggle, every file checked against the manifest")
code("""
import kagglehub
# Public dataset: normally downloads without a login. If it asks for credentials, run
# kagglehub.login() in a new cell (username + API key from kaggle.com > Settings > API) and re-run this cell.
root = Path(kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product"))
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m["file"] = [root / p for p in m["path"]]
missing = [str(p) for p in m["file"] if not p.exists()]
assert not missing, missing[:5]
bad = (m["file"].map(sha256) != m["file_sha256"]).sum()
assert bad == 0, f"{bad} files differ from the local copy"
print("all", len(m), "files match")
""")

md("## 3 · Preprocessing (identical to `src/preprocess.py`) and train-only augmentation")
code(PREPROCESS + """

class Rot90:
    # the part is round and shot from the top: any 90-degree turn is a physically real view
    def __call__(self, img): return img.rotate(90 * random.randint(0, 3))

AUG = T.Compose([Rot90(), T.RandomHorizontalFlip(), T.RandomRotation(15, fill=128),
                 T.ColorJitter(brightness=0.2, contrast=0.2), T.RandomApply([T.GaussianBlur(3)], p=0.2)])

class DS(torch.utils.data.Dataset):
    def __init__(self, df, train): self.df, self.train = df.reset_index(drop=True), train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img = to_gray_image(self.df.file[i]).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        if self.train: img = AUG(img)          # augmentation only ever touches training images
        return torch.from_numpy(to_input(img)), torch.tensor(float(self.df.label[i]))

split = {s: m[m.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
def loader(s, train=False, bs=32):
    g = torch.Generator().manual_seed(SEED)
    return torch.utils.data.DataLoader(DS(split[s], train), batch_size=bs, shuffle=train, num_workers=2, generator=g)
""")

md("## 4 · B · linear probe on frozen ImageNet features")
code("""
W = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
def backbone():
    net = torchvision.models.efficientnet_b0(weights=W)
    return net

@torch.no_grad()
def pooled(net, s):
    net.eval(); feats = []
    for x, _ in loader(s):
        f = net.features(x.to(DEV)); feats.append(net.avgpool(f).flatten(1).cpu().numpy())
    return np.concatenate(feats)

net = backbone().to(DEV)
E = {s: pooled(net, s) for s in ["train", "val", "test"]}
y = {s: split[s].label.values for s in split}
probe_log = []
for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    clf = LogisticRegression(C=C, max_iter=5000).fit(E["train"], y["train"])
    probe_log.append({"C": C, "val_ap": average_precision_score(y["val"], clf.predict_proba(E["val"])[:, 1])})
probe_log = pd.DataFrame(probe_log); print(probe_log)
best_C = float(probe_log.loc[probe_log.val_ap.idxmax(), "C"])
clf = LogisticRegression(C=best_C, max_iter=5000).fit(E["train"], y["train"])
pd.concat([split[s][["path", "label", "split"]].assign(score=clf.predict_proba(E[s])[:, 1]) for s in ["val", "test"]]
          ).to_csv(OUT / "reports/predictions/linear_probe.csv", index=False)
probe_log.to_csv(OUT / "reports/tuning_linear_probe.csv", index=False)
""")

md("## 5 · C · fine-tune EfficientNet-B0 (early stopping on validation average precision)")
code("""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        b = backbone()
        self.features, self.pool = b.features, nn.AdaptiveAvgPool2d(1)
        self.drop, self.head = nn.Dropout(0.3), nn.Linear(1280, 1)
    def forward(self, x):
        return self.head(self.drop(self.pool(self.features(x)).flatten(1))).squeeze(1)

@torch.no_grad()
def predict(model, s):
    model.eval(); out = []
    for x, _ in loader(s): out.append(model(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)

def run_phase(model, params, epochs, log, phase, patience=4):
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train(); t0, tot = time.time(), 0.0
        for x, yb in loader("train", train=True):
            opt.zero_grad(); loss = lossf(model(x.to(DEV)), yb.to(DEV)); loss.backward(); opt.step(); tot += loss.item() * len(x)
        sched.step()
        z = predict(model, "val"); yv = y["val"]
        vloss = float(lossf(torch.tensor(z), torch.tensor(yv, dtype=torch.float32)))
        ap = average_precision_score(yv, z)
        log.append({"phase": phase, "epoch": ep + 1, "train_loss": tot / len(split["train"]), "val_loss": vloss,
                    "val_ap": ap, "val_auc": roc_auc_score(yv, z), "sec": time.time() - t0})
        print(log[-1])
        if ap > best + 1e-4:
            best, best_state, wait = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return best

model, log = Net().to(DEV), []
for p in model.features.parameters(): p.requires_grad = False          # phase 1: head only
run_phase(model, [{"params": model.head.parameters(), "lr": 1e-3}], 4, log, "head")
for p in model.features[5:].parameters(): p.requires_grad = True       # phase 2: last blocks too
run_phase(model, [{"params": model.features[5:].parameters(), "lr": 1e-4},
                  {"params": model.head.parameters(), "lr": 5e-4}], 15, log, "fine-tune")
pd.DataFrame(log).to_csv(OUT / "reports/train_log.csv", index=False)
best_row = max([r for r in log if r["phase"] == "fine-tune"], key=lambda r: r["val_ap"])
print("best fine-tune epoch:", best_row)
""")

md("## 6 · Predictions (test written once, after every choice above is fixed)")
code("""
preds = []
for s in ["val", "test"]:
    z = predict(model, s)
    preds.append(split[s][["path", "label", "split"]].assign(logit=z, score=1 / (1 + np.exp(-z))))
pd.concat(preds).to_csv(OUT / "reports/predictions/finetuned.csv", index=False)
print(pd.concat(preds).groupby("split").apply(lambda d: average_precision_score(d.label, d.score)))
""")

md("""
## 7 · Export to ONNX for CPU serving

Two outputs: the defect **logit** and the last **feature map** (1280 × 7 × 7). Because the head is
global-average-pool + one linear layer, the class activation map is exactly
`sum_c w_c * feature_map_c`, computed locally with the saved head weights; no gradients needed.
""")
code("""
class Export(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        f = self.m.features(x)
        return self.m.head(self.m.pool(f).flatten(1)).squeeze(1), f

model.eval().cpu()
export = Export(model).eval()     # a new nn.Module starts in train mode; export() restores that mode afterwards
dummy = torch.from_numpy(np.stack([to_input(split["val"].file[i]) for i in range(4)]))
torch.onnx.export(export, dummy, OUT / "models/model.onnx", input_names=["image"], output_names=["logit", "features"],
                  dynamic_axes={"image": {0: "n"}, "logit": {0: "n"}, "features": {0: "n"}}, opset_version=17, dynamo=False)
np.savez(OUT / "models/head.npz", w=model.head.weight.detach().numpy()[0], b=model.head.bias.detach().numpy())

# Parity: ONNX on CPU vs the step-6 predictions of the trained model (GPU), on all validation images
import onnxruntime as ort
sess = ort.InferenceSession(str(OUT / "models/model.onnx"))
pv = pd.read_csv(OUT / "reports/predictions/finetuned.csv").query("split == 'val'").reset_index(drop=True)
assert (split["val"].path.values == pv.path.values).all()
x = np.stack([to_input(f) for f in split["val"].file])
z_onnx = np.concatenate([sess.run(["logit"], {"image": x[i:i + 32]})[0] for i in range(0, len(x), 32)])
diff = float(np.abs(z_onnx - pv.logit.values).max()); print("max |onnx - torch| logit on val:", diff); assert diff < 1e-2

meta = {"backbone": "efficientnet_b0 IMAGENET1K_V1", "torch": torch.__version__, "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "manifest_sha256": MANIFEST_SHA,
        "linear_probe_C": best_C, "best_finetune_epoch": best_row["epoch"], "best_val_ap": best_row["val_ap"],
        "onnx_vs_torch_val_max_diff": diff, "epochs_run": len(log), "train_minutes": round(sum(r["sec"] for r in log) / 60, 1),
        "onnx_mb": round(os.path.getsize(OUT / "models/model.onnx") / 1e6, 1)}
(OUT / "models/colab_meta.json").write_text(json.dumps(meta, indent=2)); print(meta)
""")

md("## 8 · Download")
code("""
import shutil
shutil.make_archive("cv_outputs", "zip", OUT)
from google.colab import files
files.download("cv_outputs.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
(ROOT / "notebooks" / "02_train_colab.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/02_train_colab.ipynb, manifest", MANIFEST_SHA[:12])
