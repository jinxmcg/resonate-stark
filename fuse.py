"""Fusion of the relational and text rankings by weighted reciprocal-rank fusion:
score(c) = w * 1/(60 + rank_rel(c)) + (1-w) * 1/(60 + rank_text(c)); w chosen on STaRK `train`
(grid), reported on `val`. Reads data/rel_{split}.json and data/text_{split}.json (top-100 each)."""
import json, sys, argparse
import numpy as np
import stark_shim  # noqa
from stark_qa import load_qa
from metrics import stark_metrics, summarize

def fused(rel, txt, w, k0=60):
    s = {}
    for i, c in enumerate(rel): s[c] = s.get(c, 0) + w / (k0 + i + 1)
    for i, c in enumerate(txt): s[c] = s.get(c, 0) + (1 - w) / (k0 + i + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: -x[1])]

def run(split, w, rel, txt, qa, idx):
    rows = []
    for i in idx:
        q, qid, ans, _ = qa[i]
        r = rel.get(str(qid), {}).get("top", []) if isinstance(rel.get(str(qid)), dict) else rel.get(str(qid), [])
        t = txt.get(str(qid), [])
        rows.append(stark_metrics(fused(r, t, w), ans))
    return summarize(rows)

ap = argparse.ArgumentParser(); ap.add_argument("--text-tag", default=""); ap.add_argument("--rel-tag", default=""); A = ap.parse_args()
suf = ("_" + A.text_tag) if A.text_tag else ""; rsuf = ("_" + A.rel_tag) if A.rel_tag else ""
qa = load_qa("prime"); sp = qa.get_idx_split()
rel = {s: json.load(open(f"data/rel_{s}{rsuf}.json")) for s in ("train", "val")}
txt = {s: json.load(open(f"data/text_{s}{suf}.json")) for s in ("train", "val")}
print("text:", suf or "minilm", "rel:", rsuf or "default")
tr, va = sp["train"].tolist(), sp["val"].tolist()
best = None
for w in np.linspace(0, 1, 21):
    m = run("train", w, rel["train"], txt["train"], qa, tr)
    print(f"train w={w:.2f} " + " ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
    if best is None or m["mrr"] > best[1]["mrr"]: best = (w, m)
w = best[0]; print(f"chosen on train: w={w:.2f} ({best[1]})")
for name, ww in (("relational only", 1.0), ("text only", 0.0), (f"fused w={w:.2f}", w)):
    m = run("val", ww, rel["val"], txt["val"], qa, va)
    print(f"VAL {name:18s}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
json.dump({"w": float(w), "train": best[1]}, open(f"data/fusion{suf}.json", "w"))
