"""Write STaRK-format predictions for one split with the FROZEN pipeline (P1 committed read):
eval_results_{split}.csv with columns idx, query_id, pred_rank (top-100 node ids, best first).
Reads only the query text and ids of the split; never the answers. Answers are scored once, by
STaRK's own metrics, in the same run and printed (the committed read) when --score is given.
Pipeline: retrieve.py ranking (beta, anchors) + text ranking (embedder tag) fused by RRF with the
weight chosen on train (data/fusion_<tag>.json)."""
import argparse, json, subprocess, sys, numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
def fused(rel, txt, w, k0=60):                      # same RRF as fuse.py (kept local: fuse.py is a script)
    s = {}
    for i, c in enumerate(rel): s[c] = s.get(c, 0) + w / (k0 + i + 1)
    for i, c in enumerate(txt): s[c] = s.get(c, 0) + (1 - w) / (k0 + i + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: -x[1])]
from metrics import stark_metrics, summarize
p = argparse.ArgumentParser()
p.add_argument("--split", required=True, choices=["test", "test-0.1", "human_generated_eval", "val"])
p.add_argument("--rel", required=True, help="relational ranking json for the split (from retrieve.py --split ...)")
p.add_argument("--text", required=True, help="text ranking json for the split (from embed_text2.py)")
p.add_argument("--w", type=float, required=True, help="fusion weight chosen on train")
p.add_argument("--score", action="store_true", help="the one committed read: print STaRK metrics")
a = p.parse_args()
qa = load_qa("prime", human_generated_eval=(a.split == "human_generated_eval"))
idx = qa.get_idx_split()[a.split].tolist() if a.split != "human_generated_eval" else list(range(len(qa)))
rel = json.load(open(a.rel)); txt = json.load(open(a.text))
rows, out = [], []
for i in idx:
    q, qid, ans, _ = qa[i]
    r = rel.get(str(qid), {}); r = r.get("top", []) if isinstance(r, dict) else r
    ranked = fused(r, txt.get(str(qid), []), a.w)[:100]
    out.append((i, int(qid), ranked))
    if a.score: rows.append(stark_metrics(ranked, ans))
with open(f"eval_results_{a.split}.csv", "w") as f:
    f.write("idx,query_id,pred_rank\n")
    for i, qid, ranked in out: f.write(f'{i},{qid},"{ranked}"\n')
print(f"wrote eval_results_{a.split}.csv with {len(out)} rows")
if a.score: print(f"COMMITTED {a.split}:", {k: round(v, 4) for k, v in summarize(rows).items()})
