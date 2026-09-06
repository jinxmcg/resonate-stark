"""Write STaRK-format predictions for one split with the FROZEN pipeline (P1 committed read):
eval_results_{split}.csv with columns idx, query_id, pred_rank (top-100 node ids, best first).
Reads only the query text and ids of the split; never the answers. Answers are scored once, in the
same run, when --score is given (the committed read), with metrics.py: Hit@1, Hit@5, Recall@20 and
MRR over the top-100 list, an answer outside the top-100 counting as reciprocal rank 0. The official
stark_qa Evaluator, as run by the leaderboard (top-100 ids, scores -i, all other candidates tied
below), can only score higher: an answer outside the top-100 then gets reciprocal rank <= 1/101 by
tie order. eval_check.py measures the difference on val and rescores the committed files officially.
Pipeline: retrieve.py ranking (beta, anchors) + text ranking (embedder tag) fused by RRF with the
weight chosen on train (data/fusion_<tag>.json); with --rerank, the train-fit listwise reranker
(rerank.py, P2 lever 1) reorders the fused candidates of every query with relational output."""
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
p.add_argument("--split", required=True, choices=["test", "test-0.1", "human_generated_eval", "val", "train"])
p.add_argument("--rel", required=True, help="relational ranking json for the split (from retrieve.py --split ...)")
p.add_argument("--text", required=True, help="text ranking json for the split (from embed_text2.py)")
p.add_argument("--w", type=float, required=True, help="fusion weight chosen on train")
p.add_argument("--score", action="store_true", help="the one committed read: print STaRK metrics")
p.add_argument("--rerank", default=None, help="data/rerank_<reltag>_<texttag>.json from rerank.py (P2)")
p.add_argument("--scores-out", default=None, help="write per-query reranker scores of the top-5 (lever B selector features)")
p.add_argument("--t2l", default=None, help="text-to-latent ranking json for the split (third candidate source, needs a reranker fit with --t2l-tag)")
a = p.parse_args()
qa = load_qa("prime", human_generated_eval=(a.split == "human_generated_eval"))
idx = qa.get_idx_split()[a.split].tolist() if a.split != "human_generated_eval" else list(range(len(qa)))
rel = json.load(open(a.rel)); txt = json.load(open(a.text))
RR = None
if a.rerank:
    from rerank import build
    RR = json.load(open(a.rerank)); mu, sd = np.array(RR["mu"], np.float32), np.array(RR["sd"], np.float32)
    d = np.load("data/kg.npz"); N = len(d["node_type"])
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    T2L = json.load(open(a.t2l)) if a.t2l else None
    feats = build(rel, txt, a.w, logdeg, [int(qa[i][1]) for i in idx], T2L)
    n_rr = 0
rows, out, SC = [], [], {}
for i in idx:
    q, qid, ans, _ = qa[i]
    r = rel.get(str(qid), {}); r = r.get("top", []) if isinstance(r, dict) else r
    ranked = fused(r, txt.get(str(qid), []), a.w)[:100]
    if RR is not None and feats.get(int(qid)) is not None:
        cands, f = feats[int(qid)]; at = rel[str(qid)]["answer_type"]
        wv = np.array(RR["w_type"].get(str(at), RR["w_global"]), np.float32)
        sc = ((f - mu) / sd) @ wv
        order = np.argsort(-sc); ranked = [cands[j] for j in order][:100]; n_rr += 1
        SC[int(qid)] = {"top": [float(sc[j]) for j in order[:5]], "n_cands": int(len(cands)), "n_supported": int(sum(1 for e in rel[str(qid)]["feats"]["exact"] if e > 0)), "max_exact": float(max(rel[str(qid)]["feats"]["exact"] or [0])), "n_mentions": len(rel[str(qid)].get("mentions", []))}
    out.append((i, int(qid), ranked))
    if a.score: rows.append(stark_metrics(ranked, ans))
with open(f"eval_results_{a.split}.csv", "w") as f:
    f.write("idx,query_id,pred_rank\n")
    for i, qid, ranked in out: f.write(f'{i},{qid},"{ranked}"\n')
if a.scores_out: json.dump(SC, open(a.scores_out, "w"))
print(f"wrote eval_results_{a.split}.csv with {len(out)} rows" + (f", reranked {n_rr}" if RR is not None else ""))
if a.score: print(f"COMMITTED {a.split}:", {k: round(v, 4) for k, v in summarize(rows).items()})
