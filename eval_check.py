"""Metric audit for the committed prediction files.
(1) Equivalence of metrics.py with the official stark_qa Evaluator on `val` predictions produced by
    the submitted pipeline (val is a development split; no model change is involved).
(2) Official-evaluator numbers and bootstrap 95% confidence intervals (resampling questions) for the
    three committed prediction files in results_p2/. This re-scores the frozen files; nothing is tuned.
Official leaderboard scoring (review_submission.py in the STaRK space) builds pred_dict from the
top-100 ids with scores -i; candidates outside the top-100 tie at min-1, so an answer outside the
top-100 gets a reciprocal rank <= 1/101 (tie order decides), where metrics.py counts it as 0.
Usage: PYTHONPATH=lib uv run python eval_check.py --val eval_results_val.csv
"""
import argparse, csv, ast, json, time, numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from stark_qa.evaluator import Evaluator
from metrics import stark_metrics

def official(ev, ranked, ans):
    pred = {int(c): -i for i, c in enumerate(ranked[:100])}
    r = ev(pred, torch.LongTensor(ans), metrics=["hit@1", "hit@5", "recall@20", "mrr"])
    return {"hit1": float(r["hit@1"]), "hit5": float(r["hit@5"]), "recall20": float(r["recall@20"]), "mrr": float(r["mrr"])}

def load_csv(path):
    return {int(r["query_id"]): ast.literal_eval(r["pred_rank"]) for r in csv.DictReader(open(path))}

def boot(rows, n=2000, seed=0):
    rng = np.random.default_rng(seed); A = np.array([[r[k] for k in ("hit1", "hit5", "recall20", "mrr")] for r in rows]); out = {}
    idx = rng.integers(0, len(A), size=(n, len(A))); means = A[idx].mean(1)
    for j, k in enumerate(("hit1", "hit5", "recall20", "mrr")):
        lo, hi = np.percentile(means[:, j], [2.5, 97.5]); out[k] = (A[:, j].mean(), lo, hi)
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--val", default=None); p.add_argument("--splits", default="test,test-0.1,human_generated_eval"); a = p.parse_args()
    N = len(np.load("data/kg.npz")["node_type"]); ev = Evaluator(list(range(N)))
    if a.val:
        qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist(); P = load_csv(a.val); loc, off = [], []
        for i in idx:
            q, qid, ans, _ = qa[i]; r = P[int(qid)]; loc.append(stark_metrics(r, ans)); off.append(official(ev, r, ans))
        for k in ("hit1", "hit5", "recall20", "mrr"):
            print(f"val {k:9s} metrics.py {np.mean([x[k] for x in loc]):.4f}   official {np.mean([x[k] for x in off]):.4f}   max |diff| per query {max(abs(x[k]-y[k]) for x, y in zip(loc, off)):.4f}", flush=True)
    for s in a.splits.split(","):
        human = s == "human_generated_eval"; qa = load_qa("prime", human_generated_eval=human)
        idx = qa.get_idx_split()[s].tolist() if not human else list(range(len(qa))); P = load_csv(f"results_p2/eval_results_{s}.csv"); off = []
        for i in idx:
            q, qid, ans, _ = qa[i]; off.append(official(ev, P[int(qid)], ans))
        ci = boot(off)
        print(f"{s} (n={len(idx)}) official evaluator, mean [95% bootstrap CI]: " + "  ".join(f"{k} {m*100:.2f} [{lo*100:.1f}, {hi*100:.1f}]" for k, (m, lo, hi) in ci.items()), flush=True)
    print("EVAL_CHECK_DONE")

if __name__ == "__main__":
    main()
