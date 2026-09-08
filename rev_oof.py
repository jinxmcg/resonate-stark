"""P6 fail-fast: do the reverse-operator features help the reranker? Train split only.

Groups = the RELATIONAL shortlist (retrieve.py --dump-feats --rev top-100) of each question.
Features, arm "base": z, exact, exact/nm, rel_rrf, logdeg — the relational subset of the P3
reranker's feature set. Arm "rev": base + rev_raw_max, rev_raw_mean, rev_nov_max, rev_nov_mean.
Model: the P3 reranker's listwise logistic regression (rerank.fit; steps 400, lr 0.05, l2 1e-3,
features standardised on the fitting folds), one global weight vector (1,000 questions is too few
for per-type fits). Out of fold: fold = position of the question in the split slice % 5; each fold
is scored by the model fit on the other four, on the plain AND the terse groups together (the P3
reranker is likewise fit on plain + paraphrased train). Nothing is fit on a question it scores.
Writes the reranked rankings in retrieve.py's json format; predict.py --score does the scoring.

Reads: train only. Usage:
PYTHONPATH=lib uv run python rev_oof.py --plain data/rel_train_p6_plain.json --terse data/rel_train_p6_terse.json --limit 1000 --out-prefix data/p6oof
"""
import argparse, json, numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from rerank import fit

BASE = ["z", "exact", "exact/nm", "rel_rrf", "logdeg"]
REVF = ["rev_raw_max", "rev_raw_mean", "rev_nov_max", "rev_nov_mean"]


def build_groups(rel, qids, logdeg, use_rev):
    """{qid: (candidates, features (n, F))}; None when the question has no relational output."""
    G = {}
    for qid in qids:
        r = rel.get(str(qid))
        if not r or not r.get("top") or "feats" not in r:
            G[qid] = None; continue
        top = r["top"]; f = r["feats"]; n = len(top); nm = max(1, len(r.get("mentions", [])))
        ex = np.array(f["exact"], np.float32)
        cols = [np.array(f["z"], np.float32), ex, ex / nm,
                np.array([1.0 / (60 + i + 1) for i in range(n)], np.float32),
                logdeg[np.array(top, np.int64)]]
        if use_rev:
            for k in REVF:
                v = f.get(k)
                cols.append(np.array(v if v else [0.0] * n, np.float32))
        G[qid] = (top, np.stack(cols, 1))
    return G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plain", required=True); p.add_argument("--terse", required=True)
    p.add_argument("--split", default="train", choices=["train", "val"]); p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--out-prefix", default="data/p6oof"); p.add_argument("--device", default="cuda")
    a = p.parse_args(); dev = torch.device(a.device)
    qa = load_qa("prime"); idx = qa.get_idx_split()[a.split].tolist()
    if a.limit: idx = idx[:a.limit]
    qids = [int(qa[i][1]) for i in idx]; ANS = {int(qa[i][1]): set(qa[i][2]) for i in idx}
    fold_of = {q: j % a.folds for j, q in enumerate(qids)}
    d = np.load("data/kg.npz"); N = len(d["node_type"])
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    REL = {"plain": json.load(open(a.plain)), "terse": json.load(open(a.terse))}
    for arm, use_rev in (("base", False), ("rev", True)):
        names = BASE + (REVF if use_rev else [])
        G = {w: build_groups(REL[w], qids, logdeg, use_rev) for w in ("plain", "terse")}
        ranked = {w: {} for w in ("plain", "terse")}
        W = []
        for k in range(a.folds):
            tr = []
            for w in ("plain", "terse"):
                for q in qids:
                    if fold_of[q] == k or G[w][q] is None: continue
                    cands, f = G[w][q]
                    y = np.array([1.0 if c in ANS[q] else 0.0 for c in cands], np.float32)
                    if y.sum() == 0: continue
                    tr.append((f, y))
            X = np.concatenate([x for x, _ in tr]); mu, sd = X.mean(0), X.std(0) + 1e-6
            wv = fit([((x - mu) / sd, y) for x, y in tr], dev, l2=a.l2)
            W.append(wv)
            print(f"  [{arm}] fold {k}: {len(tr)} fitting groups, weights " + ", ".join(f"{n_} {v:+.3f}" for n_, v in zip(names, wv)), flush=True)
            for w in ("plain", "terse"):
                for q in qids:
                    if fold_of[q] != k: continue
                    if G[w][q] is None:
                        r = REL[w].get(str(q)) or {}
                        ranked[w][str(q)] = {"top": r.get("top", []), "answer_type": r.get("answer_type")}
                        continue
                    cands, f = G[w][q]
                    s = ((f - mu) / sd) @ wv
                    ranked[w][str(q)] = {"top": [int(cands[j]) for j in np.argsort(-s)][:100], "answer_type": REL[w][str(q)]["answer_type"]}
        Wm = np.stack(W)
        print(f"[{arm}] out-of-fold weights, mean over {a.folds} folds: " + ", ".join(f"{n_} {m:+.3f} (sd {s:.3f})" for n_, m, s in zip(names, Wm.mean(0), Wm.std(0))), flush=True)
        for w in ("plain", "terse"):
            out = f"{a.out_prefix}_{arm}_{w}.json"
            json.dump(ranked[w], open(out, "w")); print(f"[{arm}] {w}: wrote {out} ({len(ranked[w])} questions)", flush=True)
    print("REV_OOF_DONE")


if __name__ == "__main__":
    main()
