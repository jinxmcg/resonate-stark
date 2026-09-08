"""P10 step 3: are P6's reverse-operator features what fixes NAME COLLISIONS?

P5's closing line said the MS / CAD / CP / HR losses must come from "the readout over the
candidate's neighbourhood — which candidate has the relation the question asks for". rev_raw is
exactly that, and P6 only ever tested it averaged over the whole slice. Here it is tested where it
was aimed. A val question is a COLLISION question if some matched name string resolves to two or
more distinct entity ids in its own parse. The relational-shortlist reranker of rev_oof.py is fit on
train (plain + paraphrased, all groups) with and without the four rev columns, applied to val, and
the reranked rankings are written for predict.py --score --qids to score each subset officially.

Usage: PYTHONPATH=lib uv run python p10_collide.py --train data/rel_train_rev.json --train-para data/rel_train_para_rev.json \
         --val-plain data/rel_val_rev.json --val-terse data/rel_val_terse_rev.json --out-prefix data/p10c
"""
import argparse, json, collections, numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from rerank import fit
from rev_oof import build_groups, BASE, REVF


def collisions(rel):
    """question ids whose parse resolved one name string to two or more distinct entity ids"""
    out = []
    for qid, r in rel.items():
        by = collections.defaultdict(set)
        for m in (r.get("mentions") or []):
            by[m[1]].add(int(m[0]))
        if any(len(v) >= 2 for v in by.values()): out.append(int(qid))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--train-para", required=True)
    p.add_argument("--val-plain", required=True); p.add_argument("--val-terse", required=True)
    p.add_argument("--out-prefix", default="data/p10c"); p.add_argument("--device", default="cuda"); p.add_argument("--l2", type=float, default=1e-3)
    a = p.parse_args(); dev = torch.device(a.device)
    qa = load_qa("prime"); sp = qa.get_idx_split()
    tr = [int(qa[i][1]) for i in sp["train"].tolist()]; va = [int(qa[i][1]) for i in sp["val"].tolist()]
    ANS = {int(qa[i][1]): set(qa[i][2]) for i in sp["train"].tolist() + sp["val"].tolist()}
    d = np.load("data/kg.npz"); N = len(d["node_type"])
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    RT = json.load(open(a.train)); RTP = json.load(open(a.train_para))
    RV = {"plain": json.load(open(a.val_plain)), "terse": json.load(open(a.val_terse))}
    col = sorted(set(collisions(RV["plain"])) | set(collisions(RV["terse"])))
    rest = [q for q in va if q not in set(col)]
    json.dump(col, open(f"{a.out_prefix}_collision_qids.json", "w")); json.dump(rest, open(f"{a.out_prefix}_rest_qids.json", "w"))
    print(f"collision questions on val: {len(col)} of {len(va)} ({len(col)/len(va):.1%}); remainder {len(rest)}", flush=True)
    for arm, use_rev in (("base", False), ("rev", True)):
        names = BASE + (REVF if use_rev else [])
        G = []
        for R in (RT, RTP):
            B = build_groups(R, tr, logdeg, use_rev)
            for q in tr:
                if B[q] is None: continue
                cands, f = B[q]; y = np.array([1.0 if c in ANS[q] else 0.0 for c in cands], np.float32)
                if y.sum() == 0: continue
                G.append((f, y))
        X = np.concatenate([x for x, _ in G]); mu, sd = X.mean(0), X.std(0) + 1e-6
        w = fit([((x - mu) / sd, y) for x, y in G], dev, l2=a.l2)
        print(f"[{arm}] {len(G)} train groups; weights " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(names, w)), flush=True)
        for wording in ("plain", "terse"):
            B = build_groups(RV[wording], va, logdeg, use_rev); out = {}
            for q in va:
                r = RV[wording].get(str(q)) or {}
                if B[q] is None:
                    out[str(q)] = {"top": r.get("top", []), "answer_type": r.get("answer_type")}; continue
                cands, f = B[q]; s = ((f - mu) / sd) @ w
                out[str(q)] = {"top": [int(cands[j]) for j in np.argsort(-s)][:100], "answer_type": r.get("answer_type")}
            json.dump(out, open(f"{a.out_prefix}_{arm}_{wording}.json", "w"))
            print(f"[{arm}] {wording}: wrote {a.out_prefix}_{arm}_{wording}.json", flush=True)
    print("P10C_DONE")


if __name__ == "__main__":
    main()
