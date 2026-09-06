"""Lever 10 data: conjunctive queries sampled from the TRAIN edges (2% link holdout excluded).
Each query: answer type, 2-3 constraints (anchor id, chain of (r, d) operators from the type
signatures), answers = exact intersection of the walks. Kept if 1 <= |ans| <= 50 and each single
constraint's reach is > 2x the intersection. Writes data/conj_train.json and data/conj_val.json
(disjoint anchor sets). Usage: PYTHONPATH=lib uv run python conj_sample.py --n-train 60000 --n-val 2000"""
import argparse, json, random, time, numpy as np, sys, os
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from train_prime import load_kg
from retrieve import Parser, Adjacency

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n-train", type=int, default=60000); p.add_argument("--n-val", type=int, default=2000); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    random.seed(a.seed); rng = np.random.default_rng(a.seed)
    N, n_rel, (h, r, t), val, node_type = load_kg(); kg = {"h": h, "r": r, "t": t}
    adj = Adjacency(kg, n_rel, N)
    names = json.load(open("data/names.json")); dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}; ktype = {v: k for k, v in tn.items()}
    parser = Parser(names, node_type, tn, json.load(open("data/signatures.json")))
    chains = {}                                                       # (mention type, answer type) -> list of chains
    for (mt, at), ops in parser.ops.items(): chains.setdefault((mt, at), []).extend([((rr, d),) for (rr, d) in ops])
    for (mt, at), ops2 in parser.ops2.items(): chains.setdefault((mt, at), []).extend(list(ops2))
    by_type = {ti: np.nonzero(node_type == ti)[0] for ti in tn}
    val_anchor = set(rng.choice(N, size=N // 10, replace=False).tolist())    # 10% of nodes reserved as val anchors
    out = {"train": [], "val": []}; t0 = time.time(); tries = 0
    keys = list(chains.keys())
    while len(out["train"]) < a.n_train or len(out["val"]) < a.n_val:
        tries += 1
        at = tn[random.randrange(len(tn))]; k = random.choice([2, 2, 3])
        cons, sets = [], []
        for _ in range(k):
            cand = [key for key in keys if key[1] == at]
            if not cand: break
            mt, _ = random.choice(cand); chain = random.choice(chains[(mt, at)])
            pool = by_type[ktype[mt]]
            if len(pool) == 0: break
            an = int(rng.choice(pool)); reach = set(adj.reach(an, chain).tolist())
            reach = {x for x in reach if node_type[x] == ktype[at]}
            if not reach: break
            cons.append((an, [list(op) for op in chain])); sets.append(reach)
        if len(cons) < k: continue
        inter = set.intersection(*sets)
        if not (1 <= len(inter) <= 50): continue
        if any(len(s_) <= 2 * len(inter) for s_ in sets): continue
        split = "val" if all(c[0] in val_anchor for c in cons) else ("train" if not any(c[0] in val_anchor for c in cons) else None)
        if split is None or len(out[split]) >= (a.n_val if split == "val" else a.n_train): continue
        out[split].append({"answer_type": at, "constraints": cons, "answers": sorted(inter)})
        if tries % 20000 == 0: print(f"tries {tries}: train {len(out['train'])} val {len(out['val'])} ({round(time.time()-t0)} s)", flush=True)
    for s_ in ("train", "val"): json.dump(out[s_], open(f"data/conj_{s_}.json", "w"))
    ks = [len(q["constraints"]) for q in out["train"]]; na = [len(q["answers"]) for q in out["train"]]
    print(f"CONJ_DONE train {len(out['train'])} val {len(out['val'])}; constraints mean {np.mean(ks):.2f}; answers mean {np.mean(na):.1f} median {np.median(na):.0f}; {round(time.time()-t0)} s")

if __name__ == "__main__":
    main()
