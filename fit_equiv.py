"""Equivalence check for rerank.fit's batched path against the original per-group loop.
Same groups, same steps, same seed-free initialisation: the fitted weights must agree.
Usage: PYTHONPATH=lib uv run python fit_equiv.py --rel-tag p8A_lp_ancf --text-tag bgeft2 --t2l-tag pjoint"""
import argparse, json, time, numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from rerank import build, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rel-tag", required=True); p.add_argument("--text-tag", required=True); p.add_argument("--t2l-tag", default=None)
    p.add_argument("--groups", type=int, default=1500); p.add_argument("--device", default="cuda")
    a = p.parse_args(); dev = torch.device(a.device)
    qa = load_qa("prime"); sp = qa.get_idx_split()
    rel = json.load(open(f"data/rel_train_{a.rel_tag}.json")); txt = json.load(open(f"data/text_train_bgeft2oof.json"))
    t2l = json.load(open(f"data/text_train_{a.t2l_tag}oof.json")) if a.t2l_tag else None
    w_rrf = json.load(open(f"data/fusion_{a.text_tag}.json"))["w"]
    d = np.load("data/kg.npz"); N = len(d["node_type"])
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    info = {int(qa[i][1]): qa[i][2] for i in sp["train"].tolist()}
    B = build(rel, txt, w_rrf, logdeg, list(info.keys()), t2l)
    gtr = []
    for qid, v in B.items():
        if v is None: continue
        cands, f = v; ans = set(info[qid])
        y = np.array([1.0 if c in ans else 0.0 for c in cands], np.float32)
        if y.sum() == 0: continue
        gtr.append((f, y))
        if len(gtr) >= a.groups: break
    allf = np.concatenate([x for x, _ in gtr]); mu, sd = allf.mean(0), allf.std(0) + 1e-6
    G = [((x - mu) / sd, y) for x, y in gtr]
    print(f"{len(G)} groups, {G[0][0].shape[1]} features", flush=True)
    t0 = time.time(); w_slow = fit(G, dev, slow=True); t_slow = time.time() - t0
    t0 = time.time(); w_fast = fit(G, dev); t_fast = time.time() - t0
    d_abs = np.abs(w_slow - w_fast).max(); d_rel = d_abs / max(np.abs(w_slow).max(), 1e-9)
    print("slow:", np.round(w_slow, 4))
    print("fast:", np.round(w_fast, 4))
    print(f"max |difference| {d_abs:.2e} (relative {d_rel:.2e});  {t_slow:.1f}s -> {t_fast:.1f}s ({t_slow/max(t_fast,1e-9):.0f}x)")
    print("FIT_EQUIV_OK" if d_rel < 1e-3 else f"FIT_EQUIV_FAILED relative {d_rel:.2e}")


if __name__ == "__main__":
    main()
