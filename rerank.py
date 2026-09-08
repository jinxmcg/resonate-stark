"""P2 lever 1: a learned reranker over the fused candidate set, fit on STaRK `train` queries.
Candidate set per query = RRF fusion of the relational top-100 and the text top-100 (same as
predict.py). Features per candidate: relational z-score sum and exact-support count (from
retrieve.py --dump-feats), relational RRF term and in-relational flag, text RRF term and in-text
flag, log total degree, number of mentions, fused RRF score. Model: listwise logistic regression
(softmax over the query's candidates, cross-entropy on the true answers) with one weight vector per
answer type initialised from a global fit, L2, all fit on `train`; reported on `val`. Queries with
no relational output keep the fused (text) order. Never reads test.
Usage: PYTHONPATH=lib uv run python rerank.py --rel-tag ancft --text-tag bgeft
"""
import argparse, json, numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from metrics import stark_metrics, summarize


def fused(rel, txt, w, k0=60):
    s = {}
    for i, c in enumerate(rel): s[c] = s.get(c, 0) + w / (k0 + i + 1)
    for i, c in enumerate(txt): s[c] = s.get(c, 0) + (1 - w) / (k0 + i + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: -x[1])], s


NF = 11


def build(rel, txt, w, logdeg, qids, t2l=None):
    """-> {qid: (cands, feats (n,NF)) } for queries with relational output; others -> None.
    t2l: optional third ranking (text-to-latent top-100): its candidates join the set, two features."""
    out = {}
    for qid in qids:
        r = rel.get(str(qid)); t = txt.get(str(qid), [])
        if not r or not r.get("top") or "feats" not in r:
            out[qid] = None; continue
        top = r["top"]; z = r["feats"]["z"]; ex = r["feats"]["exact"]
        cands, fs = fused(top, t, w)
        u = (t2l or {}).get(str(qid), [])
        seen = set(cands); cands = cands + [c for c in u if c not in seen]
        rpos = {c: i for i, c in enumerate(top)}; tpos = {c: i for i, c in enumerate(t)}; upos = {c: i for i, c in enumerate(u)}
        nm = len(r.get("mentions", [])) or 1
        f = np.zeros((len(cands), NF), np.float32)
        for j, c in enumerate(cands):
            ri = rpos.get(c); ti = tpos.get(c); ui = upos.get(c)
            f[j] = [z[ri] if ri is not None else 0.0, ex[ri] if ri is not None else 0.0,
                    (ex[ri] / nm) if ri is not None else 0.0,
                    1.0 / (60 + ri + 1) if ri is not None else 0.0, 1.0 if ri is not None else 0.0,
                    1.0 / (60 + ti + 1) if ti is not None else 0.0, 1.0 if ti is not None else 0.0,
                    logdeg[c], fs.get(c, 0.0),
                    1.0 / (60 + ui + 1) if ui is not None else 0.0, 1.0 if ui is not None else 0.0]
        out[qid] = (cands, f)
    return out


def fit(groups, dev, steps=400, lr=0.05, l2=1e-3, init=None, slow=False):
    """Listwise logistic regression: softmax over each query's candidates, cross-entropy on its
    answers, L2. `slow` keeps the original per-group Python loop; the default pads the groups into
    one (G, Cmax, F) tensor with a mask and runs the same loss as a single batched step — identical
    mathematics (verified to 1e-5 on the weights by scripts/fit_equiv.py), ~100x fewer kernels."""
    if slow:
        X = [torch.from_numpy(x).to(dev) for x, y in groups]; Y = [torch.from_numpy(y).to(dev) for x, y in groups]
        M = X[0].shape[1]
        w = torch.zeros(M, device=dev) if init is None else torch.tensor(init, device=dev, dtype=torch.float32).clone()
        w.requires_grad_(True); opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            opt.zero_grad(); loss = 0.0
            for x, y in zip(X, Y):
                ls = torch.log_softmax(x @ w, 0)
                loss = loss - (ls * y).sum() / y.sum()
            loss = loss / len(X) + l2 * (w * w).sum(); loss.backward(); opt.step()
        return w.detach().cpu().numpy()
    G = len(groups); M = groups[0][0].shape[1]; C = max(len(y) for _, y in groups)
    Xp = np.zeros((G, C, M), np.float32); Yp = np.zeros((G, C), np.float32); Ms = np.zeros((G, C), bool)
    for g, (x, y) in enumerate(groups):
        n = len(y); Xp[g, :n] = x; Yp[g, :n] = y; Ms[g, :n] = True
    X = torch.from_numpy(Xp).to(dev); Y = torch.from_numpy(Yp).to(dev); Mk = torch.from_numpy(Ms).to(dev)
    ny = Y.sum(1)
    w = torch.zeros(M, device=dev) if init is None else torch.tensor(init, device=dev, dtype=torch.float32).clone()
    w.requires_grad_(True); opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        ls = torch.log_softmax((X @ w).masked_fill(~Mk, -float("inf")), 1)
        loss = -((ls * Y).sum(1) / ny).mean() + l2 * (w * w).sum()
        loss.backward(); opt.step()
    return w.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--rel-tag", required=True); p.add_argument("--text-tag", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--min-group", type=int, default=100); p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--drop", default="", help="ablation: comma-separated feature names to zero out (z,exact,exact/nm,rel_rrf,in_rel,txt_rrf,in_txt,logdeg,fused)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--slow-fit", action="store_true", help="the original per-group fit loop (reproduces earlier runs exactly)")
    p.add_argument("--aug-rel-tag", default=None, help="extra train groups from paraphrased train: data/rel_train_<tag>.json")
    p.add_argument("--aug-text-tag", default=None, help="... and data/text_train_<tag>.json")
    p.add_argument("--t2l-tag", default=None, help="third ranking: data/text_{train,val}_<tag>.json (text-to-latent)")
    p.add_argument("--aug-t2l-tag", default=None, help="text-to-latent ranking of the paraphrased train questions")
    p.add_argument("--train-text-tag", default=None, help="tag of the TRAIN text ranking when it differs from val (out-of-fold features)")
    p.add_argument("--train-t2l-tag", default=None, help="tag of the TRAIN text-to-latent ranking when it differs from val (out-of-fold)")
    p.add_argument("--save-tag", default="", help="suffix for the saved reranker file")
    a = p.parse_args(); dev = torch.device(a.device)
    qa = load_qa("prime"); sp = qa.get_idx_split()
    rel = {s: json.load(open(f"data/rel_{s}_{a.rel_tag}.json")) for s in ("train", "val")}
    txt = {"train": json.load(open(f"data/text_train_{a.train_text_tag or a.text_tag}.json")), "val": json.load(open(f"data/text_val_{a.text_tag}.json"))}
    w_rrf = json.load(open(f"data/fusion_{a.text_tag}.json"))["w"]
    d = np.load("data/kg.npz"); N = len(d["node_type"])
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    info = {s: {int(qa[i][1]): (qa[i][2], i) for i in sp[s].tolist()} for s in ("train", "val")}
    T2L = {"train": json.load(open(f"data/text_train_{a.train_t2l_tag or a.t2l_tag}.json")), "val": json.load(open(f"data/text_val_{a.t2l_tag}.json"))} if a.t2l_tag else {"train": None, "val": None}
    B = {s: build(rel[s], txt[s], w_rrf, logdeg, list(info[s].keys()), T2L[s]) for s in ("train", "val")}
    if a.aug_rel_tag:
        arel = json.load(open(f"data/rel_train_{a.aug_rel_tag}.json")); atxt = json.load(open(f"data/text_train_{a.aug_text_tag}.json"))
        at2l = json.load(open(f"data/text_train_{a.aug_t2l_tag}.json")) if a.aug_t2l_tag else None
        aug = build(arel, atxt, w_rrf, logdeg, list(info["train"].keys()), at2l)
        B["train"] = {**B["train"], **{-qid: v for qid, v in aug.items()}}      # negative keys: paraphrased copies
        rel["train"] = {**rel["train"], **{str(-int(k)): v for k, v in arel.items()}}
        info["train"] = {**info["train"], **{-qid: v for qid, v in info["train"].items()}}
        print("augmented with paraphrased train:", sum(v is not None for v in aug.values()), "groups")
    names = ["z", "exact", "exact/nm", "rel_rrf", "in_rel", "txt_rrf", "in_txt", "logdeg", "fused", "t2l_rrf", "in_t2l"]
    drop = [names.index(x) for x in a.drop.split(",") if x]
    for D in B.values():
        for v in D.values():
            if v is not None: v[1][:, drop] = 0.0
    allf = np.concatenate([v[1] for v in B["train"].values() if v is not None]); mu, sd = allf.mean(0), allf.std(0) + 1e-6
    gtr = []
    for qid, v in B["train"].items():
        if v is None: continue
        cands, f = v; ans = set(info["train"][qid][0])
        y = np.array([1.0 if c in ans else 0.0 for c in cands], np.float32)
        if y.sum() == 0: continue
        gtr.append((rel["train"][str(qid)]["answer_type"], (f - mu) / sd, y))
    print(f"train groups with a reachable answer: {len(gtr)} / {sum(v is not None for v in B['train'].values())} covered")
    w_g = fit([(x, y) for _, x, y in gtr], dev, l2=a.l2, slow=a.slow_fit)
    print("global weights:", {n: round(float(x), 3) for n, x in zip(names, w_g)})
    w_t = {}
    for t in sorted(set(at for at, _, _ in gtr)):
        sub = [(x, y) for at, x, y in gtr if at == t]
        w_t[t] = fit(sub, dev, steps=200, l2=a.l2, init=w_g, slow=a.slow_fit) if len(sub) >= a.min_group else w_g
    for tag, use_type in (("global", False), ("per-type", True)):
        rows_base, rows_rr = [], []
        for qid, (ans, i) in info["val"].items():
            v = B["val"][qid]
            if v is None:
                r = rel["val"].get(str(qid), {}).get("top", []); ranked, _ = fused(r, txt["val"].get(str(qid), []), w_rrf)
                rows_base.append(stark_metrics(ranked[:100], ans)); rows_rr.append(stark_metrics(ranked[:100], ans)); continue
            cands, f = v; at = rel["val"][str(qid)]["answer_type"]
            wv = w_t.get(at, w_g) if use_type else w_g
            s = ((f - mu) / sd) @ wv
            rows_base.append(stark_metrics(cands[:100], ans))
            rows_rr.append(stark_metrics([cands[i_] for i_ in np.argsort(-s)][:100], ans))
        if tag == "global":
            print("VAL fused (input)   :", {k: round(v, 4) for k, v in summarize(rows_base).items()})
        print(f"VAL reranked {tag:8s}:", {k: round(v, 4) for k, v in summarize(rows_rr).items()})
    if a.no_save: return
    json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "w_global": w_g.tolist(), "w_type": {str(k): v.tolist() for k, v in w_t.items()},
               "w_rrf": w_rrf, "rel_tag": a.rel_tag, "text_tag": a.text_tag, "t2l_tag": a.t2l_tag},
              open(f"data/rerank_{a.rel_tag}_{a.text_tag}{'_' + a.t2l_tag if a.t2l_tag else ''}{'_aug' if a.aug_rel_tag else ''}{a.save_tag}.json", "w"))


if __name__ == "__main__":
    main()
