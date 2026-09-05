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


def build(rel, txt, w, logdeg, qids):
    """-> {qid: (cands, feats (n,9)) } for queries with relational output; others -> None."""
    out = {}
    for qid in qids:
        r = rel.get(str(qid)); t = txt.get(str(qid), [])
        if not r or not r.get("top") or "feats" not in r:
            out[qid] = None; continue
        top = r["top"]; z = r["feats"]["z"]; ex = r["feats"]["exact"]
        cands, fs = fused(top, t, w)
        rpos = {c: i for i, c in enumerate(top)}; tpos = {c: i for i, c in enumerate(t)}
        nm = len(r.get("mentions", [])) or 1
        f = np.zeros((len(cands), 9), np.float32)
        for j, c in enumerate(cands):
            ri = rpos.get(c); ti = tpos.get(c)
            f[j] = [z[ri] if ri is not None else 0.0, ex[ri] if ri is not None else 0.0,
                    (ex[ri] / nm) if ri is not None else 0.0,
                    1.0 / (60 + ri + 1) if ri is not None else 0.0, 1.0 if ri is not None else 0.0,
                    1.0 / (60 + ti + 1) if ti is not None else 0.0, 1.0 if ti is not None else 0.0,
                    logdeg[c], fs[c]]
        out[qid] = (cands, f)
    return out


def fit(groups, dev, steps=400, lr=0.05, l2=1e-3, init=None):
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


def main():
    p = argparse.ArgumentParser(); p.add_argument("--rel-tag", required=True); p.add_argument("--text-tag", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--min-group", type=int, default=100); p.add_argument("--l2", type=float, default=1e-3)
    a = p.parse_args(); dev = torch.device(a.device)
    qa = load_qa("prime"); sp = qa.get_idx_split()
    rel = {s: json.load(open(f"data/rel_{s}_{a.rel_tag}.json")) for s in ("train", "val")}
    txt = {s: json.load(open(f"data/text_{s}_{a.text_tag}.json")) for s in ("train", "val")}
    w_rrf = json.load(open(f"data/fusion_{a.text_tag}.json"))["w"]
    d = np.load("data/kg.npz"); N = len(d["node_type"])
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    info = {s: {int(qa[i][1]): (qa[i][2], i) for i in sp[s].tolist()} for s in ("train", "val")}
    B = {s: build(rel[s], txt[s], w_rrf, logdeg, list(info[s].keys())) for s in ("train", "val")}
    allf = np.concatenate([v[1] for v in B["train"].values() if v is not None]); mu, sd = allf.mean(0), allf.std(0) + 1e-6
    gtr = []
    for qid, v in B["train"].items():
        if v is None: continue
        cands, f = v; ans = set(info["train"][qid][0])
        y = np.array([1.0 if c in ans else 0.0 for c in cands], np.float32)
        if y.sum() == 0: continue
        gtr.append((rel["train"][str(qid)]["answer_type"], (f - mu) / sd, y))
    print(f"train groups with a reachable answer: {len(gtr)} / {sum(v is not None for v in B['train'].values())} covered")
    w_g = fit([(x, y) for _, x, y in gtr], dev, l2=a.l2)
    names = ["z", "exact", "exact/nm", "rel_rrf", "in_rel", "txt_rrf", "in_txt", "logdeg", "fused"]
    print("global weights:", {n: round(float(x), 3) for n, x in zip(names, w_g)})
    w_t = {}
    for t in sorted(set(at for at, _, _ in gtr)):
        sub = [(x, y) for at, x, y in gtr if at == t]
        w_t[t] = fit(sub, dev, steps=200, l2=a.l2, init=w_g) if len(sub) >= a.min_group else w_g
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
    json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "w_global": w_g.tolist(), "w_type": {str(k): v.tolist() for k, v in w_t.items()},
               "w_rrf": w_rrf, "rel_tag": a.rel_tag, "text_tag": a.text_tag}, open(f"data/rerank_{a.rel_tag}_{a.text_tag}.json", "w"))


if __name__ == "__main__":
    main()
