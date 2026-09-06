"""Lever B step 2: the reading selector. For every question and each of the four readings, features of
the reading alone (type probabilities, anchor similarities, names, walk support, reranker scores);
label = the reading with the best MRR. A listwise logistic model (softmax over the four readings) fit
on train + paraphrased train (out-of-fold parses), evaluated on val plain and paraphrased.
Usage: PYTHONPATH=lib uv run python lever_b_select.py"""
import json, csv, ast, numpy as np, torch, stark_shim
from stark_qa import load_qa
from metrics import stark_metrics, summarize
READ = ["r1", "r2", "r3", "r4"]

NT = np.load("data/kg.npz")["node_type"]; TN = {int(k): v for k, v in json.load(open("data/dicts.json"))["node_type_dict"].items()}

def load(split, V):
    P = {r: {int(x["query_id"]): ast.literal_eval(x["pred_rank"]) for x in csv.DictReader(open(f"results_b/{split}_{V}_{r}.csv"))} for r in READ}
    SC = {r: json.load(open(f"data/scB_{split}_{V}_{r}.json")) for r in READ}
    LP = {r: (json.load(open(f"data/lpB_{split}_{V}_{r}.json")) if r != "r2" else {}) for r in READ}
    suf = "_para" if V == "para" else ""
    TX = json.load(open(f"data/text_{split}_bgeft2{suf}.json")) if split == "val" else json.load(open(f"data/text_train_bgeft2oof{suf}.json"))
    U = json.load(open(f"data/text_{split}_pjoint{suf}.json")) if split == "val" else json.load(open(f"data/text_train_pjointoof{suf}.json"))
    RL = {r: json.load(open(f"data/relB_{split}_{V}_{r}.json")) if False else None for r in READ}   # answer types come from the parse files / pattern parser below
    return P, SC, LP, TX, U

def feats(qid, r, SC, LP):
    k = str(qid); sc = SC[r].get(k) or {}; lp = LP[r].get(k) or {}
    tp = lp.get("type_probs", [0, 0]); sims = lp.get("anchor_sims", [])
    top = sc.get("top", []); t1 = top[0] if top else 0.0; t2 = top[1] if len(top) > 1 else t1; t5 = top[-1] if top else t1
    f = [1.0 if r == x else 0.0 for x in READ]
    f += [tp[0] if tp else 0.0, tp[1] if len(tp) > 1 else 0.0, float(lp.get("type_rank", 0)), float(len(lp.get("anchors", []))),
          max(sims) if sims else 0.0, (sum(sims) / len(sims)) if sims else 0.0, float(sc.get("n_mentions", 0)), float(sc.get("n_supported", 0)),
          float(sc.get("max_exact", 0)), float(sc.get("n_cands", 0)), t1, t1 - t2, t1 - t5, 1.0 if top else 0.0]
    return f

def build(split, V, qa, idx):
    P, SC, LP, TX, U = load(split, V); X, Y, M, Q = [], [], [], []
    for i in idx:
        q, qid, ans, _ = qa[i]; qid = int(qid); k = str(qid)
        ms = [stark_metrics(P[r][qid], ans) for r in READ]
        F = [feats(qid, r, SC, LP) for r in READ]
        tops = [P[r][qid][0] if P[r][qid] else -1 for r in READ]; t1 = [f[14] for f in F]; base_type_p1 = F[0][4]
        tx10 = TX.get(k, [])[:10]; u10 = U.get(k, [])[:10]
        for j, r in enumerate(READ):                                   # cross-reading features: agreement and score relative to the P3 reading
            F[j] += [1.0 if tops[j] == tops[0] else 0.0, float(sum(1 for t in tops if t == tops[j]) - 1), t1[j] - t1[0], base_type_p1, 1.0 if tops[j] == tops[1] else 0.0]
            # parser-free type vote: the type of this reading's top answer vs the types of the text ranker's and the readout's top-10
            tt = int(NT[tops[j]]) if tops[j] >= 0 else -1
            F[j] += [sum(1 for c in tx10 if int(NT[c]) == tt) / 10.0, sum(1 for c in u10 if int(NT[c]) == tt) / 10.0,
                     1.0 if tops[j] in tx10 else 0.0, 1.0 if tops[j] in u10 else 0.0]
        X.append(F); Y.append([m["mrr"] for m in ms]); M.append(ms); Q.append(qid)
    return np.array(X, np.float32), np.array(Y, np.float32), M

def main():
    qa = load_qa("prime"); sp = qa.get_idx_split(); tr, va = sp["train"].tolist(), sp["val"].tolist(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = []; Ytr = []
    for V in ("plain", "para"):
        X, Y, _ = build("train", V, qa, tr); Xtr.append(X); Ytr.append(Y)
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, device=dev); Yt = torch.tensor(Ytr, device=dev)
    # target: softmax over readings weighted by MRR (listwise); ties spread the mass
    tgt = Yt / Yt.sum(1, keepdim=True).clamp_min(1e-6); tgt[Yt.sum(1) == 0] = 0.25
    w = torch.zeros(Xt.shape[-1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for step in range(600):
        s = Xt @ w + b; loss = -(tgt * torch.log_softmax(s, 1)).sum(1).mean() + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    wv = w.detach().cpu().numpy(); bv = float(b.detach().cpu())
    names = ["is_r1", "is_r2", "is_r3", "is_r4", "type_p1", "type_p2", "type_rank", "n_anchors", "max_sim", "mean_sim", "n_mentions", "n_supported", "max_exact", "n_cands", "rr_top1", "rr_margin12", "rr_spread15", "has_scores", "agree_r1", "n_agree", "top1_rel_r1", "r1_type_p1", "agree_r2", "tx_type_vote", "u_type_vote", "top1_in_tx10", "top1_in_u10"]
    print("selector weights:", {n: round(float(x), 2) for n, x in zip(names, wv)})
    for V in ("plain", "para"):
        X, Y, M = build("val", V, qa, va); s = ((X - mu) / sd) @ wv + bv; pick = s.argmax(1)
        sel = [M[j][pick[j]] for j in range(len(M))]; orc = [M[j][int(Y[j].argmax())] for j in range(len(M))]; r1 = [M[j][0] for j in range(len(M))]
        acc = np.mean([Y[j][pick[j]] >= Y[j].max() - 1e-9 for j in range(len(M))])
        print(f"== val {V}")
        print("  always r1 (P3)   :", {k: round(v * 100, 1) for k, v in summarize(r1).items()})
        print("  selector         :", {k: round(v * 100, 1) for k, v in summarize(sel).items()}, f"| picks a best reading on {acc*100:.1f}% | picks: {np.bincount(pick, minlength=4).tolist()}")
        print("  oracle           :", {k: round(v * 100, 1) for k, v in summarize(orc).items()})
    json.dump({"w": wv.tolist(), "b": bv, "mu": mu.tolist(), "sd": sd.tolist(), "names": names}, open("data/selectorB.json", "w"))
    # non-linear selector: 2-layer MLP, listwise loss, early stopping on the last 10% of TRAIN questions (val untouched)
    torch.manual_seed(0); n = Xt.shape[0]; cut = int(n * 0.9); perm = torch.randperm(n, device=dev); tr_i, es_i = perm[:cut], perm[cut:]
    mlp = torch.nn.Sequential(torch.nn.Linear(Xt.shape[-1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=2e-3, weight_decay=1e-4); best = (1e9, None)
    for ep in range(400):
        mlp.train(); s_ = mlp(Xt[tr_i]).squeeze(-1); loss = -(tgt[tr_i] * torch.log_softmax(s_, 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            mlp.eval()
            with torch.no_grad(): es = -(tgt[es_i] * torch.log_softmax(mlp(Xt[es_i]).squeeze(-1), 1)).sum(1).mean().item()
            if es < best[0]: best = (es, {k: v.clone() for k, v in mlp.state_dict().items()})
    mlp.load_state_dict(best[1]); mlp.eval()
    for V in ("plain", "para"):
        X, Y, M = build("val", V, qa, va)
        with torch.no_grad(): s_ = mlp(torch.tensor((X - mu) / sd, device=dev)).squeeze(-1).cpu().numpy()
        pick = s_.argmax(1); sel = [M[j][pick[j]] for j in range(len(M))]; acc = np.mean([Y[j][pick[j]] >= Y[j].max() - 1e-9 for j in range(len(M))])
        print(f"== val {V}: MLP selector :", {k: round(v * 100, 1) for k, v in summarize(sel).items()}, f"| picks a best reading on {acc*100:.1f}% | picks: {np.bincount(pick, minlength=4).tolist()}")
        # soft mixture: RRF of the four readings weighted by the selector's softmax
        P, _, _, _, _ = load("val", V); pr = np.exp(s_ - s_.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        for T in (1.0, 0.5):
            rows = []
            for j, i in enumerate(va):
                q, qid, ans, _ = qa[i]; qid = int(qid); w_ = pr[j] ** (1 / T); w_ /= w_.sum(); sc = {}
                for rj, r in enumerate(READ):
                    for rank, c in enumerate(P[r][qid][:100]): sc[c] = sc.get(c, 0.0) + w_[rj] / (60 + rank + 1)
                rows.append(stark_metrics([c for c, _ in sorted(sc.items(), key=lambda x: -x[1])][:100], ans))
            print(f"   soft mixture T={T}:", {k: round(v * 100, 1) for k, v in summarize(rows).items()})
    torch.save({"state": mlp.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(), "names": names}, "models/selectorB_mlp.pt")

if __name__ == "__main__":
    main()
