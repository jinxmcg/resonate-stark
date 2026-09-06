"""Lever 10: train the AND. Starts from a ResonatE checkpoint; each step = edge batch + conjunction batch
(+ question batch if --questions). Soft-AND readout over constraints in the loss. Evaluates the synthetic
exact-AND val set with sum / min / softmin for the base and the trained table, and held-out link MRR.
Usage: PYTHONPATH=lib uv run python and_train.py --steps 1000 --tag p_and"""
import argparse, json, time, sys, os, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from resonate_wiki import SparseTableResonatE, score_batch, clip_grad_norm_
from train_prime import load_kg
from joint_train import mrr_holdout
from metrics import stark_metrics, summarize

def constraint_scores(model, cons, n_rel, dev):
    """cons: list of (anchor, chain) -> (k, N) scores, each = Re<out(hop(E[a], chain)), E_t> * exp(tau)"""
    E = model.table(); S = []
    for an, chain in cons:
        z = model.embed(torch.tensor([an], device=dev)); op = None
        for (rr, d) in chain:
            op = torch.tensor([rr + (0 if d == 0 else n_rel)], device=dev); z = model.hop(z, op)
        S.append(torch.real(model.out(z, op) @ E.conj().t())[0] * model.log_tau.exp())
    return torch.stack(S)

def combine(S, agg, T=1.0):
    if agg == "sum": return S.sum(0)
    if agg == "min": return S.min(0).values
    return -T * torch.logsumexp(-S / T, 0)                              # softmin

@torch.no_grad()
def eval_conj(model, Q, n_rel, dev, type_mask=None, label=""):
    model.eval(); res = {}
    for agg in ("sum", "min", "softmin"):
        rows = []
        for q in Q:
            S = constraint_scores(model, [(c[0], [tuple(x) for x in c[1]]) for c in q["constraints"]], n_rel, dev)
            s = combine(S, agg)
            if type_mask is not None: s = s.masked_fill(~type_mask[q["answer_type"]], -1e9)
            rows.append(stark_metrics(torch.topk(s, 100).indices.tolist(), q["answers"]))
        res[agg] = summarize(rows); print(f"CONJ {label} {agg:7s}:", {k: round(v, 4) for k, v in res[agg].items()}, flush=True)
    return res

def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="models/p_joint.pt"); p.add_argument("--tag", default="p_and")
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--cbatch", type=int, default=128); p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--lr-table", type=float, default=1e-3); p.add_argument("--lr-ops", type=float, default=1e-4); p.add_argument("--wc", type=float, default=1.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=1000); p.add_argument("--typed", action="store_true", help="restrict the synthetic eval to the answer type")
    a = p.parse_args(); dev = torch.device("cuda"); torch.manual_seed(a.seed)
    N, n_rel, (h, r, t), val, node_type = load_kg(); ck = torch.load(a.model, map_location=dev, weights_only=False); ar = ck["args"]
    model = SparseTableResonatE(N, 2 * n_rel, k=ar["k"], block_size=ar["block_size"], sparse_grad=False, device=dev); model.load_state_dict(ck["model"])
    dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}
    type_mask = {v: torch.from_numpy(node_type == k).to(dev) for k, v in tn.items()} if a.typed else None
    CT = json.load(open("data/conj_train.json")); CV = json.load(open("data/conj_val.json"))[:a.n_eval]
    print(f"conjunctions: train {len(CT)} val {len(CV)}", flush=True)
    mt, mh = mrr_holdout(model, val, N, n_rel, dev); print(f"[link] before {(mt+mh)/2:.4f}", flush=True)
    eval_conj(model, CV, n_rel, dev, type_mask, "base")
    ops = [q for n_, q in model.named_parameters() if n_ != "E_real"]
    opt = torch.optim.Adam([{"params": [model.E_real], "lr": a.lr_table}, {"params": ops, "lr": a.lr_ops}]); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    hh, rr_, tt = (torch.from_numpy(x).to(dev) for x in (h, r, t)); gen = torch.Generator(device=dev); gen.manual_seed(a.seed); t0 = time.time(); model.train()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(hh), (2048,), device=dev, generator=gen); rev = torch.rand(2048, device=dev, generator=gen) < 0.5
        src = torch.where(rev, tt[idx], hh[idx]); dst = torch.where(rev, hh[idx], tt[idx]); negs = torch.randint(0, N, (4096,), device=dev, generator=gen)
        logits, z, e_pos = score_batch(model, src, rr_[idx] + rev.long() * n_rel, dst, negs)
        loss_e = F.cross_entropy(logits, torch.zeros(2048, dtype=torch.long, device=dev)) + 0.1 * (z - e_pos).abs().pow(2).sum(-1).mean()
        batch = [CT[j] for j in torch.randint(0, len(CT), (a.cbatch,), generator=torch.Generator().manual_seed(a.seed + step)).tolist()]
        loss_c = 0.0
        for q in batch:
            S = constraint_scores(model, [(c[0], [tuple(x) for x in c[1]]) for c in q["constraints"]], n_rel, dev)
            s = combine(S, "softmin", a.T); ls = torch.log_softmax(s, 0)
            loss_c = loss_c - torch.logsumexp(ls[torch.tensor(q["answers"], device=dev)], 0)
        loss_c = loss_c / len(batch); loss = loss_e + a.wc * loss_c
        opt.zero_grad(set_to_none=True); loss.backward(); clip_grad_norm_(list(model.parameters()), 1.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"step {step}/{a.steps} edge {loss_e.item():.3f} conj {float(loss_c):.3f} ({round(time.time()-t0)} s)", flush=True)
    model.eval(); mt, mh = mrr_holdout(model, val, N, n_rel, dev); print(f"[link] after {(mt+mh)/2:.4f}", flush=True)
    eval_conj(model, CV, n_rel, dev, type_mask, "trained")
    torch.save({"model": model.state_dict(), "args": ar, "N": N, "n_rel": n_rel}, f"models/{a.tag}.pt"); print("AND_DONE", round(time.time() - t0), "s", flush=True)

if __name__ == "__main__":
    main()
