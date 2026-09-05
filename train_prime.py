"""ResonatE on the STaRK-Prime knowledge graph through the sparse shell (P1).
All 8.1M edges are the corpus; a seeded 2 % slice is held out only as a link-prediction
sanity check (MRR over 500 uniform negatives). Both directions are separate operators
(op = r for h->t, r + R for t->h). Uniform negatives over all nodes, as retrieval will rank.
Usage: uv run python train_prime.py --k 12 --block-size 4 --steps 12500 --save models/p_k12b4.pt
"""
import argparse, sys, time, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/geocore/resonate")          # shared modules: resonate.py, resonate_wiki.py, rowadagrad.py
from resonate_wiki import SparseTableResonatE, score_batch, clip_grad_norm_
from rowadagrad import RowAdagrad


def load_kg(path="data/kg.npz", holdout=0.02, seed=0):
    d = np.load(path)
    h, r, t = d["h"], d["r"], d["t"]; n_rel = int(d["n_rel"]); N = len(d["node_type"])
    rng = np.random.default_rng(seed)
    m = rng.random(len(h)) < holdout
    return N, n_rel, (h[~m], r[~m], t[~m]), (h[m], r[m], t[m]), d["node_type"]


@torch.no_grad()
def mrr_holdout(model, val, N, n_rel, dev, n=5000, negs=500, seed=123):
    h, r, t = val; rng = np.random.default_rng(seed)
    idx = rng.choice(len(h), size=min(n, len(h)), replace=False)
    out = []
    for rev in (False, True):
        src = torch.from_numpy(t[idx] if rev else h[idx]).to(dev)
        dst = torch.from_numpy(h[idx] if rev else t[idx]).to(dev)
        rel = torch.from_numpy(r[idx] + (n_rel if rev else 0)).to(dev)
        ng = torch.from_numpy(rng.integers(0, N, size=(len(idx), negs))).to(dev)
        z = model.out(model.hop(model.embed(src), rel), rel)
        sp = torch.real((z * model.rows(dst).conj()).sum(-1)) * model.log_tau.exp()
        sn = torch.real((z[:, None, :] * model.rows(ng).conj()).sum(-1)) * model.log_tau.exp()
        rank = 1 + (sn > sp[:, None]).sum(1)
        out.append((1.0 / rank.float()).mean().item())
    return out  # [tail-side MRR, head-side MRR]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=12); p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=12500); p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--neg", type=int, default=4096); p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--table-lr", type=float, default=0.3); p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--rev-frac", type=float, default=0.5); p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--save", required=True); p.add_argument("--log-every", type=int, default=1000)
    a = p.parse_args()
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    N, n_rel, (h, r, t), val, node_type = load_kg()
    print(f"PrimeKG: {N:,} nodes, {n_rel} relations ({2*n_rel} operators), {len(h):,} train edges, {len(val[0]):,} held out", flush=True)
    model = SparseTableResonatE(N, 2 * n_rel, k=a.k, block_size=a.block_size, sparse_grad=True, device=dev)
    model.train()
    print(f"params {model.n_params():,} (M={model.m}, block {a.block_size}x{a.block_size})", flush=True)
    h, r, t = (torch.from_numpy(x).to(dev) for x in (h, r, t))
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    opts = [torch.optim.Adam(model.other_params(), lr=a.lr), RowAdagrad(model.table_params(), lr=a.table_lr)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.steps) for o in opts]
    params = list(model.parameters()); t0 = time.time(); n_train = len(h)
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, n_train, (a.batch,), device=dev, generator=gen)
        rev = torch.rand(a.batch, device=dev, generator=gen) < a.rev_frac
        hb, rb, tb = h[idx], r[idx], t[idx]
        src = torch.where(rev, tb, hb); dst = torch.where(rev, hb, tb); rel = rb + rev.long() * n_rel
        negs = torch.randint(0, N, (a.neg,), device=dev, generator=gen)
        logits, z, e_pos = score_batch(model, src, rel, dst, negs)
        loss = F.cross_entropy(logits, torch.zeros(a.batch, dtype=torch.long, device=dev))
        loss = loss + a.lam * (z - e_pos).abs().pow(2).sum(-1).mean()
        for o in opts: o.zero_grad(set_to_none=True)
        loss.backward(); clip_grad_norm_(params, a.clip)
        for o in opts: o.step()
        for s in scheds: s.step()
        if step % a.log_every == 0 or step == a.steps:
            print(f"step {step}/{a.steps}  loss {loss.item():.3f}  tau {model.log_tau.exp().item():.2f}  ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    mt, mh = mrr_holdout(model, val, N, n_rel, dev)
    print(f"[holdout] MRR over 500 uniform negatives: tail-side {mt:.4f}  head-side {mh:.4f}  mean {(mt+mh)/2:.4f}", flush=True)
    torch.save({"model": model.state_dict(), "args": vars(a), "N": N, "n_rel": n_rel}, a.save)
    print("saved", a.save, flush=True)


if __name__ == "__main__":
    main()
