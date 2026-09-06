"""RotatE (L2 variant) on the STaRK-Prime graph with train_prime.py's regime (lever 8 control).
Entities: complex (N, M); relation: phase vector theta_r (M,), reverse = -theta; score = gamma - sum_m
|h_m e^{i theta} - t_m|. Same batch / negatives / steps / holdout as train_prime.py. Saves
models/rotate.pt {"E": real view (N, 2M), "theta": (n_rel, M), "gamma", "N", "n_rel", "M"}."""
import argparse, time, sys, os, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from train_prime import load_kg

def rot(h, theta):                      # h complex (B, M); theta (B, M)
    return h * torch.polar(torch.ones_like(theta), theta)

def dist(z, t):                         # L2 distance between complex vectors (RotatE's L2 variant; the L1-of-moduli form costs ~15x more per step here)
    return torch.linalg.vector_norm(torch.view_as_real(z - t), dim=(-2, -1))

def dist_all(z, E):                     # L2 distances of every z (B, M) to every row of E (K, M) through one matmul
    zr = torch.view_as_real(z).reshape(z.shape[0], -1); Er = torch.view_as_real(E).reshape(E.shape[0], -1)
    d2 = (zr * zr).sum(1, keepdim=True) + (Er * Er).sum(1)[None, :] - 2 * zr @ Er.t()
    return d2.clamp_min(0).sqrt()

@torch.no_grad()
def mrr_holdout(E, theta, gamma, val, N, n_rel, dev, n=5000, negs=500, seed=123, chunk=250):
    h, r, t = val; rng = np.random.default_rng(seed); idx = rng.choice(len(h), size=min(n, len(h)), replace=False); out = []
    for rev in (False, True):
        src = torch.from_numpy(t[idx] if rev else h[idx]).to(dev); dst = torch.from_numpy(h[idx] if rev else t[idx]).to(dev)
        rel = torch.from_numpy(r[idx]).to(dev); ng = torch.from_numpy(rng.integers(0, N, size=(len(idx), negs))).to(dev); rr = []
        for b in range(0, len(idx), chunk):
            th = theta[rel[b:b+chunk]] * (-1 if rev else 1); z = rot(E[src[b:b+chunk]], th)
            sp = gamma - dist(z, E[dst[b:b+chunk]]); sn = gamma - torch.stack([dist(z[j][None], E[ng[b+j]]) for j in range(len(z))]) if False else gamma - dist(z[:, None, :], E[ng[b:b+chunk]])
            rr.append(1.0 / (1 + (sn > sp[:, None]).sum(1)).float())
        out.append(torch.cat(rr).mean().item())
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=144); p.add_argument("--steps", type=int, default=50000); p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--neg", type=int, default=4096); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--gamma", type=float, default=12.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="models/rotate.pt"); a = p.parse_args(); dev = torch.device("cuda"); torch.manual_seed(a.seed)
    N, n_rel, (h, r, t), val, _ = load_kg(); print(f"PrimeKG: {N:,} nodes, {n_rel} relations, {len(h):,} train edges, {len(val[0]):,} held out", flush=True)
    E_real = torch.nn.Parameter((torch.rand(N, 2 * a.M, device=dev) * 2 - 1) * (a.gamma + 2) / a.M)     # RotatE-style init range
    theta = torch.nn.Parameter((torch.rand(n_rel, a.M, device=dev) * 2 - 1) * np.pi)
    opt = torch.optim.Adam([E_real, theta], lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    h, r, t = (torch.from_numpy(x).to(dev) for x in (h, r, t)); gen = torch.Generator(device=dev); gen.manual_seed(a.seed); t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(h), (a.batch,), device=dev, generator=gen); rev = torch.rand(a.batch, device=dev, generator=gen) < 0.5
        src = torch.where(rev, t[idx], h[idx]); dst = torch.where(rev, h[idx], t[idx]); th = theta[r[idx]] * torch.where(rev, -1.0, 1.0)[:, None]
        negs = torch.randint(0, N, (a.neg,), device=dev, generator=gen)
        E = torch.view_as_complex(E_real.view(N, a.M, 2)); z = rot(E[src], th)
        sp = a.gamma - dist(z, E[dst])
        sn = a.gamma - dist_all(z, E[negs])
        loss = F.cross_entropy(torch.cat([sp[:, None], sn], 1), torch.zeros(a.batch, dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps: print(f"step {step}/{a.steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    E = torch.view_as_complex(E_real.detach().view(N, a.M, 2))
    mt, mh = mrr_holdout(E, theta.detach(), a.gamma, val, N, n_rel, dev); print(f"[holdout] RotatE MRR over 500 uniform negatives: tail {mt:.4f} head {mh:.4f} mean {(mt+mh)/2:.4f}", flush=True)
    torch.save({"E": E_real.detach().cpu(), "theta": theta.detach().cpu(), "gamma": a.gamma, "N": N, "n_rel": n_rel, "M": a.M}, a.save); print("ROTATE_DONE", round(time.time() - t0), "s")

if __name__ == "__main__":
    main()
