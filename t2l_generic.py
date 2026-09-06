"""Lever 8: the identical STaRK projector on top of different tables (ResonatE or RotatE), frozen or
joint. Same encoder, head, loss, data, epochs and learning rates as text2latent.py / joint_train.py.
Readout = the table's own scoring of a query vector: ResonatE 'dot' (unit-norm z, Re<z,E_t>*exp(s));
RotatE 'dist' (gamma - L1 of moduli). 'dot' on RotatE is a control.
Usage: PYTHONPATH=lib uv run python t2l_generic.py --table rotate --readout dist [--joint] --tag t2l_rotate"""
import argparse, json, time, sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from resonate import cnorm
from resonate_wiki import SparseTableResonatE, score_batch, clip_grad_norm_
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModel
from metrics import stark_metrics, summarize
from train_prime import load_kg
from rotate_prime import rot, dist, dist_all, mrr_holdout as rot_mrr
from joint_train import mrr_holdout as res_mrr
QPRE = "Represent this sentence for searching relevant passages: "

class Proj(nn.Module):
    def __init__(self, enc_path, m, readout):
        super().__init__(); self.enc = AutoModel.from_pretrained(enc_path); self.head = nn.Linear(self.enc.config.hidden_size, 2 * m)
        self.scale = nn.Parameter(torch.tensor(3.0)); self.m = m; self.readout = readout
    def forward(self, enc):
        z = torch.view_as_complex(self.head(self.enc(**enc).last_hidden_state[:, 0]).view(-1, self.m, 2).contiguous())
        return cnorm(z) if self.readout == "dot" else z
    def score(self, z, E, gamma=12.0):
        if self.readout == "dot": return torch.real(z @ E.conj().t()) * self.scale.exp()
        return gamma - dist_all(z, E)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--table", choices=["resonate", "rotate"], required=True); p.add_argument("--readout", choices=["dot", "dist"], required=True)
    p.add_argument("--ckpt", default=None); p.add_argument("--encoder", default="models/bge_ft2"); p.add_argument("--extra", default="data/para_train.json")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch", type=int, default=32); p.add_argument("--joint", action="store_true"); p.add_argument("--tag", required=True); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = torch.device("cuda"); torch.manual_seed(a.seed)
    N, n_rel, (h, r, t), val, node_type = load_kg(); ck_path = a.ckpt or ("models/p_k12b4_50k.pt" if a.table == "resonate" else "models/rotate.pt")
    if a.table == "resonate":
        ck = torch.load(ck_path, map_location=dev, weights_only=False); ar = ck["args"]
        model = SparseTableResonatE(N, 2 * n_rel, k=ar["k"], block_size=ar["block_size"], sparse_grad=False, device=dev); model.load_state_dict(ck["model"]); model.eval()
        table = lambda: model.table(); M = model.m; gamma = 12.0
        link = lambda: res_mrr(model, val, N, n_rel, dev); tparams = list(model.parameters())
    else:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False); M = ck["M"]; gamma = ck["gamma"]
        E_real = nn.Parameter(ck["E"].to(dev)); theta = nn.Parameter(ck["theta"].to(dev))
        table = lambda: torch.view_as_complex(E_real.view(N, M, 2)); link = lambda: rot_mrr(table().detach(), theta.detach(), gamma, val, N, n_rel, dev); tparams = [E_real, theta]
    mt, mh = link(); print(f"[link] {a.table} before: held-out MRR mean {(mt+mh)/2:.4f}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.encoder); net = Proj(a.encoder, M, a.readout).to(dev)
    qa = load_qa("prime"); sp = qa.get_idx_split(); EXTRA = json.load(open(a.extra)) if a.extra else {}; train = []
    for i in sp["train"].tolist():
        q, qid, ans, _ = qa[i]; train.append((q, ans))
        if str(int(qid)) in EXTRA: train.append((EXTRA[str(int(qid))], ans))
    groups = [{"params": net.enc.parameters(), "lr": 2e-5}, {"params": list(net.head.parameters()) + [net.scale], "lr": 1e-3}]
    if a.joint: groups += [{"params": tparams, "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01); steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if a.joint else torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    hh, rr_, tt = (torch.from_numpy(x).to(dev) for x in (h, r, t)); gen = torch.Generator(device=dev); gen.manual_seed(a.seed); t0 = time.time(); step = 0
    net.train()
    for ep in range(a.epochs):
        perm = torch.randperm(len(train)).tolist(); lq = le = 0.0; nb = 0
        for b in range(0, len(perm), a.batch):
            loss_e = torch.tensor(0.0, device=dev)
            if a.joint:
                idx = torch.randint(0, len(hh), (2048,), device=dev, generator=gen); rev = torch.rand(2048, device=dev, generator=gen) < 0.5
                src = torch.where(rev, tt[idx], hh[idx]); dst = torch.where(rev, hh[idx], tt[idx]); negs = torch.randint(0, N, (4096,), device=dev, generator=gen)
                if a.table == "resonate":
                    model.train(); logits, z, e_pos = score_batch(model, src, rr_[idx] + rev.long() * n_rel, dst, negs)
                    loss_e = F.cross_entropy(logits, torch.zeros(2048, dtype=torch.long, device=dev)) + 0.1 * (z - e_pos).abs().pow(2).sum(-1).mean()
                else:
                    E = table(); th = theta[rr_[idx]] * torch.where(rev, -1.0, 1.0)[:, None]; z = rot(E[src], th)
                    sp_ = gamma - dist(z, E[dst]); sn = gamma - dist_all(z, E[negs])
                    loss_e = F.cross_entropy(torch.cat([sp_[:, None], sn], 1), torch.zeros(2048, dtype=torch.long, device=dev))
            batch = [train[j] for j in perm[b:b + a.batch]]
            enc = tok([QPRE + q for q, _ in batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            E = table() if a.joint else table().detach()
            s = net.score(net(enc), E, gamma); ls = torch.log_softmax(s, 1); pos = torch.zeros_like(s, dtype=torch.bool)
            for k_, (_, ans) in enumerate(batch): pos[k_, torch.tensor(ans, device=dev)] = True
            loss_q = -(torch.logsumexp(ls.masked_fill(~pos, -1e9), 1)).mean(); loss = loss_e + loss_q
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if a.joint: torch.nn.utils.clip_grad_norm_(tparams, 1.0)
            opt.step(); sched.step(); step += 1; lq += loss_q.item(); le += float(loss_e); nb += 1
            if step % 200 == 0: print(f"ep {ep} step {step}/{steps} question {loss_q.item():.3f} edge {float(loss_e):.3f} ({round(time.time()-t0)} s)", flush=True)
        print(f"epoch {ep} mean question loss {lq/nb:.3f} edge {le/nb:.3f}", flush=True)
    net.eval()
    if a.table == "resonate": model.eval()
    mt, mh = link(); print(f"[link] {a.table} after: held-out MRR mean {(mt+mh)/2:.4f}", flush=True)
    E = table().detach()
    for split_tag, qfile in (("plain", None), ("para", "data/para_val.json")):
        QS = json.load(open(qfile)) if qfile else {}; idx = sp["val"].tolist(); rows = []
        for b in range(0, len(idx), 64):
            chunk = idx[b:b + 64]
            with torch.no_grad():
                enc = tok([QPRE + QS.get(str(int(qa[i][1])), qa[i][0]) for i in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                top = torch.topk(net.score(net(enc), E, gamma), 100, 1).indices.cpu().numpy()
            for k_, i in enumerate(chunk): rows.append(stark_metrics(top[k_].tolist(), qa[i][2]))
        print(f"QA {a.tag} ({a.table}, {a.readout}, {'joint' if a.joint else 'frozen'}) val {split_tag:5s}:", {k: round(v, 4) for k, v in summarize(rows).items()}, flush=True)
    print("T2LG_DONE", round(time.time() - t0), "s")

if __name__ == "__main__":
    main()
