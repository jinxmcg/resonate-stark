"""P2 lever 5: train the ResonatE table on graph edges AND questions at once (see PLAN_PRIME.md).
Starts from a trained graph model; every step takes one edge batch (train_prime.py loss) and one
question batch (text-to-latent loss); table and operators receive both gradients.
Outputs: models/<tag>.pt (SparseTableResonatE checkpoint, loadable by retrieve.py), models/<tag>_enc
(encoder) + models/<tag>_head.pt; prints held-out link MRR before/after and QA on plain/paraphrased val.
Usage: PYTHONPATH=lib uv run python joint_train.py --tag p_joint --extra data/para_train.json
       PYTHONPATH=lib uv run python joint_train.py --eval-table models/t2lj.pt   (link MRR of a saved table only)"""
import argparse, json, time, sys, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from resonate import cnorm
from resonate_wiki import SparseTableResonatE, score_batch, clip_grad_norm_
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModel
from metrics import stark_metrics, summarize
from train_prime import load_kg


@torch.no_grad()
def mrr_holdout(model, val, N, n_rel, dev, n=5000, negs=500, seed=123, chunk=250):
    """train_prime.mrr_holdout, chunked (the one-shot version needs ~3 GB and shares the card)."""
    h, r, t = val; rng = np.random.default_rng(seed)
    idx = rng.choice(len(h), size=min(n, len(h)), replace=False); out = []
    for rev in (False, True):
        src = torch.from_numpy(t[idx] if rev else h[idx]).to(dev); dst = torch.from_numpy(h[idx] if rev else t[idx]).to(dev)
        rel = torch.from_numpy(r[idx] + (n_rel if rev else 0)).to(dev); ng = torch.from_numpy(rng.integers(0, N, size=(len(idx), negs))).to(dev)
        rr = []
        for b in range(0, len(idx), chunk):
            z = model.out(model.hop(model.embed(src[b:b+chunk]), rel[b:b+chunk]), rel[b:b+chunk])
            sp = torch.real((z * model.rows(dst[b:b+chunk]).conj()).sum(-1)) * model.log_tau.exp()
            sn = torch.real((z[:, None, :] * model.rows(ng[b:b+chunk]).conj()).sum(-1)) * model.log_tau.exp()
            rr.append(1.0 / (1 + (sn > sp[:, None]).sum(1)).float())
        out.append(torch.cat(rr).mean().item())
    return out
from text2latent import T2L, QPRE


def qa_eval(net, tok, E, qa, sp, dev, node_type, ktype, label):
    rel_val = json.load(open("data/rel_val_ancf.json"))
    for split_tag, qfile in (("plain", None), ("para", "data/para_val.json")):
        QS = json.load(open(qfile)) if qfile else {}; idx = sp["val"].tolist(); rows, rows_t = [], []
        for b in range(0, len(idx), 64):
            chunk = idx[b:b + 64]
            with torch.no_grad():
                enc = tok([QPRE + QS.get(str(int(qa[i][1])), qa[i][0]) for i in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                s = net.score(net(enc), E)
            top = torch.topk(s, 100, 1).indices.cpu().numpy()
            for r, i in enumerate(chunk):
                q, qid, ans, _ = qa[i]; rows.append(stark_metrics(top[r].tolist(), ans))
                at = rel_val.get(str(int(qid)), {}).get("answer_type")
                if at is not None:
                    st = s[r].clone(); st[node_type != ktype[at]] = -1e9; tt = torch.topk(st, 100).indices.cpu().numpy().tolist()
                else: tt = top[r].tolist()
                rows_t.append(stark_metrics(tt, ans))
        print(f"QA {label} val {split_tag:5s} all entities :", {k: round(v, 4) for k, v in summarize(rows).items()}, flush=True)
        print(f"QA {label} val {split_tag:5s} parser type  :", {k: round(v, 4) for k, v in summarize(rows_t).items()}, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/p_k12b4_50k.pt"); p.add_argument("--encoder", default="models/bge_ft2")
    p.add_argument("--extra", default=None); p.add_argument("--epochs", type=int, default=3); p.add_argument("--qbatch", type=int, default=32)
    p.add_argument("--ebatch", type=int, default=2048); p.add_argument("--neg", type=int, default=4096); p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--lr-table", type=float, default=1e-3); p.add_argument("--lr-ops", type=float, default=1e-4); p.add_argument("--lr-enc", type=float, default=2e-5); p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--wq", type=float, default=1.0, help="weight of the question loss"); p.add_argument("--tag", default="p_joint"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-table", default=None, help="only measure held-out link MRR of the table stored in this text2latent checkpoint")
    a = p.parse_args(); dev = torch.device("cuda"); torch.manual_seed(a.seed)
    N, n_rel, (h, r, t), val, node_type_np = load_kg()
    ck = torch.load(a.model, map_location=dev, weights_only=False); ar = ck["args"]
    model = SparseTableResonatE(N, 2 * n_rel, k=ar["k"], block_size=ar["block_size"], sparse_grad=False, device=dev)
    model.load_state_dict(ck["model"]); model.eval()
    mt, mh = mrr_holdout(model, val, N, n_rel, dev); print(f"[link] base table: held-out MRR tail {mt:.4f} head {mh:.4f} mean {(mt+mh)/2:.4f}", flush=True)
    if a.eval_table:
        E2 = torch.load(a.eval_table, map_location="cpu", weights_only=False)["E"]
        with torch.no_grad(): model.E_real.copy_(E2.reshape(N, -1).to(dev))
        mt, mh = mrr_holdout(model, val, N, n_rel, dev); print(f"[link] table from {a.eval_table}: held-out MRR tail {mt:.4f} head {mh:.4f} mean {(mt+mh)/2:.4f}", flush=True)
        return
    node_type = torch.from_numpy(node_type_np).to(dev)
    tn = {int(k): v for k, v in json.load(open("data/dicts.json"))["node_type_dict"].items()}; ktype = {v: k for k, v in tn.items()}
    tok = AutoTokenizer.from_pretrained(a.encoder); net = T2L(a.encoder, model.m).to(dev)
    qa = load_qa("prime"); sp = qa.get_idx_split(); EXTRA = json.load(open(a.extra)) if a.extra else {}
    train = []
    for i in sp["train"].tolist():
        q, qid, ans, _ = qa[i]; train.append((q, ans))
        if str(int(qid)) in EXTRA: train.append((EXTRA[str(int(qid))], ans))
    print("questions (incl. paraphrases):", len(train), "| train edges:", len(h), flush=True)
    h, r, t = (torch.from_numpy(x).to(dev) for x in (h, r, t)); gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    ops = [q for n_, q in model.named_parameters() if n_ != "E_real"]
    opt = torch.optim.Adam([{"params": [model.E_real], "lr": a.lr_table}, {"params": ops, "lr": a.lr_ops},
                            {"params": net.enc.parameters(), "lr": a.lr_enc}, {"params": list(net.head.parameters()) + [net.scale], "lr": a.lr_head}])
    steps = a.epochs * ((len(train) + a.qbatch - 1) // a.qbatch); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train(); net.train(); t0 = time.time(); step = 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(train)).tolist(); le = lq = 0.0; nb = 0
        for b in range(0, len(perm), a.qbatch):
            # edge batch
            idx = torch.randint(0, len(h), (a.ebatch,), device=dev, generator=gen); rev = torch.rand(a.ebatch, device=dev, generator=gen) < 0.5
            src = torch.where(rev, t[idx], h[idx]); dst = torch.where(rev, h[idx], t[idx]); rel = r[idx] + rev.long() * n_rel
            negs = torch.randint(0, N, (a.neg,), device=dev, generator=gen)
            logits, z, e_pos = score_batch(model, src, rel, dst, negs)
            loss_e = F.cross_entropy(logits, torch.zeros(a.ebatch, dtype=torch.long, device=dev)) + a.lam * (z - e_pos).abs().pow(2).sum(-1).mean()
            # question batch
            batch = [train[j] for j in perm[b:b + a.qbatch]]
            enc = tok([QPRE + q for q, _ in batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            s = net.score(net(enc), model.table()); ls = torch.log_softmax(s, 1)
            pos = torch.zeros_like(s, dtype=torch.bool)
            for rr, (_, ans) in enumerate(batch): pos[rr, torch.tensor(ans, device=dev)] = True
            loss_q = -(torch.logsumexp(ls.masked_fill(~pos, -1e9), 1)).mean()
            loss = loss_e + a.wq * loss_q
            opt.zero_grad(set_to_none=True); loss.backward()
            clip_grad_norm_(list(model.parameters()), 1.0); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1; le += loss_e.item(); lq += loss_q.item(); nb += 1
            if step % 100 == 0: print(f"ep {ep} step {step}/{steps} edge {loss_e.item():.3f} question {loss_q.item():.3f} {round(time.time()-t0)} s", flush=True)
        print(f"epoch {ep} mean edge loss {le/nb:.3f} question loss {lq/nb:.3f}", flush=True)
    model.eval(); net.eval()
    torch.save({"model": model.state_dict(), "args": ar, "N": N, "n_rel": n_rel}, f"models/{a.tag}.pt")     # save BEFORE any evaluation
    torch.save({"head": net.head.state_dict(), "scale": net.scale.detach().cpu(), "encoder": f"models/{a.tag}_enc", "E": None}, f"models/{a.tag}_head.pt")
    net.enc.save_pretrained(f"models/{a.tag}_enc"); tok.save_pretrained(f"models/{a.tag}_enc")
    mt, mh = mrr_holdout(model, val, N, n_rel, dev); print(f"[link] joint table: held-out MRR tail {mt:.4f} head {mh:.4f} mean {(mt+mh)/2:.4f}", flush=True)
    E = model.table().detach()
    qa_eval(net, tok, E, qa, sp, dev, node_type, ktype, "joint")
    print("JOINT_DONE", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
