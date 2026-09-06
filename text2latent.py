"""P2 lever 4: text-to-latent. A question encoder trained INTO the ResonatE latent space.
z_q = cnorm(head(bge(question))) with M complex dims; score(q, t) = Re<z_q, E_t> * exp(s), with the
entity table E frozen from the trained graph model. Loss: softmax cross-entropy over the whole
table with multiple positives (the answer nodes). Trains on `train` (+ paraphrases), evaluates on
plain and paraphrased `val` (all entities, and restricted to the parser's answer type when it found
one). Writes data/text_val_t2l.json / data/text_val_t2l_para.json (top-100) for fusion / reranking.
Usage: PYTHONPATH=lib uv run python text2latent.py --encoder models/bge_ft2 --extra data/para_train.json
"""
import argparse, json, time, sys, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from resonate import cnorm
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModel
from metrics import stark_metrics, summarize
from retrieve import load_model

QPRE = "Represent this sentence for searching relevant passages: "


class T2L(nn.Module):
    def __init__(self, enc_path, m):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_path)
        self.head = nn.Linear(self.enc.config.hidden_size, 2 * m)
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.m = m
    def forward(self, enc):
        h = self.enc(**enc).last_hidden_state[:, 0]                      # bge: CLS pooling
        z = self.head(h).view(-1, self.m, 2)
        return cnorm(torch.view_as_complex(z.contiguous()))
    def score(self, z, E):
        return torch.real(z @ E.conj().t()) * self.scale.exp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/p_k12b4_50k.pt"); p.add_argument("--encoder", default="models/bge_ft2")
    p.add_argument("--extra", default=None, help="paraphrased train json"); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--tag", default="t2l"); p.add_argument("--unfreeze-table", action="store_true"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); dev = torch.device("cuda")
    graph, n_rel = load_model(a.model, dev)
    E = graph.table().detach().clone()                                   # (N, M) complex, frozen
    d = np.load("data/kg.npz"); node_type = torch.from_numpy(d["node_type"]).to(dev)
    tn = {int(k): v for k, v in json.load(open("data/dicts.json"))["node_type_dict"].items()}; ktype = {v: k for k, v in tn.items()}
    tok = AutoTokenizer.from_pretrained(a.encoder); net = T2L(a.encoder, E.shape[1]).to(dev)
    E_param = None
    if a.unfreeze_table:
        E_param = nn.Parameter(torch.view_as_real(E).clone()); E_param.requires_grad_(True)
    qa = load_qa("prime"); sp = qa.get_idx_split()
    EXTRA = json.load(open(a.extra)) if a.extra else {}
    train = []
    for i in sp["train"].tolist():
        q, qid, ans, _ = qa[i]; train.append((q, ans))
        if str(int(qid)) in EXTRA: train.append((EXTRA[str(int(qid))], ans))
    print("train questions (incl. paraphrases):", len(train), flush=True)
    groups = [{"params": net.enc.parameters(), "lr": a.lr}, {"params": list(net.head.parameters()) + [net.scale], "lr": a.lr_head}]
    if E_param is not None: groups.append({"params": [E_param], "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    N = E.shape[0]; t0 = time.time(); step = 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(train)).tolist(); net.train(); tot = 0.0
        for b in range(0, len(perm), a.batch):
            batch = [train[j] for j in perm[b:b + a.batch]]
            enc = tok([QPRE + q for q, _ in batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            z = net(enc)
            Et = torch.view_as_complex(E_param) if E_param is not None else E
            s = net.score(z, Et)                                          # (B, N)
            ls = torch.log_softmax(s, 1)
            pos = torch.zeros_like(s, dtype=torch.bool)
            for r, (_, ans) in enumerate(batch): pos[r, torch.tensor(ans, device=dev)] = True
            loss = -(torch.logsumexp(ls.masked_fill(~pos, -1e9), 1)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            tot += loss.item()
            if step % 100 == 0: print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} {round(time.time()-t0)} s", flush=True)
        print(f"epoch {ep} mean loss {tot / max(1, (len(perm) + a.batch - 1) // a.batch):.3f}", flush=True)
    net.eval(); Et = (torch.view_as_complex(E_param) if E_param is not None else E).detach()
    rel_val = json.load(open("data/rel_val_ancf.json"))                 # parser answer types on val (val is a development split)
    for split_tag, qfile in (("", None), ("_para", "data/para_val.json")):
        QS = json.load(open(qfile)) if qfile else {}
        idx = sp["val"].tolist(); rows_all, rows_typed, out, out_typed = [], [], {}, {}
        for b in range(0, len(idx), 64):
            chunk = idx[b:b + 64]
            qs = [QS.get(str(int(qa[i][1])), qa[i][0]) for i in chunk]
            with torch.no_grad():
                enc = tok([QPRE + q for q in qs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                s = net.score(net(enc), Et)
            top = torch.topk(s, 100, 1).indices.cpu().numpy()
            for r, i in enumerate(chunk):
                q, qid, ans, _ = qa[i]; rows_all.append(stark_metrics(top[r].tolist(), ans)); out[int(qid)] = top[r].tolist()
                at = rel_val.get(str(int(qid)), {}).get("answer_type")
                if at is not None:
                    st = s[r].clone(); st[node_type != ktype[at]] = -1e9; tt = torch.topk(st, 100).indices.cpu().numpy().tolist()
                else:
                    tt = top[r].tolist()
                rows_typed.append(stark_metrics(tt, ans)); out_typed[int(qid)] = tt
        print(f"VAL{split_tag or ' plain'} text-to-latent, all entities   :", {k: round(v, 4) for k, v in summarize(rows_all).items()}, flush=True)
        print(f"VAL{split_tag or ' plain'} text-to-latent, parser type    :", {k: round(v, 4) for k, v in summarize(rows_typed).items()}, flush=True)
        json.dump(out, open(f"data/text_val_{a.tag}{split_tag}.json", "w")); json.dump(out_typed, open(f"data/text_val_{a.tag}typed{split_tag}.json", "w"))
    # train split rankings too (for fusion weight / reranker fitting)
    idx = sp["train"].tolist(); out = {}
    for b in range(0, len(idx), 64):
        chunk = idx[b:b + 64]
        with torch.no_grad():
            enc = tok([QPRE + qa[i][0] for i in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            top = torch.topk(net.score(net(enc), Et), 100, 1).indices.cpu().numpy()
        for r, i in enumerate(chunk): out[int(qa[i][1])] = top[r].tolist()
    json.dump(out, open(f"data/text_train_{a.tag}.json", "w"))
    torch.save({"head": net.head.state_dict(), "scale": net.scale.detach().cpu(), "encoder": a.encoder, "E": (E_param.detach().cpu() if E_param is not None else None)}, f"models/{a.tag}.pt")
    net.enc.save_pretrained(f"models/{a.tag}_enc"); tok.save_pretrained(f"models/{a.tag}_enc")
    print("T2L_DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
