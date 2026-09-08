"""P9: one trunk, three heads — the latent parser, the text ranker and the text-to-latent readout
served by a single 110M encoder instead of three (see PLAN_PRIME.md "P9").

Trunk: stock BAAI/bge-base-en-v1.5, CLS pooling. Heads on the CLS vector: answer type, operators,
K=3 complex anchor vectors, one complex text-to-latent vector; the text ranker uses the L2-normalised
CLS itself, the convention the document matrix already follows. The entity table is frozen. All five
losses are taken on the same batch with equal weight.

Usage:
  train  : PYTHONPATH=lib uv run python shared_trunk.py --train --tag st_f0 --fold 5:0
  parse  : ... --predict-parse train --tag st_f0 --fold 5:0 --out data/lparse_train_st_f0.json
  text   : ... --predict-text  train --tag st_f0 --fold 5:0 --out data/text_train_st_f0.json
  t2l    : ... --predict-t2l   train --tag st_f0 --fold 5:0 --out data/t2l_train_st_f0.json
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from resonate import cnorm
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModel
from retrieve import load_model
from metrics import stark_metrics, summarize

QPRE = "Represent this sentence for searching relevant passages: "
BASE = "BAAI/bge-base-en-v1.5"


class Shared(nn.Module):
    def __init__(self, enc_path, m, n_types, n_ops, kanc=3):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_path); H = self.enc.config.hidden_size
        self.type_head = nn.Linear(H, n_types); self.op_head = nn.Linear(H, n_ops)
        self.anc_head = nn.Linear(H, kanc * 2 * m); self.t2l_head = nn.Linear(H, 2 * m)
        self.scale = nn.Parameter(torch.tensor(3.0)); self.t2l_scale = nn.Parameter(torch.tensor(3.0))
        self.m = m; self.kanc = kanc

    def h(self, enc):
        return self.enc(**enc).last_hidden_state[:, 0]                       # bge: CLS pooling

    def text(self, h):
        return F.normalize(h, dim=-1)                                        # the text ranker's vector

    def anchors(self, h):
        return cnorm(torch.view_as_complex(self.anc_head(h).view(-1, self.kanc, self.m, 2).contiguous()))

    def t2l(self, h):
        return cnorm(torch.view_as_complex(self.t2l_head(h).view(-1, 1, self.m, 2).contiguous()))

    def anc_scores(self, z, E):
        return torch.real(z @ E.conj().t()) * self.scale.exp()               # (B, K, N)

    def t2l_scores(self, z, E):
        return (torch.real(z @ E.conj().t()) * self.t2l_scale.exp()).max(1).values   # (B, N)


def rows(a, tn):
    """the weak-label rows the latent parser was trained on, plus each row's answers and answer doc"""
    L = json.load(open(a.labels)); data = L["data"]; pos = L["pos"]
    qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
    ANS = [qa[i][2] for i in tr]
    docs = {json.loads(l)["id"]: json.loads(l)["text"][:1500] for l in open("data/docs.jsonl")}
    out = []
    for (txt, at, anc, ops), p in zip(data, pos):
        out.append((txt, int(at), list(anc), np.array(ops, np.float32), ANS[p], p))
    return out, docs, qa, tr


def fold_filter(rowlist, fold, keep):
    if not fold: return rowlist
    K, k = (int(x) for x in fold.split(":"))
    return [r for r in rowlist if ((r[-1] % K == k) == keep)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="st"); p.add_argument("--fold", default=None, help="K:k — train without fold k; predict only fold k")
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict-parse", default=None); p.add_argument("--predict-text", default=None); p.add_argument("--predict-t2l", default=None)
    p.add_argument("--out", default=None); p.add_argument("--queries", default=None)
    p.add_argument("--model", default="models/p_joint.pt"); p.add_argument("--labels", default="data/lp_labels.json")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--kanc", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-len", type=int, default=128); p.add_argument("--doc-len", type=int, default=256)
    p.add_argument("--w-text", type=float, default=1.0, help="P15: weight of the text-ranking loss (P9 used 1.0 and the text head was the one that lost)")
    a = p.parse_args(); torch.manual_seed(a.seed); dev = torch.device("cuda")
    graph, n_rel = load_model(a.model, dev); E = graph.table().detach()
    dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}
    ck_path = f"models/{a.tag}.pt"

    if a.train:
        data, docs, qa, tr = rows(a, tn)
        data = fold_filter(data, a.fold, keep=False)
        print(f"rows {len(data)} (fold {a.fold} held out)", flush=True)
        tok = AutoTokenizer.from_pretrained(BASE)
        net = Shared(BASE, E.shape[1], len(tn), 2 * n_rel, a.kanc).to(dev)
        opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": a.lr},
                                 {"params": [q for n_, q in net.named_parameters() if not n_.startswith("enc.")], "lr": a.lr_head}], weight_decay=0.01)
        steps = a.epochs * ((len(data) + a.batch - 1) // a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_head], total_steps=steps, pct_start=0.1)
        net.train(); step = 0; t0 = time.time()
        for ep in range(a.epochs):
            perm = torch.randperm(len(data)).tolist(); tot = np.zeros(5); nb = 0
            for b in range(0, len(perm), a.batch):
                batch = [data[j] for j in perm[b:b + a.batch]]
                enc = tok([QPRE + x[0] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to(dev)
                h = net.h(enc)
                l_type = F.cross_entropy(net.type_head(h), torch.tensor([x[1] for x in batch], device=dev))
                l_ops = F.binary_cross_entropy_with_logits(net.op_head(h), torch.tensor(np.stack([x[3] for x in batch]), device=dev))
                s = net.anc_scores(net.anchors(h), E).max(1).values                  # (B, N)
                ls = torch.log_softmax(s, 1); pos = torch.zeros_like(s, dtype=torch.bool); has = []
                for r, x in enumerate(batch):
                    if x[2]: pos[r, torch.tensor(x[2], device=dev)] = True; has.append(r)
                l_anc = -(torch.logsumexp(ls[has].masked_fill(~pos[has], -1e9), 1)).mean() if has else torch.zeros((), device=dev)
                st_ = net.t2l_scores(net.t2l(h), E)
                lst = torch.log_softmax(st_, 1); pa = torch.zeros_like(st_, dtype=torch.bool); hasa = []
                for r, x in enumerate(batch):
                    if len(x[4]): pa[r, torch.tensor(list(x[4]), device=dev)] = True; hasa.append(r)
                l_t2l = -(torch.logsumexp(lst[hasa].masked_fill(~pa[hasa], -1e9), 1)).mean() if hasa else torch.zeros((), device=dev)
                denc = tok([docs[int(x[4][0])] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=a.doc_len).to(dev)
                dq = net.text(h); dd = net.text(net.h(denc))                          # in-batch negatives, bge convention
                sim = dq @ dd.t() * 20.0
                l_txt = F.cross_entropy(sim, torch.arange(len(batch), device=dev))
                loss = l_type + l_ops + l_anc + l_t2l + a.w_text * l_txt
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
                tot += [l_type.item(), l_ops.item(), l_anc.item(), l_t2l.item(), l_txt.item()]; nb += 1
                if step % 100 == 0:
                    print(f"ep {ep} step {step}/{steps} type {l_type.item():.3f} ops {l_ops.item():.3f} anc {l_anc.item():.3f} t2l {l_t2l.item():.3f} txt {l_txt.item():.3f} ({round(time.time()-t0)} s)", flush=True)
            print(f"epoch {ep} means type {tot[0]/nb:.3f} ops {tot[1]/nb:.3f} anc {tot[2]/nb:.3f} t2l {tot[3]/nb:.3f} txt {tot[4]/nb:.3f}", flush=True)
        net.eval()
        # anchor similarity floor, tuned exactly as latent_parser.py tunes it: best F1 against the
        # weak labels on the first 1,000 rows the model was trained on
        sub = data[:1000]; sims, hits = [], []
        with torch.no_grad():
            for b in range(0, len(sub), 64):
                batch = sub[b:b + 64]
                enc = tok([QPRE + x[0] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to(dev)
                s = net.anc_scores(net.anchors(net.h(enc)), E); v, ix = s.max(2)
                for r, x in enumerate(batch):
                    for k in range(a.kanc): sims.append(v[r, k].item()); hits.append(int(ix[r, k].item()) in set(x[2]))
        sims = np.array(sims); hits = np.array(hits); best = (None, -1)
        for thr in np.quantile(sims, np.linspace(0.05, 0.95, 19)):
            keep = sims >= thr
            if keep.sum() == 0: continue
            prec = hits[keep].mean(); rec = hits[keep].sum() / max(1, hits.sum()); f1 = 2 * prec * rec / max(1e-9, prec + rec)
            if f1 > best[1]: best = (float(thr), float(f1))
        floor = best[0]; print(f"anchor floor {floor:.3f} (F1 {best[1]:.3f})", flush=True)
        torch.save({"state": net.state_dict(), "kanc": a.kanc, "sim_floor": floor, "encoder": BASE, "fold": a.fold}, ck_path)
        print(f"saved {ck_path}  ({sum(v.numel() for v in net.state_dict().values())/1e6:.1f}M parameters)", flush=True)
        print("TRAIN_DONE"); return

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(ck["encoder"])
    net = Shared(ck["encoder"], E.shape[1], len(tn), 2 * n_rel, ck["kanc"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    split = a.predict_parse or a.predict_text or a.predict_t2l
    qa = load_qa("prime"); idx = qa.get_idx_split()[split].tolist()
    if a.fold:
        K, k = (int(x) for x in a.fold.split(":")); idx = [i for p_, i in enumerate(idx) if p_ % K == k]
    QS = json.load(open(a.queries)) if a.queries else {}
    texts = [QS.get(str(int(qa[i][1])), qa[i][0]) for i in idx]
    out, rows_m = {}, []
    with torch.no_grad():
        if a.predict_text:
            docs = [json.loads(l) for l in open("data/docs.jsonl")]
            D = []
            for b in range(0, len(docs), 128):
                denc = tok([d["text"][:1500] for d in docs[b:b + 128]], return_tensors="pt", padding=True, truncation=True, max_length=a.doc_len).to(dev)
                D.append(net.text(net.h(denc)))
                if b % 12800 == 0: print(f"docs {b}/{len(docs)}", flush=True)
            D = torch.cat(D)
            np.save(f"data/doc_emb_{a.tag}.npy", D.cpu().numpy().astype(np.float16))
        for b in range(0, len(idx), 64):
            chunk = idx[b:b + 64]
            enc = tok([QPRE + t for t in texts[b:b + 64]], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to(dev)
            h = net.h(enc)
            if a.predict_text:
                top = torch.topk(net.text(h) @ D.t(), 100, 1).indices.cpu().numpy()
                for r, i in enumerate(chunk): out[int(qa[i][1])] = top[r].tolist(); rows_m.append(stark_metrics(top[r].tolist(), qa[i][2]))
            elif a.predict_t2l:
                top = torch.topk(net.t2l_scores(net.t2l(h), E), 100, 1).indices.cpu().numpy()
                for r, i in enumerate(chunk): out[int(qa[i][1])] = top[r].tolist(); rows_m.append(stark_metrics(top[r].tolist(), qa[i][2]))
            else:
                lt = net.type_head(h); lo = net.op_head(h); s = net.anc_scores(net.anchors(h), E)
                v, ix = s.max(2)
                for r, i in enumerate(chunk):
                    anc = [int(ix[r, kk]) for kk in range(ck["kanc"]) if v[r, kk].item() >= ck["sim_floor"]]
                    out[int(qa[i][1])] = {"answer_type": tn[int(lt[r].argmax())],
                                          "anchors": sorted(set(anc)),
                                          "ops": [int(o) for o in (torch.sigmoid(lo[r]) >= 0.5).nonzero().flatten().tolist()]}
    json.dump(out, open(a.out, "w"))
    if rows_m: print(f"{a.tag} {split} (fold {a.fold}) n={len(rows_m)}:", {k: round(v, 4) for k, v in summarize(rows_m).items()}, flush=True)
    print(f"wrote {a.out} ({len(out)} questions)"); print("PREDICT_DONE")


if __name__ == "__main__":
    main()
