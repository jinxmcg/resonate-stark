"""P2 lever 7: latent parser — read a question into the model's structured query space (answer type,
anchor entities, relation operators) with a learned head, then let ResonatE + the graph walk answer.
Train on `train` (+ paraphrases) with weak labels derived from the graph; write predictions for a
question set in the format retrieve.py --lparse consumes: {qid: {"answer_type", "anchors": [ids],
"ops": [operator ids r + n_rel*d]}}. See PLAN_PRIME.md lever 7.
Usage: PYTHONPATH=lib uv run python latent_parser.py --train --tag lp  (then) --predict val --queries data/para_val.json --out data/lparse_val_para.json"""
import argparse, json, time, sys, os, re, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
from resonate import cnorm
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModel
from retrieve import Parser, Adjacency, load_model, K as KW
QPRE = "Represent this sentence for searching relevant passages: "


class LP(nn.Module):
    def __init__(self, enc_path, m, n_types, n_ops, kanc=3):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_path); H = self.enc.config.hidden_size
        self.type_head = nn.Linear(H, n_types); self.op_head = nn.Linear(H, n_ops)
        self.anc_head = nn.Linear(H, kanc * 2 * m); self.scale = nn.Parameter(torch.tensor(3.0)); self.m = m; self.kanc = kanc
    def forward(self, enc):
        h = self.enc(**enc).last_hidden_state[:, 0]
        z = cnorm(torch.view_as_complex(self.anc_head(h).view(-1, self.kanc, self.m, 2).contiguous()))
        return self.type_head(h), self.op_head(h), z
    def anc_scores(self, z, E):
        return torch.real(z @ E.conj().t()) * self.scale.exp()            # (B, K, N)


def weak_labels(qa, idx, QS, parser, adj, n_rel, LP_ents, tn, ktype):
    """per question: (text, answer type id, anchor ids, operator label vector)"""
    out = []
    for i in idx:
        q, qid, ans, _ = qa[i]; q = QS.get(str(int(qid)), q); k = str(int(qid))
        at_id = collections.Counter(int(parser.node_type[a_]) for a_ in ans).most_common(1)[0][0]; at = tn[at_id]
        anchors = set()
        for n, ids in parser.mentions(q): anchors.update(ids[:3])
        for e in (LP_ents.get(k) or {}).get("entities") or []:
            n = (e.get("name") or "").strip().lower() if isinstance(e, dict) else ""
            if n in parser.by_name: anchors.update(parser.by_name[n][:3])
        ops = np.zeros(2 * n_rel, np.float32); A = set(ans); good = set()
        for a_ in anchors:
            mt = tn[int(parser.node_type[a_])]
            for (r, d) in parser.ops.get((mt, at), ()):
                if len(set(adj.reach(a_, ((r, d),)).tolist()) & A): ops[r + n_rel * d] = 1; good.add(a_)
            if not parser.ops.get((mt, at)):
                for (op1, op2) in parser.ops2.get((mt, at), ()):
                    if len(set(adj.reach(a_, (op1, op2)).tolist()) & A): ops[op1[0] + n_rel * op1[1]] = 1; ops[op2[0] + n_rel * op2[1]] = 1; good.add(a_)
        out.append((q, at_id, sorted(good) if good else sorted(anchors), ops))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/p_joint.pt"); p.add_argument("--encoder", default="models/p_joint_enc"); p.add_argument("--tag", default="lp")
    p.add_argument("--train", action="store_true"); p.add_argument("--extra", default="data/para_train.json"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--predict", default=None, help="split to parse (train/val)"); p.add_argument("--queries", default=None); p.add_argument("--out", default=None)
    p.add_argument("--kanc", type=int, default=3); p.add_argument("--sim-floor", type=float, default=None, help="anchor similarity floor (default: tuned on train)"); p.add_argument("--op-thr", type=float, default=0.5)
    a = p.parse_args(); dev = torch.device("cuda")
    graph, n_rel = load_model(a.model, dev); E = graph.table().detach()
    d = np.load("data/kg.npz"); names = json.load(open("data/names.json")); node_type = d["node_type"]
    dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}; ktype = {v: k for k, v in tn.items()}
    sigs = json.load(open("data/signatures.json")); parser = Parser(names, node_type, tn, sigs)
    qa = load_qa("prime"); sp = qa.get_idx_split()
    tok = AutoTokenizer.from_pretrained(a.encoder)
    if a.train:
        adj = Adjacency(d, n_rel, len(node_type))
        LPt = json.load(open("data/llmparse_train.json")); LPp = json.load(open("data/llmparse_train_para.json"))
        EXTRA = json.load(open(a.extra)) if a.extra else {}
        t0 = time.time(); tr = sp["train"].tolist()
        data = weak_labels(qa, tr, {}, parser, adj, n_rel, LPt, tn, ktype) + (weak_labels(qa, tr, EXTRA, parser, adj, n_rel, LPp, tn, ktype) if EXTRA else [])
        n_anc = sum(1 for x in data if x[2]); n_ops = sum(1 for x in data if x[3].sum() > 0)
        print(f"weak labels: {len(data)} questions, {n_anc} with anchors, {n_ops} with a hitting operator ({round(time.time()-t0)} s)", flush=True)
        net = LP(a.encoder, E.shape[1], len(tn), 2 * n_rel, a.kanc).to(dev)
        opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 2e-5}, {"params": [q for n_, q in net.named_parameters() if not n_.startswith("enc.")], "lr": 1e-3}], weight_decay=0.01)
        steps = a.epochs * ((len(data) + a.batch - 1) // a.batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps, pct_start=0.1)
        net.train(); step = 0
        for ep in range(a.epochs):
            perm = torch.randperm(len(data)).tolist(); tot = np.zeros(3)
            for b in range(0, len(perm), a.batch):
                batch = [data[j] for j in perm[b:b + a.batch]]
                enc = tok([QPRE + x[0] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                lt, lo, z = net(enc)
                l_type = F.cross_entropy(lt, torch.tensor([x[1] for x in batch], device=dev))
                l_ops = F.binary_cross_entropy_with_logits(lo, torch.tensor(np.stack([x[3] for x in batch]), device=dev))
                s = net.anc_scores(z, E).max(1).values                     # (B, N): best of K vectors per entity
                ls = torch.log_softmax(s, 1); pos = torch.zeros_like(s, dtype=torch.bool); has = []
                for rr, x in enumerate(batch):
                    if x[2]: pos[rr, torch.tensor(x[2], device=dev)] = True; has.append(rr)
                l_anc = -(torch.logsumexp(ls[has].masked_fill(~pos[has], -1e9), 1)).mean() if has else torch.tensor(0.0, device=dev)
                loss = l_type + l_ops + l_anc
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
                tot += [l_type.item(), l_ops.item(), l_anc.item()]
                if step % 100 == 0: print(f"ep {ep} step {step}/{steps} type {l_type.item():.3f} ops {l_ops.item():.3f} anchors {l_anc.item():.3f} ({round(time.time()-t0)} s)", flush=True)
            nb = (len(perm) + a.batch - 1) // a.batch; print(f"epoch {ep} mean type {tot[0]/nb:.3f} ops {tot[1]/nb:.3f} anchors {tot[2]/nb:.3f}", flush=True)
        net.eval()
        # tune the anchor similarity floor on train: keep anchors whose nearest-entity score exceeds the floor; choose the floor
        # that maximises anchor F1 against the weak labels on 1000 plain train questions
        sub = data[:1000]; sims, hits = [], []
        with torch.no_grad():
            for b in range(0, len(sub), 64):
                batch = sub[b:b + 64]
                enc = tok([QPRE + x[0] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                _, _, z = net(enc); s = net.anc_scores(z, E); v, ix = s.max(2)               # (B, K)
                for rr, x in enumerate(batch):
                    for kk in range(a.kanc): sims.append(v[rr, kk].item()); hits.append(int(ix[rr, kk].item()) in set(x[2]))
        sims, hits = np.array(sims), np.array(hits); best = (0, None)
        for f in np.quantile(sims, np.linspace(0.0, 0.95, 40)):
            keep = sims >= f; tp = (keep & hits).sum(); prec = tp / max(1, keep.sum()); rec = tp / max(1, hits.sum()); f1 = 2 * prec * rec / max(1e-9, prec + rec)
            if f1 > best[0]: best = (f1, float(f))
        print(f"anchor floor tuned on train: {best[1]:.3f} (F1 {best[0]:.3f}; nearest-entity precision at no floor {hits.mean():.3f})", flush=True)
        torch.save({"state": net.state_dict(), "kanc": a.kanc, "sim_floor": best[1], "encoder": a.encoder}, f"models/{a.tag}.pt"); print("LP_TRAIN_DONE", round(time.time() - t0), "s")
        return
    ck = torch.load(f"models/{a.tag}.pt", map_location="cpu", weights_only=False)
    net = LP(a.encoder, E.shape[1], len(tn), 2 * n_rel, ck["kanc"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    floor = a.sim_floor if a.sim_floor is not None else ck["sim_floor"]
    idx = sp[a.predict].tolist(); QS = json.load(open(a.queries)) if a.queries else {}; out = {}
    with torch.no_grad():
        for b in range(0, len(idx), 64):
            chunk = idx[b:b + 64]; qs = [QS.get(str(int(qa[i][1])), qa[i][0]) for i in chunk]
            enc = tok([QPRE + q for q in qs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            lt, lo, z = net(enc); s = net.anc_scores(z, E); v, ix = s.max(2); pt = torch.sigmoid(lo)
            for rr, i in enumerate(chunk):
                anchors = sorted({int(ix[rr, kk]) for kk in range(ck["kanc"]) if v[rr, kk] >= floor})
                out[int(qa[i][1])] = {"answer_type": tn[int(lt[rr].argmax())], "anchors": anchors, "ops": [int(o) for o in (pt[rr] >= a.op_thr).nonzero().flatten().tolist()]}
    json.dump(out, open(a.out, "w")); print("LP_PREDICT_DONE", len(out), "->", a.out)


if __name__ == "__main__":
    main()
