"""Ask STaRK-Prime a question in words, from the command line, with the submitted no-LLM pipeline.
Prints what the parser read, then the ranked entities with a graph-support flag ([ok] = reached by
the exact adjacency walk from the parsed anchors; [--] = not graph-supported from what was read:
proposed by the text ranker or the question readout) and the per-stage timings.
Usage: PYTHONPATH=lib uv run python ask.py "Which drugs target GCK and are indicated for diabetes?" [-k 8] [--repl]"""
import argparse, json, time, sys, os, numpy as np, torch
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
import stark_shim  # noqa
from transformers import AutoTokenizer
from transformers import AutoModel
from retrieve import Parser, Adjacency, load_model, score_query
from latent_parser import LP, QPRE
from text2latent import T2L
from rerank import build

class Ask:
    def __init__(self, dev="cuda"):
        t0 = time.time(); self.dev = torch.device(dev)
        self.graph, self.n_rel = load_model("models/p_k12b4_50k.pt", self.dev); self.E0 = self.graph.table().detach()
        d = np.load("data/kg.npz"); self.names = json.load(open("data/names.json")); self.node_type = d["node_type"]; N = len(self.node_type)
        dicts = json.load(open("data/dicts.json")); self.tn = {int(k): v for k, v in dicts["node_type_dict"].items()}; self.rel_names = {int(k): v for k, v in dicts["edge_type_dict"].items()}
        self.parser = Parser(self.names, self.node_type, self.tn, json.load(open("data/signatures.json"))); self.adj = Adjacency(d, self.n_rel, N)
        self.type_mask = {t: torch.from_numpy(self.node_type == i).to(self.dev) for i, t in self.tn.items()}
        ck = torch.load("models/lp.pt", map_location="cpu", weights_only=False)
        self.lp_tok = AutoTokenizer.from_pretrained(ck["encoder"]); self.lp = LP(ck["encoder"], self.E0.shape[1], len(self.tn), 2 * self.n_rel, ck["kanc"]).to(self.dev); self.lp.load_state_dict(ck["state"]); self.lp.eval(); self.kanc = ck["kanc"]; self.floor = ck["sim_floor"]
        jg, _ = load_model("models/p_joint.pt", self.dev); self.EJ = jg.table().detach()
        hk = torch.load("models/p_joint_head.pt", map_location="cpu", weights_only=False); self.t2l_tok = AutoTokenizer.from_pretrained("models/p_joint_enc")
        self.t2l = T2L("models/p_joint_enc", self.EJ.shape[1], nvec=hk.get("nvec", 1)).to(self.dev); self.t2l.head.load_state_dict(hk["head"]); self.t2l.scale.data = hk["scale"].to(self.dev); self.t2l.eval()
        self.tx_tok = AutoTokenizer.from_pretrained("models/bge_ft2"); self.tx = AutoModel.from_pretrained("models/bge_ft2").to(self.dev).eval()   # bge: CLS pooling + L2 norm (no sentence-transformers version dependency)
        self.D = torch.from_numpy(np.load("data/doc_emb_bgeft2.npy").astype(np.float32)).to(self.dev)
        self.RR = json.load(open("data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json")); self.mu, self.sd = np.array(self.RR["mu"], np.float32), np.array(self.RR["sd"], np.float32)
        self.logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
        self.load_s = time.time() - t0

    def sync(self):
        if self.dev.type == "cuda": torch.cuda.synchronize()

    @torch.no_grad()
    def ask(self, q, k=8):
        T = {}; self.sync(); t = time.time()
        enc = self.lp_tok([QPRE + q], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.dev)
        lt, lo, z = self.lp(enc); s = self.lp.anc_scores(z, self.E0); v, ix = s.max(2)
        at = self.tn[int(lt[0].argmax())]; anchors = sorted({int(ix[0, j]) for j in range(self.kanc) if v[0, j] >= self.floor})
        ops = {int(o) % self.n_rel for o in (torch.sigmoid(lo[0]) >= 0.5).nonzero().flatten().tolist()}
        self.sync(); T["parse"] = time.time() - t; t = time.time()
        at_, ments = self.parser.parse(q, at=at, rel_hints=ops); have = {i_ for i_, _, _, _ in ments}
        for i_ in anchors:
            if i_ in have: continue
            mt = self.tn[int(self.node_type[i_])]; w = self.parser.chain_weights(mt, at_, q, ops)
            if w: ments.append((i_, self.names[str(i_)].lower(), mt, w))
        top, fe = score_query(self.graph, self.n_rel, (at_, ments), at_, self.type_mask, self.dev, adj=self.adj, beta=30.0, feats=True)
        self.sync(); T["walk+model"] = time.time() - t; t = time.time()
        e3 = self.tx_tok([QPRE + q], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.dev)
        qv = torch.nn.functional.normalize(self.tx(**e3).last_hidden_state[:, 0], dim=-1); txt = torch.topk(qv @ self.D.t(), 100, dim=1).indices[0].tolist()
        self.sync(); T["text"] = time.time() - t; t = time.time()
        enc2 = self.t2l_tok([QPRE + q], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.dev)
        u = torch.topk(self.t2l.score(self.t2l(enc2), self.EJ), 100, 1).indices[0].tolist()
        self.sync(); T["readout"] = time.time() - t; t = time.time()
        qid = "0"; rel = {qid: {"top": top[:100], "feats": fe, "mentions": [(i_, n_, mt) for (i_, n_, mt, w) in ments], "answer_type": at_}} if top else {}
        supported = {c for c, e in zip(top, (fe or {}).get("exact", [])) if e > 0} if top else set()
        F = build(rel, {qid: txt}, self.RR["w_rrf"], self.logdeg, [0], {qid: u}) if top else {0: None}
        if F[0] is not None:
            cands, f = F[0]; wv = np.array(self.RR["w_type"].get(str(at_), self.RR["w_global"]), np.float32); sc = ((f - self.mu) / self.sd) @ wv
            ranked = [cands[j] for j in np.argsort(-sc)][:k]
        else:
            from rerank import fused; ranked = fused([], txt, self.RR["w_rrf"])[0][:k]
        T["fuse+rerank"] = time.time() - t; T["total"] = sum(T.values())
        return {"answer_type": at_, "anchors": [(i_, self.names[str(i_)], mt) for (i_, _, mt, _) in ments], "relations": sorted(self.rel_names[o] for o in ops),
                "ranked": [(c, self.names[str(c)], self.tn[int(self.node_type[c])], c in supported) for c in ranked], "timings": T}

    def show(self, q, k=8):
        r = self.ask(q, k)
        print(f"\n$ ask \"{q}\"")
        print(f"  read: answer type = {r['answer_type']} · anchors = " + (", ".join(f"{n} ({mt})" for _, n, mt in r["anchors"]) or "none") + " · relation hints = " + (", ".join(r["relations"]) or "none"))
        for i, (c, n, ty, ok) in enumerate(r["ranked"], 1):
            print(f"  {i:>2}  {'[ok]' if ok else '[--]'}  {n}  ({ty})")
        n_ok = sum(1 for *_, ok in r["ranked"] if ok); T = r["timings"]
        print(f"  {n_ok}/{len(r['ranked'])} graph-supported from what was read · parse {T['parse']*1000:.1f} ms · walk+model {T['walk+model']*1000:.1f} ms · text {T['text']*1000:.1f} ms · readout {T['readout']*1000:.1f} ms · rerank {T['fuse+rerank']*1000:.1f} ms · total {T['total']*1000:.1f} ms")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("question", nargs="?"); p.add_argument("-k", type=int, default=8); p.add_argument("--repl", action="store_true"); p.add_argument("--device", default="cuda")
    a = p.parse_args(); A = Ask(a.device); print(f"[stark-prime] 129,375 entities · models loaded in {A.load_s:.1f} s · no generative model in the loop")
    if a.question: A.show(a.question, a.k)
    if a.repl or not a.question:
        while True:
            try: q = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt): break
            if q: A.show(q, a.k)
