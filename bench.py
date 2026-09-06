"""Query-time cost of the submitted pipeline (no generative LLM; two 110M transformer encoders run).
Per-stage latency at batch size 1 over the first N val questions (median and p90, CUDA-synchronised),
batched throughput over all of val, peak GPU memory, and parameter counts. Stages: (1) latent parser
(bge encoder + heads, nearest-entity anchors), (2) graph walk + ResonatE operator scoring,
(3) text ranker (bge_ft2 encode + top-100 over 129k node embeddings), (4) table readout from the
question (p_joint encoder + head + readout), (5) RRF fusion + logistic reranker (numpy).
Usage: PYTHONPATH=lib uv run python bench.py --n 300"""
import argparse, json, time, sys, os, numpy as np, torch
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from retrieve import Parser, Adjacency, load_model, score_query
from latent_parser import LP, QPRE
from text2latent import T2L
from rerank import build

def sync(): torch.cuda.synchronize()
def stats(ts): ts = np.array(ts) * 1000; return f"median {np.median(ts):.1f} ms  p90 {np.percentile(ts, 90):.1f} ms"

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=300); a = p.parse_args(); dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    graph, n_rel = load_model("models/p_k12b4_50k.pt", dev); E0 = graph.table().detach()
    d = np.load("data/kg.npz"); names = json.load(open("data/names.json")); node_type = d["node_type"]; N = len(node_type)
    dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}
    parser = Parser(names, node_type, tn, json.load(open("data/signatures.json"))); adj = Adjacency(d, n_rel, N)
    type_mask = {t: torch.from_numpy(node_type == i).to(dev) for i, t in tn.items()}
    ck = torch.load("models/lp.pt", map_location="cpu", weights_only=False)
    lp_tok = AutoTokenizer.from_pretrained(ck["encoder"]); lp = LP(ck["encoder"], E0.shape[1], len(tn), 2 * n_rel, ck["kanc"]).to(dev); lp.load_state_dict(ck["state"]); lp.eval()
    jg, _ = load_model("models/p_joint.pt", dev); EJ = jg.table().detach()
    hk = torch.load("models/p_joint_head.pt", map_location="cpu", weights_only=False); t2l_tok = AutoTokenizer.from_pretrained("models/p_joint_enc")
    t2l = T2L("models/p_joint_enc", EJ.shape[1], nvec=hk.get("nvec", 1)).to(dev); t2l.head.load_state_dict(hk["head"]); t2l.scale.data = hk["scale"].to(dev); t2l.eval()
    st = SentenceTransformer("models/bge_ft2", device="cuda"); D = torch.from_numpy(np.load("data/doc_emb_bgeft2.npy").astype(np.float32)).to(dev)
    RR = json.load(open("data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json")); mu, sd = np.array(RR["mu"], np.float32), np.array(RR["sd"], np.float32)
    logdeg = np.log1p(np.bincount(d["h"], minlength=N) + np.bincount(d["t"], minlength=N)).astype(np.float32)
    load_s = time.time() - t0; mem_load = torch.cuda.max_memory_allocated() / 2**30
    nparams = {"graph table+ops": sum(x.numel() for x in graph.parameters()), "joint table+ops": sum(x.numel() for x in jg.parameters()),
               "parser encoder+heads": sum(x.numel() for x in lp.parameters()), "question encoder+head": sum(x.numel() for x in t2l.parameters()),
               "text ranker (bge_ft2)": sum(x.numel() for x in st.parameters())}
    qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()[:a.n]; qs = [qa[i][0] for i in idx]; qids = [int(qa[i][1]) for i in idx]
    T = {k: [] for k in ("parse", "walk+model", "text", "readout", "fuse+rerank", "total")}
    floor = ck["sim_floor"]; type_ok = set(tn.values())
    with torch.no_grad():
        for q, qid in zip(qs, qids):
            sync(); t_start = time.time()
            enc = lp_tok([QPRE + q], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            lt, lo, z = lp(enc); s = lp.anc_scores(z, E0); v, ix = s.max(2)
            at = tn[int(lt[0].argmax())]; anchors = sorted({int(ix[0, k]) for k in range(ck["kanc"]) if v[0, k] >= floor}); ops = {int(o) % n_rel for o in (torch.sigmoid(lo[0]) >= 0.5).nonzero().flatten().tolist()}
            sync(); t1 = time.time(); T["parse"].append(t1 - t_start)
            at_, ments = parser.parse(q, at=at, rel_hints=ops)
            have = {i_ for i_, _, _, _ in ments}
            for i_ in anchors:
                if i_ in have: continue
                mt = tn[int(node_type[i_])]; w = parser.chain_weights(mt, at_, q, ops)
                if w: ments.append((i_, names[str(i_)].lower(), mt, w))
            top, fe = score_query(graph, n_rel, (at_, ments), at_, type_mask, dev, adj=adj, beta=30.0, feats=True)
            sync(); t2 = time.time(); T["walk+model"].append(t2 - t1)
            qv = st.encode([QPRE + q], convert_to_tensor=True, normalize_embeddings=True); txt = torch.topk(qv @ D.t(), 100, dim=1).indices[0].tolist()
            sync(); t3 = time.time(); T["text"].append(t3 - t2)
            enc2 = t2l_tok([QPRE + q], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            u = torch.topk(t2l.score(t2l(enc2), EJ), 100, 1).indices[0].tolist()
            sync(); t4 = time.time(); T["readout"].append(t4 - t3)
            rel = {str(qid): {"top": top[:100], "feats": fe, "mentions": [(i_, n_, mt) for (i_, n_, mt, w) in ments], "answer_type": at_}}
            F = build(rel, {str(qid): txt}, RR["w_rrf"], logdeg, [qid], {str(qid): u})
            if F[qid] is not None:
                cands, f = F[qid]; wv = np.array(RR["w_type"].get(str(at_), RR["w_global"]), np.float32); sc = ((f - mu) / sd) @ wv; ranked = [cands[j] for j in np.argsort(-sc)][:100]
            t5 = time.time(); T["fuse+rerank"].append(t5 - t4); T["total"].append(t5 - t_start)
    mem_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"models loaded in {load_s:.1f} s; GPU memory after loading {mem_load:.2f} GiB; peak during batch-1 queries {mem_peak:.2f} GiB")
    for k, v in nparams.items(): print(f"  params {k:24s} {v/1e6:8.1f} M")
    print(f"batch-1 latency over {len(qs)} val questions (RTX 5090, CUDA-synchronised, first query excluded):")
    for k in T: print(f"  {k:12s} {stats(T[k][1:])}")
    # batched throughput: encoders + readout over all of val (the walk is per query and CPU-side)
    allq = [qa[i][0] for i in qa.get_idx_split()["val"].tolist()]
    with torch.no_grad():
        sync(); t = time.time()
        for b in range(0, len(allq), 64):
            enc = t2l_tok([QPRE + q for q in allq[b:b+64]], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev); torch.topk(t2l.score(t2l(enc), EJ), 100, 1)
        sync(); t_r = time.time() - t; t = time.time()
        st.encode([QPRE + q for q in allq], batch_size=128, convert_to_tensor=True, normalize_embeddings=True); sync(); t_e = time.time() - t
    print(f"batched: table readout {len(allq)} questions in {t_r:.2f} s ({t_r/len(allq)*1000:.2f} ms/q); text encoder {t_e:.2f} s ({t_e/len(allq)*1000:.2f} ms/q)")
    print("BENCH_DONE")

if __name__ == "__main__":
    main()
