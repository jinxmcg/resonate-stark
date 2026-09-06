"""Rank a question set with a trained text-to-latent model (models/<tag>.pt + models/<tag>_enc).
Usage: uv run python t2l_rank.py --tag t2l --split train --queries data/para_train.json --out data/text_train_t2l_para.json"""
import argparse, json, sys, os, numpy as np, torch
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer
from text2latent import T2L, QPRE
from retrieve import load_model
p = argparse.ArgumentParser(); p.add_argument("--tag", required=True); p.add_argument("--split", required=True); p.add_argument("--queries", default=None); p.add_argument("--out", required=True)
p.add_argument("--model", default="models/p_k12b4_50k.pt"); a = p.parse_args(); dev = torch.device("cuda")
assert a.split in ("train", "val")
ck = torch.load(f"models/{a.tag}.pt", map_location="cpu", weights_only=False)
graph, _ = load_model(a.model, dev); E = graph.table().detach().clone()
if ck.get("E") is not None: E = torch.view_as_complex(ck["E"]).to(dev)
tok = AutoTokenizer.from_pretrained(f"models/{a.tag}_enc"); net = T2L(f"models/{a.tag}_enc", E.shape[1], nvec=ck.get("nvec", 1)).to(dev)
net.head.load_state_dict(ck["head"]); net.scale.data = ck["scale"].to(dev); net.eval()
qa = load_qa("prime"); idx = qa.get_idx_split()[a.split].tolist(); QS = json.load(open(a.queries)) if a.queries else {}
out = {}
for b in range(0, len(idx), 64):
    chunk = idx[b:b + 64]; qs = [QS.get(str(int(qa[i][1])), qa[i][0]) for i in chunk]
    with torch.no_grad():
        enc = tok([QPRE + q for q in qs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
        top = torch.topk(net.score(net(enc), E), 100, 1).indices.cpu().numpy()
    for r, i in enumerate(chunk): out[int(qa[i][1])] = top[r].tolist()
json.dump(out, open(a.out, "w")); print("ranked", len(out), "->", a.out)
