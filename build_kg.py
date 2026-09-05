"""Dump the STaRK-Prime SKB into plain arrays for the probe:
data/kg.npz: h, r, t (int64), node_type (129,375,), n_rel; data/names.json: node id -> name;
data/docs.jsonl: node id -> the text STaRK gives each node (for the text retriever / BM25).
Also writes data/splits.json with the STaRK query split sizes (train/val/test ids are read
from stark_qa at run time; test ids are never used by the probe scripts)."""
import json, re, time
import numpy as np, torch
import stark_shim  # noqa: F401
from stark_qa import load_qa, load_skb

t0 = time.time()
skb = load_skb("prime", download_processed=True, root=None)
qa = load_qa("prime")
ei, et = skb.edge_index.numpy(), skb.edge_types.numpy()
nt = skb.node_types.numpy()
np.savez("data/kg.npz", h=ei[0].astype(np.int64), t=ei[1].astype(np.int64), r=et.astype(np.int64),
         node_type=nt.astype(np.int64), n_rel=np.int64(len(skb.edge_type_dict)))
N = skb.num_nodes()
names, docs = {}, []
for i in range(N):
    d = skb.get_doc_info(i, add_rel=False)
    m = re.search(r"^- name: (.*)$", d, re.M)
    names[i] = (m.group(1).strip() if m else "")
    docs.append({"id": i, "type": skb.node_type_dict[int(nt[i])], "text": d})
    if i % 20000 == 0:
        print(i, flush=True)
json.dump(names, open("data/names.json", "w"))
with open("data/docs.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")
sp = qa.get_idx_split()
json.dump({k: len(v) for k, v in sp.items()}, open("data/splits.json", "w"))
json.dump({"edge_type_dict": skb.edge_type_dict, "node_type_dict": skb.node_type_dict}, open("data/dicts.json", "w"))
print("edges", len(et), "nodes", N, "rels", len(skb.edge_type_dict), "splits", {k: len(v) for k, v in sp.items()}, "in", round(time.time()-t0), "s")
