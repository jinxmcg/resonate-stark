#!/bin/bash
# P14 (box 50209059): does a biomedical embedder retain more of its accuracy on clinical shorthand?
# Stock against stock, no fine-tuning. Val only. See PLAN_PRIME.md "P14".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p14_status.log; }
rm -f logs/p14_status.log; st "stock embedders on four wordings"
$PY - <<'PY' >> logs/p14_status.log 2>&1
import json, sys, time
sys.path[:0] = ["lib"]
import numpy as np, torch
import stark_shim
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer
from metrics import stark_metrics, summarize
MODELS = {"bge (incumbent base)": "BAAI/bge-base-en-v1.5",
          "pubmedbert": "NeuML/pubmedbert-base-embeddings",
          "medembed": "abhinand/MedEmbed-base-v0.1"}
QPRE = "Represent this sentence for searching relevant passages: "
WORD = {"plain": None, "terseA": "data/para_val_terse.json", "terseB": "data/para_val_terse_b.json", "terseC": "data/para_val_terse_c.json"}
docs = [json.loads(l)["text"][:3000] for l in open("data/docs.jsonl")]
qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()
for nm, path in MODELS.items():
    try:
        m = SentenceTransformer(path, device="cuda"); m.max_seq_length = 512
    except Exception as e:
        print(f"    {nm}: LOAD FAILED — {str(e)[:110]}"); continue
    pre = QPRE if "bge" in path.lower() else ""      # bge's own query convention; the others use none
    t0 = time.time()
    D = torch.from_numpy(m.encode(docs, batch_size=128, convert_to_numpy=True, normalize_embeddings=True)).cuda()
    res = {}
    for w, qf in WORD.items():
        QS = json.load(open(qf)) if qf else {}
        qs = [QS.get(str(int(qa[i][1])), qa[i][0]) for i in idx]
        Q = torch.from_numpy(m.encode([pre + q for q in qs], batch_size=128, convert_to_numpy=True, normalize_embeddings=True)).cuda()
        top = torch.topk(Q @ D.t(), 100, 1).indices.cpu().numpy()
        res[w] = summarize([stark_metrics(top[j].tolist(), qa[i][2]) for j, i in enumerate(idx)])["hit1"] * 100
    ret = np.mean([res[w] / res["plain"] for w in ("terseA", "terseB", "terseC")]) * 100
    print(f"    {nm:22s} plain {res['plain']:5.2f}  A {res['terseA']:5.2f}  B {res['terseB']:5.2f}  C {res['terseC']:5.2f}   shorthand retention {ret:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    del m, D; torch.cuda.empty_cache()
PY
st "P14_DONE"
