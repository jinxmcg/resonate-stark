#!/bin/bash
# Does ResonatE do work beyond the graph walk? val, LLM-override parse, bge anchors.
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
uv run python embed_text2.py --model bge --splits val > logs/embed_bge_docs.log 2>&1      # rebuild data/doc_emb_bge.npy (not copied)
uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --dump-feats --no-model --out data/rel_val_walkonly.json > logs/rel_val_walkonly.log 2>&1
uv run python retrieve.py --model $M --split val --beta 0  --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --dump-feats --out data/rel_val_modelonly.json > logs/rel_val_modelonly.log 2>&1
python3 - > logs/model_vs_walk.log 2>&1 <<'PY'
import json, sys
sys.path.insert(0, "."); import stark_shim
from stark_qa import load_qa
from metrics import stark_metrics, summarize
qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()
R = {n: json.load(open(f"data/rel_val_{n}.json")) for n in ("walkonly", "modelonly", "llm_ancf")}
names = {"walkonly": "graph walk only (exact traversal, no model)", "modelonly": "ResonatE only (no traversal)", "llm_ancf": "both (P2 retrieval)"}
rows = {n: [] for n in R}; sub = {"reachable": {n: [] for n in R}, "unreachable": {n: [] for n in R}}; nreach = 0
for i in idx:
    q, qid, ans, _ = qa[i]; k = str(int(qid))
    w = R["walkonly"].get(k, {}); ex = w.get("feats", {}).get("exact", []); top = w.get("top", [])
    reach = any(c in set(ans) and e > 0 for c, e in zip(top, ex))         # some answer is graph-supported from the parsed mentions
    nreach += reach
    for n in R:
        m = stark_metrics(R[n].get(k, {}).get("top", []), ans); rows[n].append(m); sub["reachable" if reach else "unreachable"][n].append(m)
print(f"val n={len(idx)}; queries with a graph-supported answer among the walk's top-100: {nreach} ({nreach/len(idx):.1%})")
for n in R: print(f"ALL        {names[n]:45s}", {k: round(v, 4) for k, v in summarize(rows[n]).items()})
for s in ("reachable", "unreachable"):
    for n in R: print(f"{s:10s} {names[n]:45s} n={len(sub[s][n])}", {k: round(v, 4) for k, v in summarize(sub[s][n]).items()})
PY
echo MVW_DONE >> logs/model_vs_walk.log
