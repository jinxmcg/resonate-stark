#!/bin/bash
# lever 2c(b): LLM parser on val and paraphrased val, retrieval with it, P2 reranker on top. Waits for the fine-tune to free VRAM.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
until grep -q FT_DONE logs/ft_bge2.log 2>/dev/null; do sleep 20; done
uv run python llm_parse.py --split val --limit 64 --tag smoke > logs/llm_smoke.log 2>&1
python3 -c "
import json; d=json.load(open('data/llmparse_smoke.json')); import itertools
for k,v in itertools.islice(d.items(),0,8): print(k, json.dumps(v))" >> logs/llm_smoke.log
uv run python retrieve.py --model models/p_k12b4_50k.pt --split val --limit 64 --beta 30 --anchor bge --llm-parse data/llmparse_smoke.json --out data/rel_smoke.json >> logs/llm_smoke.log 2>&1
echo SMOKE_DONE >> logs/llm_smoke.log
uv run python llm_parse.py --split val --tag val > logs/llm_val.log 2>&1
uv run python llm_parse.py --split val --queries data/para_val.json --tag val_para > logs/llm_val_para.log 2>&1
for M in fallback override; do
  X=""; [ $M = override ] && X="--llm-override-type"
  uv run python retrieve.py --model models/p_k12b4_50k.pt --split val --beta 30 --anchor bge --dump-feats --llm-parse data/llmparse_val.json $X --out data/rel_val_llm${M}_ancf.json > logs/rel_val_llm${M}.log 2>&1
  uv run python retrieve.py --model models/p_k12b4_50k.pt --split val --beta 30 --anchor bge --dump-feats --queries data/para_val.json --llm-parse data/llmparse_val_para.json $X --out data/rel_val_para_llm${M}_ancf.json > logs/rel_val_para_llm${M}.log 2>&1
  uv run python predict.py --split val --rel data/rel_val_llm${M}_ancf.json --text data/text_val_bgeft.json --w 0.5 --rerank data/rerank_ancf_bgeft.json --score > logs/llm_${M}_plain.log 2>&1
  uv run python predict.py --split val --rel data/rel_val_para_llm${M}_ancf.json --text data/text_val_bgeft_para.json --w 0.5 --rerank data/rerank_ancf_bgeft.json --score > logs/llm_${M}_para.log 2>&1
done
rm -f eval_results_val.csv
echo LLM_DONE >> logs/llm_override_para.log
