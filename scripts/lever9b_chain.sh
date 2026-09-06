#!/bin/bash
# lever 9b: soft-count AND (the walk's exact-support count, in the space); waits for lever9_chain
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
until grep -q LEVER9_DONE logs/lever9_status.log 2>/dev/null; do sleep 30; done
st() { echo "$(date +%H:%M:%S) $1" >> logs/lever9_status.log; }
run() { T=$1; shift
  uv run python retrieve.py --model $M --split val --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --out data/rel_tmp_$T.json "$@" > logs/l9_$T.log 2>&1
  uv run python retrieve.py --model $M --split val --anchor bge --queries data/para_val.json --llm-parse data/llmparse_val_para.json --llm-override-type --out data/rel_tmp_${T}_para.json "$@" > logs/l9_${T}_para.log 2>&1
  echo "== $T" >> logs/lever9_status.log
  for f in logs/l9_$T.log logs/l9_${T}_para.log; do grep "all queries\|>= 2 names" $f | sed "s/^/    /" >> logs/lever9_status.log; done
}
st "soft-count AND, model only"
run b0_count_c1 --beta 0 --agg count --agg-p 1
run b0_count_c2 --beta 0 --agg count --agg-p 2
run b0_count_c3 --beta 0 --agg count --agg-p 3
st "LEVER9B_DONE"
