#!/bin/bash
# lever 9: AND readouts in the space, model only (beta 0) and with the walk (beta 30); val plain + paraphrased
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/lever9_status.log; }
run() { # $1 tag, rest = extra args
  T=$1; shift
  uv run python retrieve.py --model $M --split val --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --out data/rel_tmp_$T.json "$@" > logs/l9_$T.log 2>&1
  uv run python retrieve.py --model $M --split val --anchor bge --queries data/para_val.json --llm-parse data/llmparse_val_para.json --llm-override-type --out data/rel_tmp_${T}_para.json "$@" > logs/l9_${T}_para.log 2>&1
  echo "== $T" >> logs/lever9_status.log
  for f in logs/l9_$T.log logs/l9_${T}_para.log; do grep "all queries\|>= 2 mentions" $f | sed "s/^/    /" >> logs/lever9_status.log; done
}
st "model only (beta 0)"
run b0_sum --beta 0 --agg sum
run b0_min --beta 0 --agg min
run b0_softmin1 --beta 0 --agg softmin --agg-p 1
run b0_softmin05 --beta 0 --agg softmin --agg-p 0.5
run b0_softmin2 --beta 0 --agg softmin --agg-p 2
run b0_logsig0 --beta 0 --agg logsig --agg-p 0
run b0_logsig1 --beta 0 --agg logsig --agg-p 1
st "with the walk (beta 30)"
run b30_sum --beta 30 --agg sum
run b30_min --beta 30 --agg min
run b30_softmin1 --beta 30 --agg softmin --agg-p 1
st "LEVER9_DONE"
