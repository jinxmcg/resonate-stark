#!/bin/bash
# P5 fail-fast on jinx (GTX 1080 Ti): reduced grid on 1,000 train questions (plain + terse), then, only if a setting helps terse
# without hurting plain, terse val paraphrases with Qwen2.5-3B (fp16) and the three-wording val protocol.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/p5_status.log; }
rm -f logs/p5_status.log; st "grid on 1,000 train questions (plain + terse)"
uv run python latent_parser.py --predict train --tag lp_p3 --out data/lparse_train_full.json > logs/p5_lp_train.log 2>&1
uv run python latent_parser.py --predict train --tag lp_p3 --queries data/para_train_terse.json --out data/lparse_train_terse.json > logs/p5_lp_train_terse.log 2>&1
for W in plain terse; do
  if [ $W = plain ]; then Q=""; LP=data/lparse_train_full.json; else Q="--queries data/para_train_terse.json"; LP=data/lparse_train_terse.json; fi
  uv run python retrieve.py --model $M --split train --limit 1000 --beta 30 --anchor bge $Q --lparse $LP --out data/rel_tmp.json 2>&1 | grep "all queries" | cut -c1-140 | sed "s/^/    train $W  no-confirm         /" >> logs/p5_status.log
  for CFG in "0.8 0.75" "0.9 0.75" "0.8 1.0" "0.9 1.0"; do
    REL=${CFG% *}; ABS=${CFG#* }
    uv run python retrieve.py --model $M --split train --limit 1000 --beta 30 --anchor bge $Q --lparse $LP --confirm models/lp_p3.pt --confirm-rel $REL --confirm-abs $ABS --out data/rel_tmp.json 2>&1 | grep "all queries\|confirm:" | cut -c1-140 | tr '\n' ' ' | sed "s/^/    train $W  rel $REL abs $ABS  /" >> logs/p5_status.log; echo >> logs/p5_status.log
  done
done
st "GRID_DONE"
