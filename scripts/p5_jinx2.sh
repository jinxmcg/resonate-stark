#!/bin/bash
# P5 fail-fast, second grid (jinx 1080 Ti): the first grid's absolute floor (0.75-1.0 x parser sim_floor) dropped half of all
# string-matched mentions on plain train questions (Hit@1 26.2 -> 19.9). This grid tests the collision-only reading: no absolute
# floor (every matched name keeps at least its best candidate), relative thresholds {0.5, 0.8, 0.95}, plus one low floor (abs 0.5).
# Reads: train split only (1,000 questions, plain + terse paraphrases). Parses reused from scripts/p5_jinx.sh.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/p5_status2.log; }
rm -f logs/p5_status2.log; st "grid 2 on 1,000 train questions (plain + terse): rel-only + abs 0.5"
for W in plain terse; do
  if [ $W = plain ]; then Q=""; LP=data/lparse_train_full.json; else Q="--queries data/para_train_terse.json"; LP=data/lparse_train_terse.json; fi
  for CFG in "0.5 0" "0.8 0" "0.95 0" "0.8 0.5"; do
    REL=${CFG% *}; ABS=${CFG#* }
    uv run python retrieve.py --model $M --split train --limit 1000 --beta 30 --anchor bge $Q --lparse $LP --confirm models/lp_p3.pt --confirm-rel $REL --confirm-abs $ABS --out data/rel_tmp.json 2>&1 | grep "all queries\|confirm:" | cut -c1-140 | tr '\n' ' ' | sed "s/^/    train $W  rel $REL abs $ABS  /" >> logs/p5_status2.log; echo >> logs/p5_status2.log
  done
done
st "GRID2_DONE"
