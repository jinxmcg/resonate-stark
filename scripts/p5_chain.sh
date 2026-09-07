#!/bin/bash
# P5: latent-confirmed anchors. (1) terse paraphrases of val; (2) grid of (REL, ABS) on 2,000 TRAIN questions, plain + terse;
# (3) val plain / natural / terse with the chosen setting vs P3 (no confirm). Relational path only; the pipeline refit is a separate step.
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/p5_status.log; }
st "1. terse paraphrases of val"
[ -f data/para_val_terse.json ] || uv run python paraphrase.py --split val --seed 3 --style terse --out data/para_val_terse.json > logs/para_val_terse.log 2>&1
st "  $(grep PARA_DONE logs/para_val_terse.log)"
st "2. train grid (2,000 questions, plain + terse), relational path with the latent parser + bge fallback"
uv run python latent_parser.py --predict train --tag lp --limit 0 --out data/lparse_train_full.json > /dev/null 2>&1 || true
uv run python latent_parser.py --predict train --tag lp --queries data/para_train_terse.json --out data/lparse_train_terse.json > /dev/null 2>&1 || true
for W in plain terse; do
  if [ $W = plain ]; then Q=""; LP=data/lparse_train_full.json; else Q="--queries data/para_train_terse.json"; LP=data/lparse_train_terse.json; fi
  uv run python retrieve.py --model $M --split train --limit 2000 --beta 30 --anchor bge $Q --lparse $LP --out data/rel_tmp.json 2>&1 | grep "all queries" | sed "s/^/    train $W  no-confirm  /" >> logs/p5_status.log
  for REL in 0.7 0.8 0.9; do for ABS in 0.5 0.75 1.0; do
    uv run python retrieve.py --model $M --split train --limit 2000 --beta 30 --anchor bge $Q --lparse $LP --confirm models/lp.pt --confirm-rel $REL --confirm-abs $ABS --out data/rel_tmp.json 2>&1 | grep "all queries\|confirm:" | tr '\n' ' ' | sed "s/^/    train $W  rel $REL abs $ABS  /" >> logs/p5_status.log; echo >> logs/p5_status.log
  done; done
done
st "GRID_DONE — choose (REL, ABS) on the train lines above, then run step 3 with P5_REL / P5_ABS set"
if [ -n "$P5_REL" ]; then
  st "3. val: plain / natural / terse, no-confirm vs confirm rel $P5_REL abs $P5_ABS"
  uv run python latent_parser.py --predict val --tag lp --queries data/para_val_terse.json --out data/lparse_val_terse.json > /dev/null 2>&1
  for W in plain para terse; do
    case $W in plain) Q=""; LP=data/lparse_val.json;; para) Q="--queries data/para_val.json"; LP=data/lparse_val_para.json;; terse) Q="--queries data/para_val_terse.json"; LP=data/lparse_val_terse.json;; esac
    uv run python retrieve.py --model $M --split val --beta 30 --anchor bge $Q --lparse $LP --dump-feats --out data/rel_val_${W}_p3.json 2>&1 | grep "all queries" | sed "s/^/    val $W  no-confirm  /" >> logs/p5_status.log
    uv run python retrieve.py --model $M --split val --beta 30 --anchor bge $Q --lparse $LP --dump-feats --confirm models/lp.pt --confirm-rel $P5_REL --confirm-abs $P5_ABS --out data/rel_val_${W}_p5.json 2>&1 | grep "all queries\|confirm:" | tr '\n' ' ' | sed "s/^/    val $W  confirm     /" >> logs/p5_status.log; echo >> logs/p5_status.log
  done
  st "P5_VAL_DONE"
fi
