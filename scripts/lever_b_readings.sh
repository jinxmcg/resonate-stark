#!/bin/bash
# Lever B, step 1: four deterministic readings of every val question (plain + paraphrased) through the P3 pipeline; oracle best-of-readings.
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
RR=data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json; W=0.45
st() { echo "$(date +%H:%M:%S) $1" >> logs/leverB_status.log; }
FLOOR=$(uv run python -c "import torch; print(torch.load('models/lp.pt', map_location='cpu', weights_only=False)['sim_floor'])" 2>/dev/null | tail -1)
LOW=$(python3 -c "print(round(float('$FLOOR')*0.8, 3))")
st "readings: floor $FLOOR, low floor $LOW"
mkdir -p results_b
for V in plain para; do
  if [ $V = plain ]; then Q=""; TX=data/text_val_bgeft2.json; TL=data/text_val_pjoint.json; else Q="--queries data/para_val.json"; TX=data/text_val_bgeft2_para.json; TL=data/text_val_pjoint_para.json; fi
  st "$V: parses (low floor, second type)"
  uv run python latent_parser.py --predict val --tag lp $Q --sim-floor $LOW --out data/lparse_val_${V}_low.json > logs/lb_parse_${V}_low.log 2>&1
  uv run python latent_parser.py --predict val --tag lp $Q --type-rank 2 --out data/lparse_val_${V}_t2.json > logs/lb_parse_${V}_t2.log 2>&1
  [ $V = plain ] && LP=data/lparse_val.json || LP=data/lparse_val_para.json
  st "$V: retrievals"
  uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse $LP --out data/rel_val_${V}_r1.json > logs/lb_rel_${V}_r1.log 2>&1
  uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --out data/rel_val_${V}_r2.json > logs/lb_rel_${V}_r2.log 2>&1
  uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse data/lparse_val_${V}_low.json --out data/rel_val_${V}_r3.json > logs/lb_rel_${V}_r3.log 2>&1
  uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse data/lparse_val_${V}_t2.json --out data/rel_val_${V}_r4.json > logs/lb_rel_${V}_r4.log 2>&1
  st "$V: scoring the four readings"
  for r in r1 r2 r3 r4; do
    uv run python predict.py --split val --rel data/rel_val_${V}_$r.json --text $TX --t2l $TL --w $W --rerank $RR --score 2>&1 | grep "COMMITTED" | sed "s/^/    $V $r /" >> logs/leverB_status.log
    mv eval_results_val.csv results_b/val_${V}_$r.csv
  done
done
st "LEVERB_READINGS_DONE"
