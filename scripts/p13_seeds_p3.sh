#!/bin/bash
# P13 amendment (RTX 5090, vast 50261550): lp_p3's own seed variance, so P11 is compared mean to mean.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p13b_status.log; }
rm -f logs/p13b_status.log; st "lp_p3 at seeds 1 and 2 (original labels, no terse)"
for S in 1 2; do
  $PY latent_parser.py --train --tag lp_p3_s$S --labels data/lp_labels.json --seed $S > logs/p13b_train_s$S.log 2>&1
  if [ ! -f models/lp_p3_s$S.pt ]; then st "ABORT: seed $S did not train — $(tail -1 logs/p13b_train_s$S.log | cut -c1-120)"; exit 1; fi
  st "  seed $S trained"
done
RR3=data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json
for W in plain terseA terseB terseC; do
  case $W in
    plain)  Q="";                                     TXF=data/text_val_bgeft2.json;        TL=data/text_val_pjoint.json ;;
    terseA) Q="--queries data/para_val_terse.json";   TXF=data/text_val_bgeft2_terse.json;  TL=data/text_val_pjoint_terse.json ;;
    terseB) Q="--queries data/para_val_terse_b.json"; TXF=data/text_val_bgeft2_terseb.json; TL=data/text_val_pjoint_terseb.json ;;
    terseC) Q="--queries data/para_val_terse_c.json"; TXF=data/text_val_bgeft2_tersec.json; TL=data/text_val_pjoint_tersec.json ;;
  esac
  for S in 1 2; do
    LP=data/lparse_val_lp_p3_s${S}_$W.json
    $PY latent_parser.py --predict val --tag lp_p3_s$S $Q --out $LP > logs/p13b_lp_s${S}_$W.log 2>&1
    $PY retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse $LP --out data/rel_val_p13b_s${S}_$W.json > logs/p13b_rel_s${S}_$W.log 2>&1
    $PY predict.py --split val --rel data/rel_val_p13b_s${S}_$W.json --text $TXF --t2l $TL --w 0.45 --rerank $RR3 --score 2>&1 | grep COMMITTED | sed "s/^/    P3 seed $S  $W  /" >> logs/p13b_status.log
  done
  st "  $W done"
done
rm -f eval_results_val.csv
st "P13B_DONE"
