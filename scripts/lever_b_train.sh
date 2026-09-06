#!/bin/bash
# Lever B step 2 data: the four readings on TRAIN (plain + paraphrased) with OUT-OF-FOLD parses, and on VAL, with selector features.
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
RR=data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json; W=0.45
st() { echo "$(date +%H:%M:%S) $1" >> logs/leverB2_status.log; }
FLOOR=$(uv run python -c "import torch; print(torch.load('models/lp.pt', map_location='cpu', weights_only=False)['sim_floor'])" 2>/dev/null | tail -1); LOW=$(python3 -c "print(round(float('$FLOOR')*0.8, 3))")
mkdir -p data/oof results_b
st "train: out-of-fold parses for r1 (with probs), r3, r4"
for k in 0 1 2 3 4; do for V in plain para; do
  [ $V = plain ] && Q="" || Q="--queries data/para_train.json"
  uv run python latent_parser.py --predict train --tag lp_f$k --fold 5:$k $Q --out data/oof/lpB_train_${V}_r1_f$k.json > /dev/null 2>&1
  uv run python latent_parser.py --predict train --tag lp_f$k --fold 5:$k $Q --sim-floor $LOW --out data/oof/lpB_train_${V}_r3_f$k.json > /dev/null 2>&1
  uv run python latent_parser.py --predict train --tag lp_f$k --fold 5:$k $Q --type-rank 2 --out data/oof/lpB_train_${V}_r4_f$k.json > /dev/null 2>&1
done; st "  fold $k parsed"; done
python3 - <<'PY'
import json
for V in ("plain","para"):
    for r in ("r1","r3","r4"):
        out={}
        for k in range(5): out.update(json.load(open(f"data/oof/lpB_train_{V}_{r}_f{k}.json")))
        json.dump(out, open(f"data/lpB_train_{V}_{r}.json","w"))
PY
st "val: parses with probs"
uv run python latent_parser.py --predict val --tag lp --out data/lpB_val_plain_r1.json > /dev/null 2>&1
uv run python latent_parser.py --predict val --tag lp --sim-floor $LOW --out data/lpB_val_plain_r3.json > /dev/null 2>&1
uv run python latent_parser.py --predict val --tag lp --type-rank 2 --out data/lpB_val_plain_r4.json > /dev/null 2>&1
uv run python latent_parser.py --predict val --tag lp --queries data/para_val.json --out data/lpB_val_para_r1.json > /dev/null 2>&1
uv run python latent_parser.py --predict val --tag lp --queries data/para_val.json --sim-floor $LOW --out data/lpB_val_para_r3.json > /dev/null 2>&1
uv run python latent_parser.py --predict val --tag lp --queries data/para_val.json --type-rank 2 --out data/lpB_val_para_r4.json > /dev/null 2>&1
for S in train val; do for V in plain para; do
  if [ $S = train ]; then [ $V = plain ] && Q="" || Q="--queries data/para_train.json"; TX=data/text_train_bgeft2$([ $V = para ] && echo _para).json; TL=data/text_train_pjointoof$([ $V = para ] && echo _para).json
  else [ $V = plain ] && Q="" || Q="--queries data/para_val.json"; TX=data/text_val_bgeft2$([ $V = para ] && echo _para).json; TL=data/text_val_pjoint$([ $V = para ] && echo _para).json; fi
  st "$S $V: retrievals r1 r2 r3 r4"
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats $Q --lparse data/lpB_${S}_${V}_r1.json --out data/relB_${S}_${V}_r1.json > logs/lbB_rel_${S}_${V}_r1.log 2>&1
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats $Q --out data/relB_${S}_${V}_r2.json > logs/lbB_rel_${S}_${V}_r2.log 2>&1
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats $Q --lparse data/lpB_${S}_${V}_r3.json --out data/relB_${S}_${V}_r3.json > logs/lbB_rel_${S}_${V}_r3.log 2>&1
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats $Q --lparse data/lpB_${S}_${V}_r4.json --out data/relB_${S}_${V}_r4.json > logs/lbB_rel_${S}_${V}_r4.log 2>&1
  st "$S $V: scoring"
  for r in r1 r2 r3 r4; do
    uv run python predict.py --split $S --rel data/relB_${S}_${V}_$r.json --text $TX --t2l $TL --w $W --rerank $RR --scores-out data/scB_${S}_${V}_$r.json --score 2>&1 | grep "COMMITTED" | sed "s/^/    $S $V $r /" >> logs/leverB2_status.log
    mv eval_results_$S.csv results_b/${S}_${V}_$r.csv
  done
done; done
st "LEVERB2_DATA_DONE"
