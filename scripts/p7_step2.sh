#!/bin/bash
# P7 step 2 (rented RTX 5090, vast 50261550): the two size-reducing configurations through the full
# P3 chain, decided on val. ref = P3 as it stands (control, existing reranker); C = joint table;
# AC = joint table + the text ranker's encoder for anchors. Reads train and val only.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p7s2_status.log; }
rm -f logs/p7s2_status.log
st "control: P3 as it stands, val plain + paraphrased, existing reranker"
$PY retrieve.py --model models/p_k12b4_50k.pt --split val --beta 30 --anchor bge --dump-feats \
    --lparse data/lparse_val.json --out data/rel_val_ref_lp_ancf.json > logs/p7s2_ref_val.log 2>&1
$PY retrieve.py --model models/p_k12b4_50k.pt --split val --beta 30 --anchor bge --dump-feats \
    --queries data/para_val.json --lparse data/lparse_val_para.json --out data/rel_val_para_ref_lp_ancf.json > logs/p7s2_ref_val_para.log 2>&1
for V in plain para; do
  if [ $V = plain ]; then RL=data/rel_val_ref_lp_ancf.json; TX=data/text_val_bgeft2.json; TL=data/text_val_pjoint.json; else RL=data/rel_val_para_ref_lp_ancf.json; TX=data/text_val_bgeft2_para.json; TL=data/text_val_pjoint_para.json; fi
  $PY predict.py --split val --rel $RL --text $TX --t2l $TL --w 0.45 --rerank data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    ref  val $V  /" >> logs/p7s2_status.log
done
st "  control done"
for ARM in C AC; do
  if [ $ARM = C ]; then M=models/p_joint.pt; ANC=bge; else M=models/p_joint.pt; ANC=bgeft2; fi
  T=p7${ARM}_lp_ancf
  st "arm $ARM ($ANC + $(basename $M .pt)): train retrievals (out-of-fold parses)"
  $PY retrieve.py --model $M --split train --beta 30 --anchor $ANC --dump-feats --lparse data/lparse_train_oof.json \
      --out data/rel_train_${T}.json > logs/p7s2_${ARM}_train.log 2>&1
  $PY retrieve.py --model $M --split train --beta 30 --anchor $ANC --dump-feats --queries data/para_train.json \
      --lparse data/lparse_train_oof_para.json --out data/rel_train_para_${T}.json > logs/p7s2_${ARM}_train_para.log 2>&1
  st "  train done; val retrievals"
  $PY retrieve.py --model $M --split val --beta 30 --anchor $ANC --dump-feats --lparse data/lparse_val.json \
      --out data/rel_val_${T}.json > logs/p7s2_${ARM}_val.log 2>&1
  grep "all queries" logs/p7s2_${ARM}_val.log | sed "s/^/    $ARM relational val plain  /" >> logs/p7s2_status.log
  $PY retrieve.py --model $M --split val --beta 30 --anchor $ANC --dump-feats --queries data/para_val.json \
      --lparse data/lparse_val_para.json --out data/rel_val_para_${T}.json > logs/p7s2_${ARM}_val_para.log 2>&1
  grep "all queries" logs/p7s2_${ARM}_val_para.log | sed "s/^/    $ARM relational val para   /" >> logs/p7s2_status.log
  st "  val done; reranker refit (out-of-fold features)"
  $PY rerank.py --rel-tag $T --text-tag bgeft2 --train-text-tag bgeft2oof --aug-rel-tag para_${T} \
      --aug-text-tag bgeft2oof_para --t2l-tag pjoint --train-t2l-tag pjointoof --aug-t2l-tag pjointoof_para \
      --save-tag _oof > logs/p7s2_${ARM}_rerank.log 2>&1
  grep "global weights\|VAL reranked" logs/p7s2_${ARM}_rerank.log | sed "s/^/    $ARM /" >> logs/p7s2_status.log
  st "  refit done; val scoring"
  RR=data/rerank_${T}_bgeft2_pjoint_aug_oof.json
  for V in plain para; do
    if [ $V = plain ]; then RL=data/rel_val_${T}.json; TX=data/text_val_bgeft2.json; TL=data/text_val_pjoint.json; else RL=data/rel_val_para_${T}.json; TX=data/text_val_bgeft2_para.json; TL=data/text_val_pjoint_para.json; fi
    $PY predict.py --split val --rel $RL --text $TX --t2l $TL --w 0.45 --rerank $RR --score 2>&1 | grep COMMITTED | sed "s/^/    $ARM   val $V  /" >> logs/p7s2_status.log
  done
  st "  arm $ARM done"
done
rm -f eval_results_val.csv
st "P7_STEP2_DONE"
