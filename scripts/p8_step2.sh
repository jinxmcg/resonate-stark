#!/bin/bash
# P8 step 2 (RTX 5090, vast 50261550): ONE val read of A' (parser-head anchors) and A'+B' (plus the
# rank-384 document matrix chosen on train in step 1), against P3. Reads train and val only.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; D=384
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p8s2_status.log; }
rm -f logs/p8s2_status.log; T=p8A_lp_ancf
st "A': train retrievals with --anchor lp (out-of-fold parses)"
$PY retrieve.py --model models/p_k12b4_50k.pt --split train --beta 30 --anchor lp --dump-feats \
    --lparse data/lparse_train_oof.json --out data/rel_train_${T}.json > logs/p8s2_train.log 2>&1
$PY retrieve.py --model models/p_k12b4_50k.pt --split train --beta 30 --anchor lp --dump-feats \
    --queries data/para_train.json --lparse data/lparse_train_oof_para.json --out data/rel_train_para_${T}.json > logs/p8s2_train_para.log 2>&1
st "  train done; val retrievals"
$PY retrieve.py --model models/p_k12b4_50k.pt --split val --beta 30 --anchor lp --dump-feats \
    --lparse data/lparse_val.json --out data/rel_val_${T}.json > logs/p8s2_val.log 2>&1
grep "covered\|all queries" logs/p8s2_val.log | tr '\n' ' ' | cut -c1-250 | sed "s/^/    A' relational val plain  /" >> logs/p8s2_status.log; echo >> logs/p8s2_status.log
$PY retrieve.py --model models/p_k12b4_50k.pt --split val --beta 30 --anchor lp --dump-feats \
    --queries data/para_val.json --lparse data/lparse_val_para.json --out data/rel_val_para_${T}.json > logs/p8s2_val_para.log 2>&1
grep "covered\|all queries" logs/p8s2_val_para.log | tr '\n' ' ' | cut -c1-250 | sed "s/^/    A' relational val para   /" >> logs/p8s2_status.log; echo >> logs/p8s2_status.log
st "  val done; projected val text rankings (d=$D)"
$PY embed_text2.py --model bgeft2 --docs data/doc_emb_bgeft2_p${D}.npy --proj data/docproj_bgeft2_${D}.npy --splits val --tag _d${D} > logs/p8s2_txt.log 2>&1
$PY embed_text2.py --model bgeft2 --docs data/doc_emb_bgeft2_p${D}.npy --proj data/docproj_bgeft2_${D}.npy --splits val --queries data/para_val.json --tag _d${D}_para > logs/p8s2_txt_para.log 2>&1
grep -h "text-only" logs/p8s2_txt.log logs/p8s2_txt_para.log | sed "s/^/    d=$D val /" >> logs/p8s2_status.log
st "  reranker refit (out-of-fold features)"
$PY rerank.py --rel-tag $T --text-tag bgeft2 --train-text-tag bgeft2oof --aug-rel-tag para_${T} \
    --aug-text-tag bgeft2oof_para --t2l-tag pjoint --train-t2l-tag pjointoof --aug-t2l-tag pjointoof_para \
    --save-tag _oof > logs/p8s2_rerank.log 2>&1
grep "VAL reranked" logs/p8s2_rerank.log | sed "s/^/    A' /" >> logs/p8s2_status.log
st "  refit done; val scoring"
RR=data/rerank_${T}_bgeft2_pjoint_aug_oof.json
for V in plain para; do
  if [ $V = plain ]; then RL=data/rel_val_${T}.json; TX=data/text_val_bgeft2.json; TXP=data/text_val_bgeft2_d${D}.json; TL=data/text_val_pjoint.json
  else RL=data/rel_val_para_${T}.json; TX=data/text_val_bgeft2_para.json; TXP=data/text_val_bgeft2_d${D}_para.json; TL=data/text_val_pjoint_para.json; fi
  $PY predict.py --split val --rel $RL --text $TX  --t2l $TL --w 0.45 --rerank $RR --score 2>&1 | grep COMMITTED | sed "s/^/    A'      val $V  /" >> logs/p8s2_status.log
  $PY predict.py --split val --rel $RL --text $TXP --t2l $TL --w 0.45 --rerank $RR --score 2>&1 | grep COMMITTED | sed "s/^/    A'+B'   val $V  /" >> logs/p8s2_status.log
done
rm -f eval_results_val.csv
st "P8_STEP2_DONE"
