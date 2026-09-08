#!/bin/bash
# P15 RESUME (2026-09-08): steps C and D only, after two defects in p15_full.sh made the
# first pass write P15FULL_DONE having scored nothing:
#   1. it never wrote data/fusion_st.json (the RRF weight rerank.py reads). Supplied as
#      w=0.45 -- the value data/fusion_bgeft2.json holds and the value step D already
#      passes as --w, so "same hyperparameters" as the registration requires.
#   2. step B wrote the val t2l rankings as t2l_val_st_<register>.json while rerank.py
#      reads data/text_val_<t2l_tag>.json. Same artefact (verified by role and byte size
#      against its pjoint counterpart); copied to the name rerank expects.
# Both were swallowed by `predict.py ... 2>&1 | grep COMMITTED`; caught only because step D
# took 106 seconds instead of ten minutes. This version reports per-register failures.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib VIRTUAL_ENV=/venv/main PATH=/venv/main/bin:$PATH
PY=/venv/main/bin/python; T=p15_st_ancf
st(){ echo "$(date +%H:%M:%S) $1" >> logs/p15f_status.log; }
st "C-resume. reranker refit (fusion_st.json supplied)"
$PY rerank.py --rel-tag $T --text-tag st --train-text-tag stoof --aug-rel-tag para_${T} --aug-text-tag stoof_para \
   --t2l-tag stt2l --train-t2l-tag stt2loof --aug-t2l-tag stt2loof_para --save-tag _oof > logs/p15f_rerank.log 2>&1
RR=data/rerank_${T}_st_stt2l_aug_oof.json
if [ ! -f $RR ]; then st "ABORT rerank: $(tail -2 logs/p15f_rerank.log | tr "\n" " " | cut -c1-160)"; exit 1; fi
grep "VAL reranked" logs/p15f_rerank.log | sed "s/^/    /" >> logs/p15f_status.log
st "D-resume. val scoring"
for W in plain para terseA terseB terseC; do
  case $W in
    plain) RL=data/rel_val_${T}.json ;; para) RL=data/rel_val_para_${T}.json ;;
    *)     RL=data/rel_val_${W}_${T}.json ;;
  esac
  OUT=$($PY predict.py --split val --rel $RL --text data/text_val_st_$W.json --t2l data/t2l_val_st_$W.json --w 0.45 --rerank $RR --score 2>&1)
  echo "$OUT" | grep COMMITTED | sed "s/^/    P15full $W  /" >> logs/p15f_status.log \
    || echo "    P15full $W FAILED: $(echo "$OUT" | tail -1 | cut -c1-120)" >> logs/p15f_status.log
done
rm -f eval_results_val.csv
st "P15FULL_DONE (resumed)"
