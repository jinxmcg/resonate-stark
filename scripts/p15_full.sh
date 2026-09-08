#!/bin/bash
# P15 FULL BUILD (box 50261550): the shared trunk end to end at text-loss 3x, one val read.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p15f_status.log; }
rm -f logs/p15f_status.log; st "A. five out-of-fold trunks + one full trunk (w-text 3)"
for k in 0 1 2 3 4; do
  $PY shared_trunk.py --train --tag st_f$k --fold 5:$k --w-text 3 > logs/p15f_tr_f$k.log 2>&1
  if [ ! -f models/st_f$k.pt ]; then st "ABORT: fold $k trunk failed — $(tail -1 logs/p15f_tr_f$k.log | cut -c1-120)"; exit 1; fi
  $PY shared_trunk.py --predict-text  train --tag st_f$k --fold 5:$k --out data/oof/st_text_f$k.json      > logs/p15f_tx_f$k.log 2>&1
  $PY shared_trunk.py --predict-text  train --tag st_f$k --fold 5:$k --queries data/para_train.json --out data/oof/st_text_f${k}_para.json > logs/p15f_txp_f$k.log 2>&1
  $PY shared_trunk.py --predict-t2l   train --tag st_f$k --fold 5:$k --out data/oof/st_t2l_f$k.json       > logs/p15f_tl_f$k.log 2>&1
  $PY shared_trunk.py --predict-t2l   train --tag st_f$k --fold 5:$k --queries data/para_train.json --out data/oof/st_t2l_f${k}_para.json  > logs/p15f_tlp_f$k.log 2>&1
  $PY shared_trunk.py --predict-parse train --tag st_f$k --fold 5:$k --out data/oof/st_parse_f$k.json     > logs/p15f_ps_f$k.log 2>&1
  $PY shared_trunk.py --predict-parse train --tag st_f$k --fold 5:$k --queries data/para_train.json --out data/oof/st_parse_f${k}_para.json > logs/p15f_psp_f$k.log 2>&1
  st "  fold $k done"
done
$PY shared_trunk.py --train --tag st_full --w-text 3 > logs/p15f_tr_full.log 2>&1
if [ ! -f models/st_full.pt ]; then st "ABORT: full trunk failed — $(tail -1 logs/p15f_tr_full.log | cut -c1-120)"; exit 1; fi
grep -E "anchor floor|saved" logs/p15f_tr_full.log | sed "s/^/    /" >> logs/p15f_status.log
$PY - <<'PY' >> logs/p15f_status.log 2>&1
import json
for kind in ("text", "t2l", "parse"):
    for suf in ("", "_para"):
        out = {}
        for k in range(5): out.update(json.load(open(f"data/oof/st_{kind}_f{k}{suf}.json")))
        name = {"text": "text_train_stoof", "t2l": "text_train_stt2loof", "parse": "lparse_train_stoof"}[kind]
        json.dump(out, open(f"data/{name}{suf}.json", "w")); print(f"    merged {name}{suf}: {len(out)}")
PY
st "B. full trunk: corpus embedding, refit projection, val rankings"
$PY shared_trunk.py --predict-text val --tag st_full --out data/text_val_st.json > logs/p15f_val_text.log 2>&1
$PY proj_docs.py --emb data/doc_emb_st_full.npy --dims 384 >> logs/p15f_status.log 2>&1
for W in plain para terseA terseB terseC; do
  case $W in
    plain)  Q="" ;;  para) Q="--queries data/para_val.json" ;;
    terseA) Q="--queries data/para_val_terse.json" ;; terseB) Q="--queries data/para_val_terse_b.json" ;; terseC) Q="--queries data/para_val_terse_c.json" ;;
  esac
  $PY shared_trunk.py --predict-text  val --tag st_full $Q --out data/text_val_st_$W.json  > logs/p15f_vt_$W.log 2>&1
  $PY shared_trunk.py --predict-t2l   val --tag st_full $Q --out data/t2l_val_st_$W.json   > logs/p15f_vl_$W.log 2>&1
  $PY shared_trunk.py --predict-parse val --tag st_full $Q --out data/lparse_val_st_$W.json > logs/p15f_vp_$W.log 2>&1
done
st "C. retrievals (trunk anchors) and the reranker refit"
T=p15_st_ancf
$PY retrieve.py --model $M --split train --beta 30 --anchor st --st-anchor models/st_full.pt --dump-feats --lparse data/lparse_train_stoof.json --out data/rel_train_${T}.json > logs/p15f_rel_tr.log 2>&1
$PY retrieve.py --model $M --split train --beta 30 --anchor st --st-anchor models/st_full.pt --dump-feats --queries data/para_train.json --lparse data/lparse_train_stoof_para.json --out data/rel_train_para_${T}.json > logs/p15f_rel_trp.log 2>&1
for W in plain para terseA terseB terseC; do
  case $W in
    plain)  Q=""; O=data/rel_val_${T}.json ;;  para) Q="--queries data/para_val.json"; O=data/rel_val_para_${T}.json ;;
    terseA) Q="--queries data/para_val_terse.json"; O=data/rel_val_terseA_${T}.json ;;
    terseB) Q="--queries data/para_val_terse_b.json"; O=data/rel_val_terseB_${T}.json ;;
    terseC) Q="--queries data/para_val_terse_c.json"; O=data/rel_val_terseC_${T}.json ;;
  esac
  $PY retrieve.py --model $M --split val --beta 30 --anchor st --st-anchor models/st_full.pt --dump-feats $Q --lparse data/lparse_val_st_$W.json --out $O > logs/p15f_rel_$W.log 2>&1
done
cp data/text_val_st_plain.json data/text_val_st.json 2>/dev/null
$PY rerank.py --rel-tag $T --text-tag st --train-text-tag stoof --aug-rel-tag para_${T} --aug-text-tag stoof_para \
   --t2l-tag stt2l --train-t2l-tag stt2loof --aug-t2l-tag stt2loof_para --save-tag _oof > logs/p15f_rerank.log 2>&1
grep "VAL reranked" logs/p15f_rerank.log | sed "s/^/    /" >> logs/p15f_status.log
st "D. val scoring"
RR=data/rerank_${T}_st_stt2l_aug_oof.json
for W in plain para terseA terseB terseC; do
  case $W in
    plain)  RL=data/rel_val_${T}.json ;;  para) RL=data/rel_val_para_${T}.json ;;
    *)      RL=data/rel_val_${W}_${T}.json ;;
  esac
  $PY predict.py --split val --rel $RL --text data/text_val_st_$W.json --t2l data/t2l_val_st_$W.json --w 0.45 --rerank $RR --score 2>&1 | grep COMMITTED | sed "s/^/    P15full $W  /" >> logs/p15f_status.log
done
rm -f eval_results_val.csv
st "P15FULL_DONE"
