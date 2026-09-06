#!/bin/bash
# Out-of-fold features for the reranker fit (fix for the in-sample train rankings): 5-fold joint text-to-latent
# and 5-fold bge fine-tune, merged into data/text_train_{t2ljoof,bgeft2oof}[_para].json; then reranker refits and val scoring.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; mkdir -p data/oof
for k in 0 1 2 3 4; do
  uv run python text2latent.py --encoder models/bge_ft2 --extra data/para_train.json --tag t2lj --unfreeze-table --fold 5:$k > logs/oof_t2lj_f$k.log 2>&1
done
for k in 0 1 2 3 4; do
  uv run python oof_embed.py --fold 5:$k --extra data/para_train.json > logs/oof_bgeft2_f$k.log 2>&1
done
python3 - <<'PY'
import json
for name, tag in (("t2ljoof", "t2lj"), ("bgeft2oof", "bgeft2")):
    for suf in ("", "_para"):
        out = {}
        for k in range(5): out.update(json.load(open(f"data/oof/{tag}_f{k}{suf}.json")))
        json.dump(out, open(f"data/text_train_{name}{suf}.json", "w")); print(name, suf, len(out))
PY
echo "== oof fits" > logs/oof_eval.log
for U in none t2lj; do
  T=""; [ $U != none ] && T="--t2l-tag t2lj --train-t2l-tag t2ljoof --aug-t2l-tag t2ljoof_para"
  uv run python rerank.py --rel-tag llm_ancf --text-tag bgeft2 --train-text-tag bgeft2oof --aug-rel-tag para_llm_ancf --aug-text-tag bgeft2oof_para $T --save-tag _oof > logs/oof_rerank_$U.log 2>&1
  RR=data/rerank_llm_ancf_bgeft2; [ $U != none ] && RR=${RR}_t2lj; RR=${RR}_aug_oof.json
  for V in plain para; do
    if [ $V = plain ]; then RL=data/rel_val_llm_ancf.json; TX=data/text_val_bgeft2.json; TL=data/text_val_t2lj.json; else RL=data/rel_val_para_llm_ancf.json; TX=data/text_val_bgeft2_para.json; TL=data/text_val_t2lj_para.json; fi
    TT=""; [ $U != none ] && TT="--t2l $TL"
    echo "== oof fit t2l=$U val=$V" >> logs/oof_eval.log
    uv run python predict.py --split val --rel $RL --text $TX --w 0.5 --rerank $RR $TT --score 2>&1 | grep "COMMITTED\|Error" >> logs/oof_eval.log
  done
done
rm -f eval_results_val.csv
echo OOF_EVAL_DONE >> logs/oof_eval.log
