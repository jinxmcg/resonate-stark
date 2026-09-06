#!/bin/bash
# out-of-fold rankings from the JOINT (edges + questions) model, then the reranker with them; waits for the first OOF chain to finish
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib
for k in 0 1 2 3 4; do uv run python joint_train.py --tag p_jointv4 --nvec 4 --extra data/para_train.json --fold 5:$k > logs/oof_pjointv4_f$k.log 2>&1; done
python3 - <<'PY'
import json
for suf in ("", "_para"):
    out = {}
    for k in range(5): out.update(json.load(open(f"data/oof/p_jointv4_f{k}{suf}.json")))
    json.dump(out, open(f"data/text_train_pjointv4oof{suf}.json", "w")); print("pjointv4oof", suf, len(out))
PY
ln -sfn p_jointv4_enc models/p_jointv4_head_enc
uv run python t2l_rank.py --tag p_jointv4_head --model models/p_jointv4.pt --split val --out data/text_val_pjointv4.json > logs/rank_pjointv4_val.log 2>&1
uv run python t2l_rank.py --tag p_jointv4_head --model models/p_jointv4.pt --split val --queries data/para_val.json --out data/text_val_pjointv4_para.json > logs/rank_pjointv4_val_para.log 2>&1

W=$(python3 -c "import json; print(json.load(open('data/fusion_bgeft2.json'))['w'])")
uv run python rerank.py --rel-tag llm_ancf --text-tag bgeft2 --train-text-tag bgeft2oof --aug-rel-tag para_llm_ancf --aug-text-tag bgeft2oof_para --t2l-tag pjointv4 --train-t2l-tag pjointv4oof --aug-t2l-tag pjointv4oof_para --save-tag _oof > logs/oof_rerank_pjointv4.log 2>&1
echo "== joint-oof fit, fusion w=$W" > logs/oof_eval_jointv4.log
for U in pjointv4; do
  RR=data/rerank_llm_ancf_bgeft2; [ $U != none ] && RR=${RR}_pjointv4; RR=${RR}_aug_oof.json
  for V in plain para; do
    if [ $V = plain ]; then RL=data/rel_val_llm_ancf.json; TX=data/text_val_bgeft2.json; TL=data/text_val_pjointv4.json; else RL=data/rel_val_para_llm_ancf.json; TX=data/text_val_bgeft2_para.json; TL=data/text_val_pjointv4_para.json; fi
    TT=""; [ $U != none ] && TT="--t2l $TL"
    echo "== oof fit t2l=$U val=$V (w=$W)" >> logs/oof_eval_jointv4.log
    uv run python predict.py --split val --rel $RL --text $TX --w $W --rerank $RR $TT --score 2>&1 | grep "COMMITTED\|Error" >> logs/oof_eval_jointv4.log
  done
done
rm -f eval_results_val.csv; echo JOINTV4_OOF_DONE >> logs/oof_eval_jointv4.log
