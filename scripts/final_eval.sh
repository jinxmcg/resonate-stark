#!/bin/bash
# P2 final combined evaluation on val (plain + paraphrased): LLM-override retrieval, bge_ft2 text, best text-to-latent,
# reranker fit plain vs augmented, with vs without text-to-latent. Waits for every producer. Never touches test/human.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
until grep -q LLMTRAIN_DONE logs/llm_train_para.log 2>/dev/null && grep -q T2L_EXIT logs/t2lj.log 2>/dev/null && grep -q T2L_EXIT logs/t2l10.log 2>/dev/null && grep -q AUG_DONE logs/aug_para_augfit.log 2>/dev/null; do sleep 30; done
M=models/p_k12b4_50k.pt
uv run python retrieve.py --model $M --split train --beta 30 --anchor bge --dump-feats --llm-parse data/llmparse_train.json --llm-override-type --out data/rel_train_llm_ancf.json > logs/rel_train_llm.log 2>&1
uv run python retrieve.py --model $M --split train --beta 30 --anchor bge --dump-feats --queries data/para_train.json --llm-parse data/llmparse_train_para.json --llm-override-type --out data/rel_train_para_llm_ancf.json > logs/rel_train_para_llm.log 2>&1
cp data/rel_val_llmoverride_ancf.json data/rel_val_llm_ancf.json; cp data/rel_val_para_llmoverride_ancf.json data/rel_val_para_llm_ancf.json
# best text-to-latent by plain-val MRR (all entities)
BEST=$(python3 - <<'PY'
import re
best=None
for t in ("t2l","t2lj","t2l10"):
    try:
        for l in open(f"logs/{t}.log"):
            if l.startswith("VAL plain text-to-latent, all entities"):
                m=float(re.search(r"'mrr': ([0-9.]+)", l).group(1))
                if best is None or m>best[1]: best=(t,m)
    except FileNotFoundError: pass
print(best[0])
PY
)
echo "best t2l: $BEST" > logs/final_eval.log
uv run python t2l_rank.py --tag $BEST --split train --queries data/para_train.json --out data/text_train_${BEST}_para.json >> logs/final_eval.log 2>&1
cp data/text_val_${BEST}_para.json data/text_val_${BEST}_para.json 2>/dev/null
for FIT in plain aug; do for U in none $BEST; do
  X=""; [ $FIT = aug ] && X="--aug-rel-tag para_llm_ancf --aug-text-tag bgeft2_para"
  T=""; if [ $U != none ]; then T="--t2l-tag $U"; [ $FIT = aug ] && T="$T --aug-t2l-tag ${U}_para"; fi
  uv run python rerank.py --rel-tag llm_ancf --text-tag bgeft2 $X $T > logs/final_rerank_${FIT}_${U}.log 2>&1
  RR=data/rerank_llm_ancf_bgeft2; [ $U != none ] && RR=${RR}_$U; [ $FIT = aug ] && RR=${RR}_aug; RR=$RR.json
  for V in plain para; do
    if [ $V = plain ]; then RL=data/rel_val_llm_ancf.json; TX=data/text_val_bgeft2.json; TL=data/text_val_${U}.json; else RL=data/rel_val_para_llm_ancf.json; TX=data/text_val_bgeft2_para.json; TL=data/text_val_${U}_para.json; fi
    TT=""; [ $U != none ] && TT="--t2l $TL"
    echo "== fit=$FIT t2l=$U val=$V" >> logs/final_eval.log
    uv run python predict.py --split val --rel $RL --text $TX --w 0.5 --rerank $RR $TT --score 2>&1 | grep "COMMITTED\|Error" >> logs/final_eval.log
  done
done; done
rm -f eval_results_val.csv
echo FINAL_DONE >> logs/final_eval.log
