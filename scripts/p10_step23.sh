#!/bin/bash
# P10 steps 2 and 3 (RTX 5090, vast 50261550). Reads train and val only. See PLAN_PRIME.md "P10".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p10_status.log; }
rm -f logs/p10_status.log; st "step 2a: inputs for the terse wording"
$PY latent_parser.py --predict val --tag lp_p3 --queries data/para_val_terse.json --out data/lparse_val_terse.json > logs/p10_lp_terse.log 2>&1
$PY embed_text2.py --model bgeft2 --reuse-docs --splits val --queries data/para_val_terse.json --tag _terse > logs/p10_txt_terse.log 2>&1
$PY embed_text2.py --model bgeft2 --docs data/doc_emb_bgeft2_p384.npy --proj data/docproj_bgeft2_384.npy --splits val --queries data/para_val_terse.json --tag _d384_terse > logs/p10_txtd_terse.log 2>&1
$PY t2l_rank.py --tag p_joint_head --model models/p_joint.pt --split val --queries data/para_val_terse.json --out data/text_val_pjoint_terse.json > logs/p10_t2l_terse.log 2>&1
$PY llm_parse.py --split val --queries data/para_val_terse.json --tag val_terse > logs/p10_llm_terse.log 2>&1
grep -h "text-only" logs/p10_txt_terse.log logs/p10_txtd_terse.log | sed "s/^/    /" >> logs/p10_status.log
grep -h "LLMPARSE_DONE" logs/p10_llm_terse.log | sed "s/^/    /" >> logs/p10_status.log
st "step 2b: three candidates x three wordings"
for W in plain para terse; do
  case $W in
    plain) Q="";                                   LP=data/lparse_val.json;       LLMP=data/llmparse_val.json;       TX=data/text_val_bgeft2.json;       TXD=data/text_val_bgeft2_d384.json;       TL=data/text_val_pjoint.json ;;
    para)  Q="--queries data/para_val.json";       LP=data/lparse_val_para.json;  LLMP=data/llmparse_val_para.json;  TX=data/text_val_bgeft2_para.json;  TXD=data/text_val_bgeft2_d384_para.json;  TL=data/text_val_pjoint_para.json ;;
    terse) Q="--queries data/para_val_terse.json"; LP=data/lparse_val_terse.json; LLMP=data/llmparse_val_terse.json; TX=data/text_val_bgeft2_terse.json; TXD=data/text_val_bgeft2_d384_terse.json; TL=data/text_val_pjoint_terse.json ;;
  esac
  $PY retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse $LP --out data/rel_val_p10P3_$W.json > logs/p10_rel_P3_$W.log 2>&1
  $PY predict.py --split val --rel data/rel_val_p10P3_$W.json --text $TX --t2l $TL --w 0.45 --rerank data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    P3   $W  /" >> logs/p10_status.log
  $PY retrieve.py --model $M --split val --beta 30 --anchor lp --dump-feats $Q --lparse $LP --out data/rel_val_p10P8_$W.json > logs/p10_rel_P8_$W.log 2>&1
  $PY predict.py --split val --rel data/rel_val_p10P8_$W.json --text $TXD --t2l $TL --w 0.45 --rerank data/rerank_p8A_lp_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    P8   $W  /" >> logs/p10_status.log
  $PY retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --llm-parse $LLMP --llm-override-type --out data/rel_val_p10LLM_$W.json > logs/p10_rel_LLM_$W.log 2>&1
  $PY predict.py --split val --rel data/rel_val_p10LLM_$W.json --text $TX --t2l $TL --w 0.45 --rerank data/rerank_llm_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    LLM  $W  /" >> logs/p10_status.log
  st "  $W done"
done
st "step 3: reverse features on the collision subset"
$PY retrieve.py --model $M --split train --beta 30 --anchor lp --dump-feats --rev --lparse data/lparse_train_oof.json --out data/rel_train_rev.json > logs/p10_rev_train.log 2>&1
$PY retrieve.py --model $M --split train --beta 30 --anchor lp --dump-feats --rev --queries data/para_train.json --lparse data/lparse_train_oof_para.json --out data/rel_train_para_rev.json > logs/p10_rev_train_para.log 2>&1
$PY retrieve.py --model $M --split val --beta 30 --anchor lp --dump-feats --rev --lparse data/lparse_val.json --out data/rel_val_rev.json > logs/p10_rev_val.log 2>&1
$PY retrieve.py --model $M --split val --beta 30 --anchor lp --dump-feats --rev --queries data/para_val_terse.json --lparse data/lparse_val_terse.json --out data/rel_val_terse_rev.json > logs/p10_rev_val_terse.log 2>&1
$PY p10_collide.py --train data/rel_train_rev.json --train-para data/rel_train_para_rev.json --val-plain data/rel_val_rev.json --val-terse data/rel_val_terse_rev.json --out-prefix data/p10c > logs/p10_collide.log 2>&1
grep -E "collision questions|weights" logs/p10_collide.log | sed "s/^/    /" >> logs/p10_status.log
for ARM in base rev; do for W in plain terse; do for S in collision rest; do
  $PY predict.py --split val --rel data/p10c_${ARM}_${W}.json --text data/empty.json --w 1.0 --qids data/p10c_${S}_qids.json --score 2>&1 | grep COMMITTED | sed "s/^/    $ARM $W $S  /" >> logs/p10_status.log
done; done; done
rm -f eval_results_val.csv
st "P10_DONE"
