#!/bin/bash
# P3 committed read (the third read of the test splits overall) with the no-LLM pipeline. Dry run on val first, then test, test-0.1, human_generated_eval.
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
RR=data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json; W=0.45
st() { echo "$(date +%H:%M:%S) $1" >> logs/committed_p3_status.log; }
run_split() {
  S=$1; st "split $S: parse"
  uv run python latent_parser.py --predict $S --tag lp --out data/lparse_$S.json > logs/cr_lp_$S.log 2>&1 || { st "parse FAILED $S"; return 1; }
  st "split $S: retrieve"
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats --lparse data/lparse_$S.json --out data/rel_${S}_lp_ancf.json > logs/cr_rel_$S.log 2>&1 || { st "retrieve FAILED $S"; return 1; }
  st "split $S: text"
  uv run python embed_text2.py --model bgeft2 --reuse-docs --splits $S > logs/cr_text_$S.log 2>&1 || { st "text FAILED $S"; return 1; }
  st "split $S: text-to-latent"
  uv run python t2l_rank.py --tag p_joint_head --model models/p_joint.pt --split $S --out data/text_${S}_pjoint.json > logs/cr_t2l_$S.log 2>&1 || { st "t2l FAILED $S"; return 1; }
  st "split $S: predict (the read)"
  uv run python predict.py --split $S --rel data/rel_${S}_lp_ancf.json --text data/text_${S}_bgeft2.json --t2l data/text_${S}_pjoint.json --w $W --rerank $RR --score > logs/cr_predict_$S.log 2>&1
  grep "wrote\|COMMITTED\|Error" logs/cr_predict_$S.log | sed "s/^/    /" >> logs/committed_p3_status.log
}
st "DRY RUN on val"
run_split val
grep -q "hit1': 0.42" logs/cr_predict_val.log || { st "DRY RUN MISMATCH — stopping before any test read"; exit 1; }
rm -f eval_results_val.csv
mkdir -p results_p3
for S in test test-0.1 human_generated_eval; do run_split $S; mv eval_results_$S.csv results_p3/ 2>/dev/null; done
st "COMMITTED_P3_DONE"
