#!/bin/bash
# P4 committed read: P3 pipeline + reading selector. Dry run on val first (must reproduce 44.36 / 68.14 / 75.32 / 55.14),
# then test, test-0.1, human_generated_eval once each -> results_p4/. Reads only the split's question text; answers only in --score.
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
RR=data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json; W=0.45
st() { echo "$(date +%H:%M:%S) $1" >> logs/committed_p4_status.log; }
FLOOR=$(uv run python -c "import torch; print(torch.load('models/lp.pt', map_location='cpu', weights_only=False)['sim_floor'])" 2>/dev/null | tail -1); LOW=$(python3 -c "print(round(float('$FLOOR')*0.8, 3))")
run_split() {
  S=$1; T=$2; st "split $S: parses (three readings)"
  uv run python latent_parser.py --predict $S --tag lp --out data/lpB_${T}_r1.json > logs/p4_parse_${S}_r1.log 2>&1 || { st "parse FAILED $S"; return 1; }
  uv run python latent_parser.py --predict $S --tag lp --sim-floor $LOW --out data/lpB_${T}_r3.json > logs/p4_parse_${S}_r3.log 2>&1
  uv run python latent_parser.py --predict $S --tag lp --type-rank 2 --out data/lpB_${T}_r4.json > logs/p4_parse_${S}_r4.log 2>&1
  st "split $S: text + readout"
  uv run python embed_text2.py --model bgeft2 --reuse-docs --splits $S > logs/p4_text_$S.log 2>&1 || { st "text FAILED $S"; return 1; }
  uv run python t2l_rank.py --tag p_joint_head --model models/p_joint.pt --split $S --out data/text_${T}_pjoint.json > logs/p4_t2l_$S.log 2>&1 || { st "t2l FAILED $S"; return 1; }
  st "split $S: four retrievals"
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats --lparse data/lpB_${T}_r1.json --out data/relB_${T}_r1.json > logs/p4_rel_${S}_r1.log 2>&1
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats --out data/relB_${T}_r2.json > logs/p4_rel_${S}_r2.log 2>&1
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats --lparse data/lpB_${T}_r3.json --out data/relB_${T}_r3.json > logs/p4_rel_${S}_r3.log 2>&1
  uv run python retrieve.py --model $M --split $S --beta 30 --anchor bge --dump-feats --lparse data/lpB_${T}_r4.json --out data/relB_${T}_r4.json > logs/p4_rel_${S}_r4.log 2>&1
  st "split $S: four scorings (no metrics) + selection"
  mkdir -p results_b
  for r in r1 r2 r3 r4; do
    uv run python predict.py --split $S --rel data/relB_${T}_$r.json --text data/text_${T}_bgeft2.json --t2l data/text_${T}_pjoint.json --w $W --rerank $RR --scores-out data/scB_${T}_$r.json > logs/p4_predict_${S}_$r.log 2>&1
    mv eval_results_$S.csv results_b/${T}_$r.csv
  done
  if [ "$S" = "val" ]; then uv run python select_readings.py --split val --tag val --score > logs/p4_select_val.log 2>&1; grep "COMMITTED\|wrote" logs/p4_select_val.log | sed "s/^/    /" >> logs/committed_p4_status.log
  else uv run python select_readings.py --split $S --tag $T --score > logs/p4_select_$S.log 2>&1; grep "COMMITTED\|wrote" logs/p4_select_$S.log | sed "s/^/    /" >> logs/committed_p4_status.log; fi
}
st "DRY RUN on val"
run_split val val
grep -q "hit1': 0.4436" logs/p4_select_val.log || { st "DRY RUN MISMATCH — stopping before any test read"; exit 1; }
rm -f eval_results_val.csv; mkdir -p results_p4; [ -n "$VAL_ONLY" ] && { st "VAL_ONLY_DONE"; exit 0; }
for S in test test-0.1 human_generated_eval; do run_split $S $S; mv eval_results_$S.csv results_p4/ 2>/dev/null; done
st "COMMITTED_P4_DONE"
