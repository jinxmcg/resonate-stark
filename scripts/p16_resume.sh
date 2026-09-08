#!/bin/bash
# P16 RESUME (2026-09-08): step 4 only, and only the (register, seed) pairs the vast.ai
# credit outage left undone. plain s0-2 and terseA s0-1 were already scored into
# logs/p16_status.log; nothing is regenerated -- not the 8B dialect, not the parsers.
# NOTE: uses /venv/main/bin/python directly. `uv run --active` has no VIRTUAL_ENV over
# ssh, so it dies with ModuleNotFoundError while the st() markers still print "done";
# that failure mode reported four registers complete in three seconds.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
export VIRTUAL_ENV=/venv/main; export PATH=/venv/main/bin:$PATH; PY="/venv/main/bin/python"
M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/p16_status.log; }
RR11=data/rerank_p11_lp_ancf_bgeft2_pjoint_aug_oof.json
st "RESUMED after the credit outage: terseA s2, then terseB (the judge) and terseC"
run_one() {
  W=$1; S=$2
  case $W in
    terseA) Q="--queries data/para_val_terse.json";   TXD=data/text_val_bgeft2_d384_terse.json;  TL=data/text_val_pjoint_terse.json ;;
    terseB) Q="--queries data/para_val_terse_b.json"; TXD=data/text_val_bgeft2_d384_terseb.json; TL=data/text_val_pjoint_terseb.json ;;
    terseC) Q="--queries data/para_val_terse_c.json"; TXD=data/text_val_bgeft2_d384_tersec.json; TL=data/text_val_pjoint_tersec.json ;;
  esac
  LP=data/lparse_val_p16_s${S}_$W.json
  $PY latent_parser.py --predict val --tag lp_p16_s$S $Q --out $LP > logs/p16_lp_s${S}_$W.log 2>&1
  $PY retrieve.py --model $M --split val --beta 30 --anchor lp --lp-anchor models/lp_p16_s$S.pt --dump-feats $Q --lparse $LP --out data/rel_val_p16_s${S}_$W.json > logs/p16_rel_s${S}_$W.log 2>&1
  $PY predict.py --split val --rel data/rel_val_p16_s${S}_$W.json --text $TXD --t2l $TL --w 0.45 --rerank $RR11 --score 2>&1 | grep COMMITTED | sed "s/^/    P16 seed $S  $W  /" >> logs/p16_status.log
}
run_one terseA 2; st "  terseA done"
for S in 0 1 2; do run_one terseB $S; done; st "  terseB done (the judge)"
for S in 0 1 2; do run_one terseC $S; done; st "  terseC done"
rm -f eval_results_val.csv
st "P16_DONE (resumed)"
