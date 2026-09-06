#!/bin/bash
# lever 8 follow-up: tune the RotatE control by held-out LINK MRR (not by QA), then the frozen projector on the best table
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib
st() { echo "$(date +%H:%M:%S) $1" >> logs/lever8_sweep_status.log; }
best=""; bestm=0
for cfg in "3e-4 12" "3e-3 12" "1e-3 6" "1e-3 24"; do set -- $cfg
  st "RotatE lr $1 gamma $2"; uv run python rotate_prime.py --lr $1 --gamma $2 --save models/rotate_lr$1_g$2.pt > logs/rotate_lr$1_g$2.log 2>&1
  m=$(grep "holdout" logs/rotate_lr$1_g$2.log | sed "s/.*mean //"); st "  link MRR $m"
  if python3 -c "import sys; sys.exit(0 if float('$m') > float('$bestm') else 1)"; then best="models/rotate_lr$1_g$2.pt"; bestm=$m; fi
done
st "best by link MRR: $best ($bestm) vs default 0.5233"
uv run python t2l_generic.py --table rotate --ckpt $best --readout dist --tag f_rotate_best_dist > logs/t2lg_f_rotate_best_dist.log 2>&1
uv run python t2l_generic.py --table rotate --ckpt $best --readout dot --tag f_rotate_best_dot > logs/t2lg_f_rotate_best_dot.log 2>&1
grep -h "^QA" logs/t2lg_f_rotate_best_*.log | sed "s/^/    /" >> logs/lever8_sweep_status.log
st "LEVER8_SWEEP_DONE"
