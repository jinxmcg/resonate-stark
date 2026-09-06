#!/bin/bash
# lever 8: RotatE control — same graph, same regime, identical projector, frozen and joint; ResonatE through the same script
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib
st() { echo "$(date +%H:%M:%S) $1" >> logs/lever8_status.log; }
st "RotatE training (50k steps)"
uv run python rotate_prime.py > logs/rotate.log 2>&1; grep "holdout\|ROTATE_DONE" logs/rotate.log | sed "s/^/    /" >> logs/lever8_status.log
for cfg in "resonate dot" "rotate dist" "rotate dot"; do set -- $cfg
  st "frozen projector: $1 $2"; uv run python t2l_generic.py --table $1 --readout $2 --tag f_$1_$2 > logs/t2lg_f_$1_$2.log 2>&1
  grep "\[link\]\|^QA" logs/t2lg_f_$1_$2.log | sed "s/^/    /" >> logs/lever8_status.log
done
for cfg in "resonate dot" "rotate dist"; do set -- $cfg
  st "joint: $1 $2"; uv run python t2l_generic.py --table $1 --readout $2 --joint --tag j_$1_$2 > logs/t2lg_j_$1_$2.log 2>&1
  grep "\[link\]\|^QA" logs/t2lg_j_$1_$2.log | sed "s/^/    /" >> logs/lever8_status.log
done
st "LEVER8_DONE"
