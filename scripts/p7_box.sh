#!/bin/bash
# P7 step 1 fail-fast (rented RTX 5090, vast 50261550): can the pipeline drop the duplicate anchor
# encoder (lever A: --anchor bgeft2) and the second entity table (lever C: --model p_joint) without
# losing more than 1.0 Hit@1 on the relational path? 1,000 TRAIN questions, plain + terse.
# Reads the train split only. See PLAN_PRIME.md "P7".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p7_status.log; }
rm -f logs/p7_status.log; st "P7 step 1: four arms x two wordings, 1,000 train questions"
for ARM in 0 A C AC; do
  case $ARM in
    0)  M=models/p_k12b4_50k.pt; ANC=bge    ;;
    A)  M=models/p_k12b4_50k.pt; ANC=bgeft2 ;;
    C)  M=models/p_joint.pt;     ANC=bge    ;;
    AC) M=models/p_joint.pt;     ANC=bgeft2 ;;
  esac
  for W in plain terse; do
    if [ $W = plain ]; then Q=""; LP=data/lparse_train_full.json; else Q="--queries data/para_train_terse.json"; LP=data/lparse_train_terse.json; fi
    $PY retrieve.py --model $M --split train --limit 1000 --beta 30 --anchor $ANC $Q --lparse $LP \
        --out data/rel_train_p7_${ARM}_${W}.json > logs/p7_${ARM}_${W}.log 2>&1
    grep "covered\|all queries" logs/p7_${ARM}_${W}.log | tr '\n' ' ' | cut -c1-260 | sed "s/^/    arm $ARM ($ANC, $(basename $M .pt)) $W  /" >> logs/p7_status.log
    echo >> logs/p7_status.log
  done
  st "  arm $ARM done"
done
st "P7_STEP1_DONE"
