#!/bin/bash
# lever 6: push the standalone readout — second paraphrase set, longer joint training, multi-vector queries
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib
st() { echo "$(date +%H:%M:%S) $1" >> logs/lever6_status.log; }
res() { grep -h "\[link\] joint\|^QA .* all entities" logs/$1.log | sed "s/^/    /" >> logs/lever6_status.log; }
st "START lever 6"
st "R3 p_jointv4: joint 3 epochs, 4 vectors per question (isolates multi-vector vs p_joint 26.6/25.4)"
uv run python joint_train.py --tag p_jointv4 --extra data/para_train.json --nvec 4 > logs/p_jointv4.log 2>&1; st "R3 done"; res p_jointv4
st "R0 second paraphrase set of train (7B model, seed 2) ..."
uv run python paraphrase.py --split train --seed 2 --out data/para_train2.json > logs/para_train2.log 2>&1; st "R0 done: $(grep -c . data/para_train2.json) lines"
st "R1 p_joint10: joint 10 epochs, two paraphrase sets, 1 vector"
uv run python joint_train.py --tag p_joint10 --extra data/para_train.json --extra2 data/para_train2.json --epochs 10 > logs/p_joint10.log 2>&1; st "R1 done"; res p_joint10
st "R2 p_joint10v4: joint 10 epochs, two paraphrase sets, 4 vectors"
uv run python joint_train.py --tag p_joint10v4 --extra data/para_train.json --extra2 data/para_train2.json --epochs 10 --nvec 4 > logs/p_joint10v4.log 2>&1; st "R2 done"; res p_joint10v4
st "LEVER6_TRAIN_DONE"
