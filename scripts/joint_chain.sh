#!/bin/bash
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib
uv run python joint_train.py --eval-table models/t2lj.pt > logs/joint_evaltable_t2lj.log 2>&1
uv run python joint_train.py --tag p_joint --extra data/para_train.json > logs/joint.log 2>&1
uv run python retrieve.py --model models/p_joint.pt --split val --beta 30 --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --out data/rel_val_joint_tmp.json > logs/rel_val_jointtable.log 2>&1
uv run python retrieve.py --model models/p_joint.pt --split val --beta 0 --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --out data/rel_val_joint_b0_tmp.json > logs/rel_val_jointtable_b0.log 2>&1
echo JOINT_CHAIN_DONE >> logs/rel_val_jointtable_b0.log
