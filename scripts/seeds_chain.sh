#!/bin/bash
# seeds: joint model (3 ep, 1 vector) seeds 1,2 -> standalone QA + link MRR; latent parser seeds 1,2 -> relational path (val plain + para)
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/seeds_status.log; }
until [ -f data/lp_labels.json ]; do sleep 20; done
for s in 1 2; do
  st "joint seed $s"; uv run python joint_train.py --tag p_joint_s$s --extra data/para_train.json --seed $s --eval-every 0 > logs/p_joint_s$s.log 2>&1
  grep "\[link\] joint\|^QA .* all entities" logs/p_joint_s$s.log | sed "s/^/    /" >> logs/seeds_status.log
done
for s in 1 2; do
  st "latent parser seed $s"; uv run python latent_parser.py --train --tag lp_s$s --seed $s > logs/lp_s$s.log 2>&1
  uv run python latent_parser.py --predict val --tag lp_s$s --out data/lparse_val_s$s.json > /dev/null 2>&1
  uv run python latent_parser.py --predict val --tag lp_s$s --queries data/para_val.json --out data/lparse_val_para_s$s.json > /dev/null 2>&1
  uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --lparse data/lparse_val_s$s.json --out data/rel_tmp.json 2>&1 | grep "all queries" | sed "s/^/    plain: /" >> logs/seeds_status.log
  uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --queries data/para_val.json --lparse data/lparse_val_para_s$s.json --out data/rel_tmp.json 2>&1 | grep "all queries" | sed "s/^/    para : /" >> logs/seeds_status.log
done
st "SEEDS_DONE"
