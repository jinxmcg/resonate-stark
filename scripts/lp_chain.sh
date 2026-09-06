#!/bin/bash
# lever 7: latent parser — train on train (+ paraphrases), parse val plain + paraphrased, retrieve with it (walk + model), compare
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/lp_status.log; }
st "LP train"
uv run python latent_parser.py --train --tag lp > logs/lp_train.log 2>&1; st "LP train exit $?"
grep "weak labels\|epoch\|anchor floor\|LP_TRAIN_DONE\|Error" logs/lp_train.log | sed "s/^/    /" >> logs/lp_status.log
st "LP predict val plain / para"
uv run python latent_parser.py --predict val --tag lp --out data/lparse_val.json > logs/lp_pred_val.log 2>&1
uv run python latent_parser.py --predict val --tag lp --queries data/para_val.json --out data/lparse_val_para.json > logs/lp_pred_val_para.log 2>&1
st "retrieve with the latent parser (base table, beta 30)"
uv run python retrieve.py --model $M --split val --beta 30 --lparse data/lparse_val.json --out data/rel_val_lp.json > logs/rel_val_lp.log 2>&1
uv run python retrieve.py --model $M --split val --beta 30 --queries data/para_val.json --lparse data/lparse_val_para.json --out data/rel_val_para_lp.json > logs/rel_val_para_lp.log 2>&1
st "retrieve with the latent parser + bge anchor fallback"
uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --lparse data/lparse_val.json --out data/rel_val_lpa.json > logs/rel_val_lpa.log 2>&1
uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --queries data/para_val.json --lparse data/lparse_val_para.json --out data/rel_val_para_lpa.json > logs/rel_val_para_lpa.log 2>&1
for f in rel_val_lp rel_val_para_lp rel_val_lpa rel_val_para_lpa; do echo "== $f" >> logs/lp_status.log; grep "^val:\|all queries\|Error" logs/$f.log | cut -c1-200 >> logs/lp_status.log; done
st "LP_CHAIN_DONE"
