#!/bin/bash
# P6 fail-fast (rented RTX 5090, vast 50261550, torch 2.11.0+cu128): reverse-operator candidate
# features rev_raw / rev_nov in the reranker, out of fold, on 1,000 TRAIN questions, plain wording
# and the terse paraphrases. Reads the train split only. See PLAN_PRIME.md "P6".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p6_status.log; }
rm -f logs/p6_status.log; st "1. relational retrieval with --rev (1,000 train questions, plain + terse)"
$PY retrieve.py --model $M --split train --limit 1000 --beta 30 --anchor bge --lparse data/lparse_train_full.json \
   --dump-feats --rev --out data/rel_train_p6_plain.json > logs/p6_rel_plain.log 2>&1
grep "all queries" logs/p6_rel_plain.log | sed "s/^/    retrieval plain  /" >> logs/p6_status.log
st "  plain done"
$PY retrieve.py --model $M --split train --limit 1000 --beta 30 --anchor bge --queries data/para_train_terse.json \
   --lparse data/lparse_train_terse.json --dump-feats --rev --out data/rel_train_p6_terse.json > logs/p6_rel_terse.log 2>&1
grep "all queries" logs/p6_rel_terse.log | sed "s/^/    retrieval terse  /" >> logs/p6_status.log
st "  terse done"
st "2. self-check (rev_raw vs the table's own readout at the anchor)"
$PY rev_check.py --model $M --rel data/rel_train_p6_plain.json --split train --limit 1000 > logs/p6_check.log 2>&1
sed "s/^/    /" logs/p6_check.log | grep -v Warning >> logs/p6_status.log
st "3. out-of-fold reranker refits (base vs rev)"
$PY rev_oof.py --plain data/rel_train_p6_plain.json --terse data/rel_train_p6_terse.json --limit 1000 \
   --out-prefix data/p6oof > logs/p6_oof.log 2>&1
grep "out-of-fold weights\|wrote" logs/p6_oof.log | sed "s/^/    /" >> logs/p6_status.log
st "4. scoring through predict.py --score (official path)"
for W in plain terse; do
  R=data/rel_train_p6_${W}.json
  $PY predict.py --split train --limit 1000 --rel $R --text data/empty.json --w 1.0 --score 2>&1 | grep COMMITTED | sed "s/^/    no-reranker $W  /" >> logs/p6_status.log
done
for ARM in base rev; do for W in plain terse; do
  $PY predict.py --split train --limit 1000 --rel data/p6oof_${ARM}_${W}.json --text data/empty.json --w 1.0 --score 2>&1 | grep COMMITTED | sed "s/^/    oof-$ARM $W  /" >> logs/p6_status.log
done; done
rm -f eval_results_train.csv
st "P6_DONE"
