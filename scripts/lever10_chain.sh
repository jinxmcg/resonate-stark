#!/bin/bash
# lever 10, fail-fast: (1) sample conjunctions; (2) 1000-step AND training, synthetic exact-AND val -> GATE; (3) STaRK val model-only only if the gate passes
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib
st() { echo "$(date +%H:%M:%S) $1" >> logs/lever10_status.log; }
st "1. sampling conjunctions"
uv run python conj_sample.py --n-train 30000 --n-val 1500 --max-seconds 480 > logs/conj_sample.log 2>&1; grep "CONJ_DONE" logs/conj_sample.log | sed "s/^/    /" >> logs/lever10_status.log
st "2. AND training, 1000 steps (typed synthetic eval)"
uv run python and_train.py --steps 1000 --tag p_and --typed > logs/and_train.log 2>&1
grep "\[link\]\|^CONJ" logs/and_train.log | sed "s/^/    /" >> logs/lever10_status.log
G=$(python3 - <<'PY'
import re
base=trained=None
for l in open("logs/and_train.log"):
    m=re.match(r"CONJ (base|trained) softmin\s*: (.*)", l)
    if m: 
        d=eval(m.group(2)); 
        if m.group(1)=="base": base=d["hit1"]
        else: trained=d["hit1"]
print("PASS" if trained is not None and trained >= 0.5 and trained >= base + 0.2 else f"FAIL base={base} trained={trained}")
PY
)
st "GATE (synthetic softmin Hit@1 >= 0.5 and >= base + 0.2): $G"
if [ "$G" = "PASS" ]; then
  st "3. STaRK val, model only, AND-trained table"
  M=models/p_and.pt
  for agg in sum min softmin; do
    uv run python retrieve.py --model $M --split val --beta 0 --anchor bge --llm-parse data/llmparse_val.json --llm-override-type --agg $agg --agg-p 1 --out data/rel_tmp_and_$agg.json > logs/l10_$agg.log 2>&1
    echo "== STaRK val model-only, AND-trained table, $agg" >> logs/lever10_status.log; grep "all queries\|>= 2 names" logs/l10_$agg.log | sed "s/^/    /" >> logs/lever10_status.log
  done
fi
st "LEVER10_DONE"
