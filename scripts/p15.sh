#!/bin/bash
# P15 (box 50270859): the shared trunk with the text loss upweighted, screened as P9 was. Train only.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p15_status.log; }
rm -f logs/p15_status.log
$PY - <<'PY' >> logs/p15_status.log 2>&1
import json, sys
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
from metrics import stark_metrics, summarize
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
fold0 = [i for p, i in enumerate(tr) if p % 5 == 0]
for name, path in (("dedicated bgeft2_f0", "data/oof/bgeft2_f0.json"), ("dedicated p_joint_f0", "data/oof/p_joint_f0.json")):
    R = json.load(open(path)); rows = [stark_metrics(R[str(int(qa[i][1]))], qa[i][2]) for i in fold0 if str(int(qa[i][1])) in R]
    print(f"    {name} (n={len(rows)}):", {k: round(v, 4) for k, v in summarize(rows).items()})
PY
for W in 3 6; do
  st "trunk with the text loss x$W (fold 0 held out)"
  $PY shared_trunk.py --train --tag st_w$W --fold 5:0 --w-text $W > logs/p15_train_w$W.log 2>&1
  if [ ! -f models/st_w$W.pt ]; then st "ABORT: w$W did not train — $(tail -1 logs/p15_train_w$W.log | cut -c1-120)"; exit 1; fi
  grep -E "anchor floor|saved" logs/p15_train_w$W.log | sed "s/^/    w$W /" >> logs/p15_status.log
  $PY shared_trunk.py --predict-text train --tag st_w$W --fold 5:0 --out data/text_train_st_w$W.json > logs/p15_text_w$W.log 2>&1
  grep "st_w$W train" logs/p15_text_w$W.log | sed "s/^/    w$W TEXT   /" >> logs/p15_status.log
  $PY shared_trunk.py --predict-t2l train --tag st_w$W --fold 5:0 --out data/t2l_train_st_w$W.json > logs/p15_t2l_w$W.log 2>&1
  grep "st_w$W train" logs/p15_t2l_w$W.log | sed "s/^/    w$W t2l(info) /" >> logs/p15_status.log
  $PY shared_trunk.py --predict-parse train --tag st_w$W --fold 5:0 --out data/lparse_train_st_w$W.json > logs/p15_parse_w$W.log 2>&1
  $PY retrieve.py --model models/p_k12b4_50k.pt --split train --fold 5:0 --beta 30 --anchor bge --lparse data/lparse_train_st_w$W.json --out /tmp/rel_p15_w$W.json > logs/p15_rel_w$W.log 2>&1
  grep "all queries" logs/p15_rel_w$W.log | sed "s/^/    w$W PARSER /" >> logs/p15_status.log
  st "  w$W done"
done
$PY retrieve.py --model models/p_k12b4_50k.pt --split train --fold 5:0 --beta 30 --anchor bge --lparse data/oof/lparse_train_f0.json --out /tmp/rel_p15_ded.json > logs/p15_rel_ded.log 2>&1
grep "all queries" logs/p15_rel_ded.log | sed "s/^/    dedicated PARSER /" >> logs/p15_status.log
st "P15_DONE"
