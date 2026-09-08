#!/bin/bash
# P9 step 1 SCREEN (RTX 5090, vast 50261550): one shared trunk trained without fold 0, each head
# compared on fold 0 against the DEDICATED model that excluded the same fold. Train split only.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p9_status.log; }
rm -f logs/p9_status.log; st "train the shared trunk without fold 0 (3 epochs)"
$PY shared_trunk.py --train --tag st_f0 --fold 5:0 > logs/p9_train.log 2>&1
grep -E "^rows|anchor floor|saved" logs/p9_train.log | sed "s/^/    /" >> logs/p9_status.log
st "head 1/3: text ranker on fold 0"
$PY shared_trunk.py --predict-text train --tag st_f0 --fold 5:0 --out data/text_train_st_f0.json > logs/p9_text.log 2>&1
grep "st_f0 train" logs/p9_text.log | sed "s/^/    shared   /" >> logs/p9_status.log
st "head 2/3: text-to-latent on fold 0"
$PY shared_trunk.py --predict-t2l train --tag st_f0 --fold 5:0 --out data/t2l_train_st_f0.json > logs/p9_t2l.log 2>&1
grep "st_f0 train" logs/p9_t2l.log | sed "s/^/    shared   /" >> logs/p9_status.log
$PY - <<'PY' >> logs/p9_status.log 2>&1
import json, sys
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
from metrics import stark_metrics, summarize
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
fold0 = [i for p, i in enumerate(tr) if p % 5 == 0]
for name, path in (("dedicated bgeft2_f0", "data/oof/bgeft2_f0.json"), ("dedicated p_joint_f0", "data/oof/p_joint_f0.json")):
    R = json.load(open(path)); rows = []
    for i in fold0:
        q, qid, ans, _ = qa[i]
        r = R.get(str(int(qid)))
        if r is None: continue
        rows.append(stark_metrics(r, ans))
    print(f"    {name} (n={len(rows)}):", {k: round(v, 4) for k, v in summarize(rows).items()})
PY
st "head 3/3: parser — relational path on fold 0, shared parse vs dedicated fold-0 parse"
$PY shared_trunk.py --predict-parse train --tag st_f0 --fold 5:0 --out data/lparse_train_st_f0.json > logs/p9_parse.log 2>&1
for P in st_f0 dedicated; do
  if [ $P = st_f0 ]; then LP=data/lparse_train_st_f0.json; else LP=data/oof/lparse_train_f0.json; fi
  $PY retrieve.py --model models/p_k12b4_50k.pt --split train --fold 5:0 --beta 30 --anchor bge --lparse $LP \
      --out /tmp/rel_p9_$P.json > logs/p9_rel_$P.log 2>&1
  grep "all queries" logs/p9_rel_$P.log | sed "s/^/    parse $P  /" >> logs/p9_status.log
done
st "P9_SCREEN_DONE"
