#!/bin/bash
# P11 (RTX 5090, vast 50261550): the latent parser trained on terse phrasing as a third augmentation.
# Runs after P10. Reads train and val only. See PLAN_PRIME.md "P11".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p11_status.log; }
rm -f logs/p11_status.log; st "1. weak labels + terse rows"
$PY - <<'PY' >> logs/p11_status.log 2>&1
import json, sys
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
L = json.load(open("data/lp_labels.json")); data = L["data"]; pos = L["pos"]
T = json.load(open("data/para_train_terse.json"))
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
qid_of = {p: int(qa[i][1]) for p, i in enumerate(tr)}
plain = {}                                            # the plain row of each train position carries that question's labels
for row, p in zip(data, pos):
    if p not in plain: plain[p] = row
add = 0
for p, row in plain.items():
    t = T.get(str(qid_of[p]))
    if t is None: continue
    data.append([t, row[1], row[2], row[3]]); pos.append(p); add += 1
json.dump({"data": data, "pos": pos}, open("data/lp_labels_terse.json", "w"))
print(f"    labels {len(data)} rows (+{add} terse), {len(set(pos))} questions")
PY
st "2. parser trained on plain + natural + terse (full, then 5 folds)"
$PY latent_parser.py --train --tag lp_p4t --labels data/lp_labels_terse.json > logs/p11_train.log 2>&1
grep -E "weak labels|anchor|floor" logs/p11_train.log | tail -2 | sed "s/^/    /" >> logs/p11_status.log
if [ ! -f models/lp_p4t.pt ]; then st "ABORT: the parser did not train — $(tail -1 logs/p11_train.log | cut -c1-120)"; exit 1; fi
for k in 0 1 2 3 4; do
  $PY latent_parser.py --train --tag lp_p4t_f$k --fold 5:$k --labels data/lp_labels_terse.json > logs/p11_f$k.log 2>&1
  $PY latent_parser.py --predict train --tag lp_p4t_f$k --fold 5:$k --out data/oof/lparse_train_p4t_f$k.json > logs/p11_f${k}_p.log 2>&1
  $PY latent_parser.py --predict train --tag lp_p4t_f$k --fold 5:$k --queries data/para_train.json --out data/oof/lparse_train_p4t_f${k}_para.json > logs/p11_f${k}_pp.log 2>&1
  if [ ! -f models/lp_p4t_f$k.pt ]; then st "ABORT: fold $k did not train — $(tail -1 logs/p11_f$k.log | cut -c1-120)"; exit 1; fi
  st "  fold $k done"
done
$PY - <<'PY'
import json
for suf in ("", "_para"):
    out = {}
    for k in range(5): out.update(json.load(open(f"data/oof/lparse_train_p4t_f{k}{suf}.json")))
    json.dump(out, open(f"data/lparse_train_p4t_oof{suf}.json", "w")); print("p4t oof", suf, len(out))
PY
st "3. val parses (three wordings) and retrievals"
$PY latent_parser.py --predict val --tag lp_p4t --out data/lparse_val_p4t.json > logs/p11_val.log 2>&1
$PY latent_parser.py --predict val --tag lp_p4t --queries data/para_val.json --out data/lparse_val_p4t_para.json > logs/p11_val_para.log 2>&1
$PY latent_parser.py --predict val --tag lp_p4t --queries data/para_val_terse.json --out data/lparse_val_p4t_terse.json > logs/p11_val_terse.log 2>&1
T=p11_lp_ancf
$PY retrieve.py --model $M --split train --beta 30 --anchor lp --lp-anchor models/lp_p4t.pt --dump-feats --lparse data/lparse_train_p4t_oof.json --out data/rel_train_${T}.json > logs/p11_rel_train.log 2>&1
$PY retrieve.py --model $M --split train --beta 30 --anchor lp --lp-anchor models/lp_p4t.pt --dump-feats --queries data/para_train.json --lparse data/lparse_train_p4t_oof_para.json --out data/rel_train_para_${T}.json > logs/p11_rel_train_para.log 2>&1
for W in plain para terse; do
  case $W in
    plain) Q=""; LP=data/lparse_val_p4t.json ;;
    para)  Q="--queries data/para_val.json"; LP=data/lparse_val_p4t_para.json ;;
    terse) Q="--queries data/para_val_terse.json"; LP=data/lparse_val_p4t_terse.json ;;
  esac
  OUT=data/rel_val_${T}.json; [ $W != plain ] && OUT=data/rel_val_${W}_${T}.json
  $PY retrieve.py --model $M --split val --beta 30 --anchor lp --lp-anchor models/lp_p4t.pt --dump-feats $Q --lparse $LP --out $OUT > logs/p11_rel_val_$W.log 2>&1
  grep "all queries" logs/p11_rel_val_$W.log | sed "s/^/    relational val $W  /" >> logs/p11_status.log
done
st "4. reranker refit and val scoring (three wordings)"
$PY rerank.py --rel-tag $T --text-tag bgeft2 --train-text-tag bgeft2oof --aug-rel-tag para_${T} --aug-text-tag bgeft2oof_para \
   --t2l-tag pjoint --train-t2l-tag pjointoof --aug-t2l-tag pjointoof_para --save-tag _oof > logs/p11_rerank.log 2>&1
grep "VAL reranked" logs/p11_rerank.log | sed "s/^/    /" >> logs/p11_status.log
RR=data/rerank_${T}_bgeft2_pjoint_aug_oof.json
for W in plain para terse; do
  case $W in
    plain) RL=data/rel_val_${T}.json;       TXD=data/text_val_bgeft2_d384.json;       TL=data/text_val_pjoint.json ;;
    para)  RL=data/rel_val_para_${T}.json;  TXD=data/text_val_bgeft2_d384_para.json;  TL=data/text_val_pjoint_para.json ;;
    terse) RL=data/rel_val_terse_${T}.json; TXD=data/text_val_bgeft2_d384_terse.json; TL=data/text_val_pjoint_terse.json ;;
  esac
  $PY predict.py --split val --rel $RL --text $TXD --t2l $TL --w 0.45 --rerank $RR --score 2>&1 | grep COMMITTED | sed "s/^/    P11  $W  /" >> logs/p11_status.log
done
rm -f eval_results_val.csv
st "P11_DONE"
