#!/bin/bash
# P16 (box 50270859): does the parser learn a REGISTER or a DIALECT? Train on two shorthand
# generators, judge on the third (Phi) that neither produced. Train and val only.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p16_status.log; }
rm -f logs/p16_status.log; st "1. second training dialect (OpenBioLLM terse on train)"
$PY paraphrase.py --split train --style terse2 --model aaditya/Llama3-OpenBioLLM-8B --seed 3 --out data/para_train_terse_c.json > logs/p16_para.log 2>&1
if [ ! -f data/para_train_terse_c.json ]; then st "ABORT: generation failed — $(tail -1 logs/p16_para.log | cut -c1-140)"; exit 1; fi
grep PARA_DONE logs/p16_para.log | sed "s/^/    /" >> logs/p16_status.log
st "2. labels: plain + natural + Qwen-terse + OpenBioLLM-terse"
$PY - <<'PY' >> logs/p16_status.log 2>&1
import json, sys
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
L = json.load(open("data/lp_labels_terse.json")); data = L["data"]; pos = L["pos"]
C = json.load(open("data/para_train_terse_c.json"))
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
qid_of = {p: int(qa[i][1]) for p, i in enumerate(tr)}
plain = {}
for row, p in zip(data, pos):
    if p not in plain: plain[p] = row
add = 0
for p, row in plain.items():
    t = C.get(str(qid_of[p]))
    if t is None: continue
    data.append([t, row[1], row[2], row[3]]); pos.append(p); add += 1
json.dump({"data": data, "pos": pos}, open("data/lp_labels_terse2.json", "w"))
print(f"    labels {len(data)} rows (+{add} OpenBioLLM-terse), {len(set(pos))} questions")
PY
st "3. two-dialect parser at three seeds"
for S in 0 1 2; do
  $PY latent_parser.py --train --tag lp_p16_s$S --labels data/lp_labels_terse2.json --seed $S > logs/p16_train_s$S.log 2>&1
  if [ ! -f models/lp_p16_s$S.pt ]; then st "ABORT: seed $S failed — $(tail -1 logs/p16_train_s$S.log | cut -c1-120)"; exit 1; fi
  st "  seed $S trained"
done
st "4. scoring: plain and the three registers (terse B is the judge)"
RR11=data/rerank_p11_lp_ancf_bgeft2_pjoint_aug_oof.json
for W in plain terseA terseB terseC; do
  case $W in
    plain)  Q="";                                     TXD=data/text_val_bgeft2_d384.json;        TL=data/text_val_pjoint.json ;;
    terseA) Q="--queries data/para_val_terse.json";   TXD=data/text_val_bgeft2_d384_terse.json;  TL=data/text_val_pjoint_terse.json ;;
    terseB) Q="--queries data/para_val_terse_b.json"; TXD=data/text_val_bgeft2_d384_terseb.json; TL=data/text_val_pjoint_terseb.json ;;
    terseC) Q="--queries data/para_val_terse_c.json"; TXD=data/text_val_bgeft2_d384_tersec.json; TL=data/text_val_pjoint_tersec.json ;;
  esac
  for S in 0 1 2; do
    LP=data/lparse_val_p16_s${S}_$W.json
    $PY latent_parser.py --predict val --tag lp_p16_s$S $Q --out $LP > logs/p16_lp_s${S}_$W.log 2>&1
    $PY retrieve.py --model $M --split val --beta 30 --anchor lp --lp-anchor models/lp_p16_s$S.pt --dump-feats $Q --lparse $LP --out data/rel_val_p16_s${S}_$W.json > logs/p16_rel_s${S}_$W.log 2>&1
    $PY predict.py --split val --rel data/rel_val_p16_s${S}_$W.json --text $TXD --t2l $TL --w 0.45 --rerank $RR11 --score 2>&1 | grep COMMITTED | sed "s/^/    P16 seed $S  $W  /" >> logs/p16_status.log
  done
  st "  $W done"
done
rm -f eval_results_val.csv
st "P16_DONE"
