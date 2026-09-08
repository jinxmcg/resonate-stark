#!/bin/bash
# P13 (RTX 5090, vast 50261550): a medical-register proxy, P11's seed variance, and P11 vs P3 on
# four wordings. Val only. See PLAN_PRIME.md "P13".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p13_status.log; }
rm -f logs/p13_status.log; st "1. medical-register proxy (OpenBioLLM-8B, style terse2)"
$PY paraphrase.py --split val --style terse2 --model aaditya/Llama3-OpenBioLLM-8B --seed 3 --out data/para_val_terse_c.json > logs/p13_para.log 2>&1
if [ ! -f data/para_val_terse_c.json ]; then st "ABORT: proxy not generated — $(tail -1 logs/p13_para.log | cut -c1-140)"; exit 1; fi
$PY - <<'PY' >> logs/p13_status.log 2>&1
import json, sys
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
import numpy as np
names = json.load(open("data/names.json")); qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()
P = {}
for nm, path in (("A Qwen", "data/para_val_terse.json"), ("B Phi", "data/para_val_terse_b.json"), ("C OpenBioLLM", "data/para_val_terse_c.json")):
    S = json.load(open(path)); P[nm] = S; leak = n = 0; lens = []
    for i in idx:
        q, qid, ans, _ = qa[i]; t = S.get(str(int(qid)))
        if t is None: continue
        n += 1; lens.append(len(t.split())); tl = t.lower()
        if any(names[str(a)].strip().lower() in tl for a in ans if len(names[str(a)].strip()) >= 4): leak += 1
    print(f"    proxy {nm}: n={n} median words {np.median(lens):.0f}  contains a true answer name {leak} ({leak/n:.1%})")
ks = list(P["C OpenBioLLM"])
for other in ("A Qwen", "B Phi"):
    same = sum(1 for k in ks if P[other].get(k, "").strip().lower() == P["C OpenBioLLM"][k].strip().lower())
    print(f"    C identical to {other}: {same} ({same/len(ks):.1%})")
PY
st "2. P11 parser at seeds 1 and 2"
for S in 1 2; do
  $PY latent_parser.py --train --tag lp_p4t_s$S --labels data/lp_labels_terse.json --seed $S > logs/p13_train_s$S.log 2>&1
  if [ ! -f models/lp_p4t_s$S.pt ]; then st "ABORT: seed $S did not train — $(tail -1 logs/p13_train_s$S.log | cut -c1-120)"; exit 1; fi
  st "  seed $S trained"
done
st "3. parses, retrievals and scoring: P3 and three P11 seeds x four wordings"
RR3=data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json
RR11=data/rerank_p11_lp_ancf_bgeft2_pjoint_aug_oof.json
for W in plain terseA terseB terseC; do
  case $W in
    plain)  Q="";                                     TXF=data/text_val_bgeft2.json;        TXD=data/text_val_bgeft2_d384.json;        TL=data/text_val_pjoint.json ;;
    terseA) Q="--queries data/para_val_terse.json";   TXF=data/text_val_bgeft2_terse.json;  TXD=data/text_val_bgeft2_d384_terse.json;  TL=data/text_val_pjoint_terse.json ;;
    terseB) Q="--queries data/para_val_terse_b.json"; TXF=data/text_val_bgeft2_terseb.json; TXD=data/text_val_bgeft2_d384_terseb.json; TL=data/text_val_pjoint_terseb.json ;;
    terseC) Q="--queries data/para_val_terse_c.json"; TXF=data/text_val_bgeft2_tersec.json; TXD=data/text_val_bgeft2_d384_tersec.json; TL=data/text_val_pjoint_tersec.json ;;
  esac
  if [ $W = terseC ]; then      # the new wording needs its own text, projected text and t2l rankings
    $PY embed_text2.py --model bgeft2 --reuse-docs --splits val $Q --tag _tersec > logs/p13_txt.log 2>&1
    $PY embed_text2.py --model bgeft2 --docs data/doc_emb_bgeft2_p384.npy --proj data/docproj_bgeft2_384.npy --splits val $Q --tag _d384_tersec > logs/p13_txtd.log 2>&1
    $PY t2l_rank.py --tag p_joint_head --model models/p_joint.pt --split val $Q --out $TL > logs/p13_t2l.log 2>&1
    $PY latent_parser.py --predict val --tag lp_p3 $Q --out data/lparse_val_tersec.json > logs/p13_lp3_c.log 2>&1
    $PY retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse data/lparse_val_tersec.json --out data/rel_val_p13P3_$W.json > logs/p13_rel_P3_$W.log 2>&1
    $PY predict.py --split val --rel data/rel_val_p13P3_$W.json --text $TXF --t2l $TL --w 0.45 --rerank $RR3 --score 2>&1 | grep COMMITTED | sed "s/^/    P3        $W  /" >> logs/p13_status.log
  fi
  for S in 0 1 2; do
    if [ $S = 0 ]; then TAG=lp_p4t; else TAG=lp_p4t_s$S; fi
    LP=data/lparse_val_${TAG}_$W.json
    $PY latent_parser.py --predict val --tag $TAG $Q --out $LP > logs/p13_lp_${TAG}_$W.log 2>&1
    $PY retrieve.py --model $M --split val --beta 30 --anchor lp --lp-anchor models/$TAG.pt --dump-feats $Q --lparse $LP --out data/rel_val_p13_${TAG}_$W.json > logs/p13_rel_${TAG}_$W.log 2>&1
    $PY predict.py --split val --rel data/rel_val_p13_${TAG}_$W.json --text $TXD --t2l $TL --w 0.45 --rerank $RR11 --score 2>&1 | grep COMMITTED | sed "s/^/    P11 seed $S  $W  /" >> logs/p13_status.log
  done
  st "  $W done"
done
rm -f eval_results_val.csv
st "P13_DONE"
