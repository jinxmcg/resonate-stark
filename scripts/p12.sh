#!/bin/bash
# P12 (RTX 5090, vast 50261550): a held-out evaluation style — different model family AND different
# prompt — then P3, P8, P11 and the Qwen-7B pipeline re-scored on it. Val only. PLAN_PRIME.md "P12".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p12_status.log; }
rm -f logs/p12_status.log; st "1. held-out proxy: Phi-3.5-mini-instruct, style terse2"
$PY paraphrase.py --split val --style terse2 --model microsoft/Phi-3.5-mini-instruct \
    --seed 3 --out data/para_val_terse_b.json > logs/p12_para.log 2>&1
if [ ! -f data/para_val_terse_b.json ]; then st "ABORT: proxy not generated — $(tail -1 logs/p12_para.log | cut -c1-140)"; exit 1; fi
grep PARA_DONE logs/p12_para.log | sed "s/^/    /" >> logs/p12_status.log
st "2. quality check (length, answer-name rate; plain baseline 4.6%)"
$PY - <<'PY' >> logs/p12_status.log 2>&1
import json, sys
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
import numpy as np
names = json.load(open("data/names.json")); qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()
for nm, path in (("terse A (Qwen)", "data/para_val_terse.json"), ("terse B (Phi)", "data/para_val_terse_b.json")):
    S = json.load(open(path)); leak = n = 0; lens = []
    for i in idx:
        q, qid, ans, _ = qa[i]; t = S.get(str(int(qid)))
        if t is None: continue
        n += 1; lens.append(len(t.split())); tl = t.lower()
        if any(names[str(a)].strip().lower() in tl for a in ans if len(names[str(a)].strip()) >= 4): leak += 1
    print(f"    {nm}: n={n} median words {np.median(lens):.0f} mean {np.mean(lens):.1f}  contains a true answer name {leak} ({leak/n:.1%})")
A = json.load(open("data/para_val_terse.json")); B = json.load(open("data/para_val_terse_b.json"))
same = sum(1 for k in B if A.get(k, "").strip().lower() == B[k].strip().lower())
print(f"    identical rewrites between the two proxies: {same} ({same/len(B):.1%}) — a low number is the point")
PY
st "3. inputs for the held-out wording"
$PY latent_parser.py --predict val --tag lp_p3  --queries data/para_val_terse_b.json --out data/lparse_val_terseb.json > logs/p12_lp.log 2>&1
$PY latent_parser.py --predict val --tag lp_p4t --queries data/para_val_terse_b.json --out data/lparse_val_p4t_terseb.json > logs/p12_lp4t.log 2>&1
$PY embed_text2.py --model bgeft2 --reuse-docs --splits val --queries data/para_val_terse_b.json --tag _terseb > logs/p12_txt.log 2>&1
$PY embed_text2.py --model bgeft2 --docs data/doc_emb_bgeft2_p384.npy --proj data/docproj_bgeft2_384.npy --splits val --queries data/para_val_terse_b.json --tag _d384_terseb > logs/p12_txtd.log 2>&1
$PY t2l_rank.py --tag p_joint_head --model models/p_joint.pt --split val --queries data/para_val_terse_b.json --out data/text_val_pjoint_terseb.json > logs/p12_t2l.log 2>&1
$PY llm_parse.py --split val --queries data/para_val_terse_b.json --tag val_terseb > logs/p12_llm.log 2>&1
grep -h "text-only" logs/p12_txt.log logs/p12_txtd.log | sed "s/^/    /" >> logs/p12_status.log
st "4. four arms on the held-out style"
Q="--queries data/para_val_terse_b.json"; TL=data/text_val_pjoint_terseb.json
$PY retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats $Q --lparse data/lparse_val_terseb.json --out data/rel_val_p12P3.json > logs/p12_rel_P3.log 2>&1
$PY predict.py --split val --rel data/rel_val_p12P3.json --text data/text_val_bgeft2_terseb.json --t2l $TL --w 0.45 --rerank data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    P3   terseB  /" >> logs/p12_status.log
$PY retrieve.py --model $M --split val --beta 30 --anchor lp $Q --dump-feats --lparse data/lparse_val_terseb.json --out data/rel_val_p12P8.json > logs/p12_rel_P8.log 2>&1
$PY predict.py --split val --rel data/rel_val_p12P8.json --text data/text_val_bgeft2_d384_terseb.json --t2l $TL --w 0.45 --rerank data/rerank_p8A_lp_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    P8   terseB  /" >> logs/p12_status.log
$PY retrieve.py --model $M --split val --beta 30 --anchor lp --lp-anchor models/lp_p4t.pt $Q --dump-feats --lparse data/lparse_val_p4t_terseb.json --out data/rel_val_p12P11.json > logs/p12_rel_P11.log 2>&1
$PY predict.py --split val --rel data/rel_val_p12P11.json --text data/text_val_bgeft2_d384_terseb.json --t2l $TL --w 0.45 --rerank data/rerank_p11_lp_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    P11  terseB  /" >> logs/p12_status.log
$PY retrieve.py --model $M --split val --beta 30 --anchor bge $Q --dump-feats --llm-parse data/llmparse_val_terseb.json --llm-override-type --out data/rel_val_p12LLM.json > logs/p12_rel_LLM.log 2>&1
$PY predict.py --split val --rel data/rel_val_p12LLM.json --text data/text_val_bgeft2_terseb.json --t2l $TL --w 0.45 --rerank data/rerank_llm_ancf_bgeft2_pjoint_aug_oof.json --score 2>&1 | grep COMMITTED | sed "s/^/    LLM  terseB  /" >> logs/p12_status.log
rm -f eval_results_val.csv
st "P12_DONE"
