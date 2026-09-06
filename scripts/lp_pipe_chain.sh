#!/bin/bash
# lever 7b: latent parser inside the full pipeline (out-of-fold train parses -> reranker) + seeds for the joint model and the parser
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/lp_pipe_status.log; }
st "A: out-of-fold latent parsers (5 folds)"
for k in 0 1 2 3 4; do
  uv run python latent_parser.py --train --tag lp_f$k --fold 5:$k > logs/lp_f$k.log 2>&1
  uv run python latent_parser.py --predict train --tag lp_f$k --fold 5:$k --out data/oof/lparse_train_f$k.json > logs/lp_f${k}_pred.log 2>&1
  uv run python latent_parser.py --predict train --tag lp_f$k --fold 5:$k --queries data/para_train.json --out data/oof/lparse_train_f${k}_para.json > logs/lp_f${k}_pred_para.log 2>&1
  st "  fold $k done ($(grep -c . data/oof/lparse_train_f$k.json) bytes-lines)"
done
python3 - <<'PY'
import json
for suf in ("", "_para"):
    out = {}
    for k in range(5): out.update(json.load(open(f"data/oof/lparse_train_f{k}{suf}.json")))
    json.dump(out, open(f"data/lparse_train_oof{suf}.json", "w")); print("lparse_train_oof", suf, len(out))
PY
st "B: retrieval with the latent parser (+bge fallback, feats) on train (oof parses) and val"
uv run python retrieve.py --model $M --split train --beta 30 --anchor bge --dump-feats --lparse data/lparse_train_oof.json --out data/rel_train_lp_ancf.json > logs/rel_train_lp.log 2>&1
uv run python retrieve.py --model $M --split train --beta 30 --anchor bge --dump-feats --queries data/para_train.json --lparse data/lparse_train_oof_para.json --out data/rel_train_para_lp_ancf.json > logs/rel_train_para_lp.log 2>&1
uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats --lparse data/lparse_val.json --out data/rel_val_lp_ancf.json > logs/rel_val_lp_feats.log 2>&1
uv run python retrieve.py --model $M --split val --beta 30 --anchor bge --dump-feats --queries data/para_val.json --lparse data/lparse_val_para.json --out data/rel_val_para_lp_ancf.json > logs/rel_val_para_lp_feats.log 2>&1
st "C: reranker fit (oof features) + val scoring"
uv run python rerank.py --rel-tag lp_ancf --text-tag bgeft2 --train-text-tag bgeft2oof --aug-rel-tag para_lp_ancf --aug-text-tag bgeft2oof_para --t2l-tag pjoint --train-t2l-tag pjointoof --aug-t2l-tag pjointoof_para --save-tag _oof > logs/oof_rerank_lp.log 2>&1
grep "VAL reranked per-type" logs/oof_rerank_lp.log | sed "s/^/    /" >> logs/lp_pipe_status.log
for V in plain para; do
  if [ $V = plain ]; then RL=data/rel_val_lp_ancf.json; TX=data/text_val_bgeft2.json; TL=data/text_val_pjoint.json; else RL=data/rel_val_para_lp_ancf.json; TX=data/text_val_bgeft2_para.json; TL=data/text_val_pjoint_para.json; fi
  echo "== latent-parser pipeline val=$V" >> logs/lp_pipe_status.log
  uv run python predict.py --split val --rel $RL --text $TX --w 0.45 --rerank data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json --t2l $TL --score 2>&1 | grep "COMMITTED\|Error" >> logs/lp_pipe_status.log
done
rm -f eval_results_val.csv
st "LP_PIPE_DONE"
