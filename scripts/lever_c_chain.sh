#!/bin/bash
# Lever C: cleaner + terse paraphrases -> embedder refit (bge_ft3) -> OOF text features -> reranker refit -> four readings re-scored -> selector data
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH PYTHONPATH=lib; M=models/p_k12b4_50k.pt
st() { echo "$(date +%H:%M:%S) $1" >> logs/leverC_status.log; }
st "1. terse paraphrases of train (7B, style terse)"
uv run python paraphrase.py --split train --seed 3 --style terse --out data/para_train_terse.json > logs/para_train_terse.log 2>&1; st "  $(grep -c . data/para_train_terse.json) lines; $(grep PARA_DONE logs/para_train_terse.log)"
python3 - <<'PY'
import json, re
cjk=re.compile(r"[぀-ヿ㐀-鿿가-힯]"); P=json.load(open("data/para_train.json")); C={k:v for k,v in P.items() if not cjk.search(v)}
json.dump(C, open("data/para_train_clean.json","w")); print("clean natural paraphrases:", len(C), "of", len(P))
PY
st "2. embedder bge_ft3 (plain + clean natural + terse)"
uv run python finetune_embed.py --extra-queries data/para_train_clean.json --extra2 data/para_train_terse.json --out models/bge_ft3 > logs/ft_bge3.log 2>&1; st "  $(grep 'pairs\|fine-tuned' logs/ft_bge3.log | tr '\n' ' ')"
uv run python embed_text2.py --model bgeft3 --splits train,val > logs/embed_bgeft3.log 2>&1
uv run python embed_text2.py --model bgeft3 --reuse-docs --splits train --queries data/para_train.json --tag _para > logs/embed_bgeft3_para_train.log 2>&1
uv run python embed_text2.py --model bgeft3 --reuse-docs --splits val --queries data/para_val.json --tag _para > logs/embed_bgeft3_para_val.log 2>&1
grep -h "text-only" logs/embed_bgeft3.log logs/embed_bgeft3_para_val.log | sed "s/^/    /" >> logs/leverC_status.log
st "3. out-of-fold text features (5 folds)"
for k in 0 1 2 3 4; do uv run python oof_embed.py --fold 5:$k --extra data/para_train_clean.json --extra2 data/para_train_terse.json --tag bgeft3 > logs/oof_bgeft3_f$k.log 2>&1; st "  fold $k $(grep OOF_DONE logs/oof_bgeft3_f$k.log)"; done
python3 - <<'PY'
import json
for suf in ("", "_para"):
    out={}
    for k in range(5): out.update(json.load(open(f"data/oof/bgeft3_f{k}{suf}.json")))
    json.dump(out, open(f"data/text_train_bgeft3oof{suf}.json","w")); print("bgeft3oof", suf, len(out))
PY
st "4. fusion weight + reranker refit with bgeft3 features"
uv run python fuse.py --text-tag bgeft3 --rel-tag lp_ancf > logs/fuse_bgeft3.log 2>&1; W=$(python3 -c "import json; print(json.load(open('data/fusion_bgeft3.json'))['w'])"); st "  w=$W"
uv run python rerank.py --rel-tag lp_ancf --text-tag bgeft3 --train-text-tag bgeft3oof --aug-rel-tag para_lp_ancf --aug-text-tag bgeft3oof_para --t2l-tag pjoint --train-t2l-tag pjointoof --aug-t2l-tag pjointoof_para --save-tag _oof > logs/oof_rerank_lpC.log 2>&1
grep "VAL reranked per-type" logs/oof_rerank_lpC.log | sed "s/^/    /" >> logs/leverC_status.log
RR=data/rerank_lp_ancf_bgeft3_pjoint_aug_oof.json
st "5. re-score the four readings (train + val, plain + para) with the new reranker"
mkdir -p results_c
for S in train val; do for V in plain para; do
  if [ $S = train ]; then TX=data/text_train_bgeft3$([ $V = para ] && echo _para).json; TL=data/text_train_pjointoof$([ $V = para ] && echo _para).json; else TX=data/text_val_bgeft3$([ $V = para ] && echo _para).json; TL=data/text_val_pjoint$([ $V = para ] && echo _para).json; fi
  for r in r1 r2 r3 r4; do
    uv run python predict.py --split $S --rel data/relB_${S}_${V}_$r.json --text $TX --t2l $TL --w $W --rerank $RR --scores-out data/scC_${S}_${V}_$r.json --score 2>&1 | grep "COMMITTED" | sed "s/^/    $S $V $r /" >> logs/leverC_status.log
    mv eval_results_$S.csv results_c/${S}_${V}_$r.csv
  done
done; done
st "LEVERC_DONE"
