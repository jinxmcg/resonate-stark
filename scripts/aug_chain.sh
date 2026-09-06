#!/bin/bash
# lever 2c augmentation chain: waits for the train paraphrase, then refits embedder + reranker on plain + paraphrased train
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
until grep -q PARA_EXIT logs/para_train.log 2>/dev/null; do sleep 30; done
uv run python finetune_embed.py --extra-queries data/para_train.json --out models/bge_ft2 > logs/ft_bge2.log 2>&1
uv run python embed_text2.py --model bgeft2 --splits train,val > logs/embed_bgeft2.log 2>&1
uv run python embed_text2.py --model bgeft2 --reuse-docs --splits train --queries data/para_train.json --tag _para > logs/embed_bgeft2_para_train.log 2>&1
uv run python embed_text2.py --model bgeft2 --reuse-docs --splits val --queries data/para_val.json --tag _para > logs/embed_bgeft2_para_val.log 2>&1
uv run python retrieve.py --model models/p_k12b4_50k.pt --split train --beta 30 --anchor bge --dump-feats --queries data/para_train.json --out data/rel_train_para_ancf.json > logs/rel_train_para.log 2>&1
uv run python fuse.py --text-tag bgeft2 --rel-tag ancf > logs/fuse_bgeft2_ancf.log 2>&1
uv run python rerank.py --rel-tag ancf --text-tag bgeft2 > logs/rerank_bgeft2_plain.log 2>&1
cp data/rerank_ancf_bgeft2.json data/rerank_ancf_bgeft2_plainfit.json
uv run python rerank.py --rel-tag ancf --text-tag bgeft2 --aug-rel-tag para_ancf --aug-text-tag bgeft2_para > logs/rerank_bgeft2_aug.log 2>&1
for T in plain para; do
  if [ $T = plain ]; then TX=data/text_val_bgeft2.json; RL=data/rel_val_ancf.json; else TX=data/text_val_bgeft2_para.json; RL=data/rel_val_para_ancf.json; fi
  uv run python predict.py --split val --rel $RL --text $TX --w 0.5 --rerank data/rerank_ancf_bgeft2_plainfit.json --score > logs/aug_${T}_plainfit.log 2>&1
  uv run python predict.py --split val --rel $RL --text $TX --w 0.5 --rerank data/rerank_ancf_bgeft2.json --score > logs/aug_${T}_augfit.log 2>&1
done
rm -f eval_results_val.csv
echo AUG_DONE >> logs/aug_para_augfit.log
