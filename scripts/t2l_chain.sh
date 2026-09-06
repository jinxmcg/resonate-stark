#!/bin/bash
# lever 4 chain: waits for the LLM parser chain to free VRAM, then trains text-to-latent (frozen table) and, if it runs, the joint version
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
until grep -q LLM_DONE logs/llm_override_para.log 2>/dev/null; do sleep 20; done
uv run python text2latent.py --encoder models/bge_ft2 --extra data/para_train.json --tag t2l > logs/t2l.log 2>&1
echo T2L_EXIT >> logs/t2l.log
uv run python text2latent.py --encoder models/bge_ft2 --extra data/para_train.json --tag t2lj --unfreeze-table > logs/t2lj.log 2>&1
echo T2L_EXIT >> logs/t2lj.log
