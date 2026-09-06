#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
for D in "z,exact,exact/nm,rel_rrf,in_rel,fused" "txt_rrf,in_txt"; do
  echo "== drop $D"
  uv run python rerank.py --rel-tag ancf --text-tag bgeft --no-save --drop "$D" 2>&1 | grep "VAL reranked per-type"
done
echo ABL_DONE
