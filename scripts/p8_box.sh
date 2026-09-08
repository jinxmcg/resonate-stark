#!/bin/bash
# P8 (rented RTX 5090, vast 50261550): fewer PARAMETERS — the anchor fallback served by the latent
# parser's own head (-208.9M) and a rank-reduced document matrix (-66.1M at d=256).
# Step 1 chooses d on train; step 2 is one val read of A' and A'+B'. See PLAN_PRIME.md "P8".
cd "$(dirname "$0")/.."; export PYTHONPATH=lib
PY="uv run --active --no-project python"
st() { echo "$(date +%H:%M:%S) $1" >> logs/p8_status.log; }
rm -f logs/p8_status.log; st "step 1: choose d on train (text ranker alone)"
$PY embed_text2.py --model bgeft2 --reuse-docs --splits train --tag _d768 > logs/p8_d768.log 2>&1
grep "text-only" logs/p8_d768.log | sed "s/^/    d=768  /" >> logs/p8_status.log
for D in 128 256 384; do
  $PY embed_text2.py --model bgeft2 --docs data/doc_emb_bgeft2_p${D}.npy --proj data/docproj_bgeft2_${D}.npy \
      --splits train --tag _d${D} > logs/p8_d${D}.log 2>&1
  grep "text-only" logs/p8_d${D}.log | sed "s/^/    d=$D  /" >> logs/p8_status.log
done
st "STEP1_DONE"
