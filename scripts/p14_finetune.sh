#!/bin/bash
# P14 step 2 (box 50270859): fine-tune PubMedBERT as the text ranker, out-of-fold, and score val.
cd "$(dirname "$0")/.."; export PYTHONPATH=lib; M=models/p_k12b4_50k.pt
PY="uv run --active --no-project python"
B=NeuML/pubmedbert-base-embeddings
st() { echo "$(date +%H:%M:%S) $1" >> logs/p14b_status.log; }
rm -f logs/p14b_status.log; st "1. fine-tune $B on train pairs"
$PY finetune_embed.py --base $B --extra-queries data/para_train.json --out models/pmb_ft > logs/p14b_ft.log 2>&1
if [ ! -d models/pmb_ft ]; then st "ABORT: fine-tune failed — $(tail -1 logs/p14b_ft.log | cut -c1-140)"; exit 1; fi
grep -E "pairs|fine-tuned" logs/p14b_ft.log | sed "s/^/    /" >> logs/p14b_status.log
st "2. corpus + val rankings with the fine-tuned model"
$PY - <<'PY' >> logs/p14b_status.log 2>&1
import json, sys, torch, numpy as np
sys.path[:0] = ["lib"]
import stark_shim
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer
from metrics import stark_metrics, summarize
QPRE = "Represent this sentence for searching relevant passages: "
docs = [json.loads(l)["text"][:3000] for l in open("data/docs.jsonl")]
m = SentenceTransformer("models/pmb_ft", device="cuda"); m.max_seq_length = 512
D = m.encode(docs, batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
np.save("data/doc_emb_pmbft.npy", D.astype(np.float16)); Dt = torch.from_numpy(D).cuda()
qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()
W = {"plain": None, "terseA": "data/para_val_terse.json", "terseB": "data/para_val_terse_b.json", "terseC": "data/para_val_terse_c.json"}
for w, f in W.items():
    QS = json.load(open(f)) if f else {}
    Q = m.encode([QPRE + QS.get(str(int(qa[i][1])), qa[i][0]) for i in idx], batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
    top = torch.topk(torch.from_numpy(Q).cuda() @ Dt.t(), 100, 1).indices.cpu().numpy()
    json.dump({int(qa[i][1]): top[j].tolist() for j, i in enumerate(idx)}, open(f"data/text_val_pmbft_{w}.json", "w"))
    print(f"    pmbft text-only val {w}:", {k: round(v, 4) for k, v in summarize([stark_metrics(top[j].tolist(), qa[i][2]) for j, i in enumerate(idx)]).items()})
PY
st "3. five out-of-fold folds"
for k in 0 1 2 3 4; do
  $PY oof_embed.py --base $B --fold 5:$k --extra data/para_train.json --tag pmbftoof > logs/p14b_oof_f$k.log 2>&1
  if [ ! -f data/oof/pmbftoof_f$k.json ]; then st "ABORT: fold $k failed — $(tail -1 logs/p14b_oof_f$k.log | cut -c1-140)"; exit 1; fi
  st "  fold $k done"
done
$PY - <<'PY'
import json
for suf in ("", "_para"):
    out = {}
    for k in range(5): out.update(json.load(open(f"data/oof/pmbftoof_f{k}{suf}.json")))
    json.dump(out, open(f"data/text_train_pmbftoof{suf}.json", "w")); print("pmbftoof", suf, len(out))
PY
st "4. reranker refit and val scoring"
cp data/text_val_pmbft_plain.json data/text_val_pmbft.json
$PY rerank.py --rel-tag lp_ancf --text-tag pmbft --train-text-tag pmbftoof --aug-rel-tag para_lp_ancf --aug-text-tag pmbftoof_para \
   --t2l-tag pjoint --train-t2l-tag pjointoof --aug-t2l-tag pjointoof_para --save-tag _oof > logs/p14b_rerank.log 2>&1
grep "VAL reranked" logs/p14b_rerank.log | sed "s/^/    /" >> logs/p14b_status.log
RR=data/rerank_lp_ancf_pmbft_pjoint_aug_oof.json
for W in plain terseA terseB terseC; do
  case $W in
    plain)  RL=data/rel_val_lp_ancf.json;          TL=data/text_val_pjoint.json ;;
    terseA) RL=data/rel_val_p10P3_terse.json;      TL=data/text_val_pjoint_terse.json ;;
    terseB) RL=data/rel_val_p12P3.json;            TL=data/text_val_pjoint_terseb.json ;;
    terseC) RL=data/rel_val_p13P3_terseC.json;     TL=data/text_val_pjoint_tersec.json ;;
  esac
  $PY predict.py --split val --rel $RL --text data/text_val_pmbft_$W.json --t2l $TL --w 0.45 --rerank $RR --score 2>&1 | grep COMMITTED | sed "s/^/    P14b $W  /" >> logs/p14b_status.log
done
rm -f eval_results_val.csv
st "P14B_DONE"
