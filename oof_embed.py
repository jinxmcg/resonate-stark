"""Out-of-fold text-ranking features for the reranker: fine-tune bge-base on train minus fold k
(plain + paraphrased pairs, same recipe as finetune_embed.py), embed the corpus, rank the held-out
fold's questions (plain + paraphrased) -> data/oof/bgeft2_f{k}.json / _para.json.
Usage: uv run python oof_embed.py --fold 5:0 --extra data/para_train.json"""
import argparse, json, random, time, os
import numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
p = argparse.ArgumentParser(); p.add_argument("--fold", required=True); p.add_argument("--extra", default=None); p.add_argument("--epochs", type=int, default=2); a = p.parse_args()
K, k = (int(x) for x in a.fold.split(":")); random.seed(0); torch.manual_seed(0)
QPRE = "Represent this sentence for searching relevant passages: "
docs = [json.loads(l) for l in open("data/docs.jsonl")]; dtext = {d["id"]: d["text"] for d in docs}
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist(); EXTRA = json.load(open(a.extra)) if a.extra else {}
ex, held = [], []
for pos, i in enumerate(tr):
    q, qid, ans, _ = qa[i]
    if pos % K == k: held.append(i); continue
    for qq in [q] + ([EXTRA[str(int(qid))]] if str(int(qid)) in EXTRA else []):
        for a_ in ans[:3]: ex.append(InputExample(texts=[QPRE + qq, dtext[a_][:1500]]))
random.shuffle(ex); print("fold", k, "pairs", len(ex), "held", len(held), flush=True)
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.max_seq_length = 384
t0 = time.time(); m.fit(train_objectives=[(DataLoader(ex, shuffle=True, batch_size=48), losses.MultipleNegativesRankingLoss(m))], epochs=a.epochs, warmup_steps=100, show_progress_bar=False)
print("fine-tuned", round(time.time() - t0), "s", flush=True); m.max_seq_length = 512
D = torch.from_numpy(m.encode([d["text"][:3000] for d in docs], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)).cuda()
os.makedirs("data/oof", exist_ok=True)
for suf, QS in (("", {}), ("_para", EXTRA)):
    Q = m.encode([QPRE + QS.get(str(int(qa[i][1])), qa[i][0]) for i in held], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    top = torch.topk(torch.from_numpy(Q).cuda() @ D.t(), 100, dim=1).indices.cpu().numpy()
    json.dump({int(qa[i][1]): top[j].tolist() for j, i in enumerate(held)}, open(f"data/oof/bgeft2_f{k}{suf}.json", "w"))
print("OOF_DONE", k, round(time.time() - t0), "s")
