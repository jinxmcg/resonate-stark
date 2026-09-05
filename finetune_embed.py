"""P2 lever 2a: fine-tune bge-base-en-v1.5 on STaRK-Prime `train` (query -> answer node text) with a
contrastive loss (in-batch negatives). Writes models/bge_ft/ and then re-embeds the corpus and the
train/val queries through embed_text2.py conventions (tag 'bgeft'). Test/human never touched."""
import json, random, time, sys
import numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
random.seed(0); torch.manual_seed(0)
docs = {json.loads(l)["id"]: json.loads(l)["text"][:1500] for l in open("data/docs.jsonl")}
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
QPRE = "Represent this sentence for searching relevant passages: "
ex = []
for i in tr:
    q, qid, ans, _ = qa[i]
    for a_ in ans[:3]:
        ex.append(InputExample(texts=[QPRE + q, docs[a_]]))
random.shuffle(ex); print("pairs", len(ex), flush=True)
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.max_seq_length = 384
dl = DataLoader(ex, shuffle=True, batch_size=48)
loss = losses.MultipleNegativesRankingLoss(m)
t0 = time.time(); m.fit(train_objectives=[(dl, loss)], epochs=2, warmup_steps=100, show_progress_bar=False)
m.save("models/bge_ft"); print("fine-tuned in", round(time.time()-t0), "s", flush=True)
print("FT_DONE")
