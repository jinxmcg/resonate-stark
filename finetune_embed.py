"""P2 lever 2a: fine-tune bge-base-en-v1.5 on STaRK-Prime `train` (query -> answer node text) with a
contrastive loss (in-batch negatives). Writes models/bge_ft/ and then re-embeds the corpus and the
train/val queries through embed_text2.py conventions (tag 'bgeft'). Test/human never touched."""
import json, random, time, sys, argparse
import numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
ap = argparse.ArgumentParser(); ap.add_argument("--extra-queries", default=None, help="json {query_id: text}: paraphrased train questions, added as extra (query, answer) pairs")
ap.add_argument("--out", default="models/bge_ft"); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--extra2", default=None); A = ap.parse_args()
random.seed(0); torch.manual_seed(0)
docs = {json.loads(l)["id"]: json.loads(l)["text"][:1500] for l in open("data/docs.jsonl")}
qa = load_qa("prime"); tr = qa.get_idx_split()["train"].tolist()
QPRE = "Represent this sentence for searching relevant passages: "
ex = []
EXTRA = json.load(open(A.extra_queries)) if A.extra_queries else {}; EXTRA2 = json.load(open(A.extra2)) if A.extra2 else {}
for i in tr:
    q, qid, ans, _ = qa[i]
    for qq in [q] + ([EXTRA[str(int(qid))]] if str(int(qid)) in EXTRA else []) + ([EXTRA2[str(int(qid))]] if str(int(qid)) in EXTRA2 else []):
        for a_ in ans[:3]:
            ex.append(InputExample(texts=[QPRE + qq, docs[a_]]))
random.shuffle(ex); print("pairs", len(ex), flush=True)
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.max_seq_length = 384
dl = DataLoader(ex, shuffle=True, batch_size=48)
loss = losses.MultipleNegativesRankingLoss(m)
t0 = time.time(); m.fit(train_objectives=[(dl, loss)], epochs=A.epochs, warmup_steps=100, show_progress_bar=False)
m.save(A.out); print("fine-tuned in", round(time.time()-t0), "s", flush=True)
print("FT_DONE")
