"""Text retriever for the probe: MiniLM sentence embeddings over every node's STaRK text and
over the train/val queries; cosine top-100 per query. Writes data/text_{split}.json and the
node embedding matrix data/doc_emb.npy. Test queries are never embedded here (P1)."""
import json, sys, time
import numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer
from metrics import stark_metrics, summarize
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
docs = [json.loads(l) for l in open("data/docs.jsonl")]
t0 = time.time()
D = m.encode([d["text"][:2000] for d in docs], batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
np.save("data/doc_emb.npy", D.astype(np.float16)); print("docs embedded", D.shape, round(time.time()-t0), "s", flush=True)
Dt = torch.from_numpy(D).to(dev)
qa = load_qa("prime"); sp = qa.get_idx_split()
for split in ("train", "val"):
    idx = sp[split].tolist(); rows, out = [], {}
    qs = [qa[i][0] for i in idx]; Q = m.encode(qs, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    S = torch.from_numpy(Q).to(dev) @ Dt.t()
    top = torch.topk(S, 100, dim=1).indices.cpu().numpy()
    for j, i in enumerate(idx):
        q, qid, ans, _ = qa[i]; rows.append(stark_metrics(top[j].tolist(), ans)); out[int(qid)] = top[j].tolist()
    print(f"MiniLM text-only {split}:", {k: round(v, 4) for k, v in summarize(rows).items()}, flush=True)
    json.dump(out, open(f"data/text_{split}.json", "w"))
print("TEXT_DONE")
