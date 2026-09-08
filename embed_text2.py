"""Text retriever with a chosen embedder (P1 addendum): --model bge|qwen|minilm. Embeds every node
text and the train/val queries, writes data/text_{split}_{tag}.json (top-100) and prints text-only
val/train metrics. Query/passage prefixes follow each model's convention. Test queries: never."""
import argparse, json, time
import numpy as np, torch
import stark_shim  # noqa
from stark_qa import load_qa
from sentence_transformers import SentenceTransformer
from metrics import stark_metrics, summarize
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", "", {}),
    "bge": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", "", {}),
    "bgeft": ("models/bge_ft", "Represent this sentence for searching relevant passages: ", "", {}),
    "bgeft2": ("models/bge_ft2", "Represent this sentence for searching relevant passages: ", "", {}),   # plain + paraphrased train
    "bgeft3": ("models/bge_ft3", "Represent this sentence for searching relevant passages: ", "", {}),   # lever C: plain + cleaned paraphrases + terse paraphrases
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", "Instruct: Given a biomedical question, retrieve the knowledge-base entries that answer it\nQuery: ", "", {"trust_remote_code": True}),
}
p = argparse.ArgumentParser(); p.add_argument("--model", default="bge"); p.add_argument("--max-len", type=int, default=512)
p.add_argument("--splits", default="train,val", help="train,val for development; test,test-0.1,human_generated_eval for the committed prediction only (no metrics)")
p.add_argument("--reuse-docs", action="store_true")
p.add_argument("--queries", default=None, help="json {query_id: text} replacing the question text (paraphrase proxy; train/val only)")
p.add_argument("--tag", default="", help="suffix for the output file name")
p.add_argument("--proj", default=None, help="P8: data/docproj_<tag>_<d>.npy — project documents and queries into the corpus's rank-d subspace (a smaller document matrix)")
p.add_argument("--docs", default=None, help="use this document matrix instead of data/doc_emb_<model>.npy (e.g. an already projected one)")
a = p.parse_args(); name, qpre, dpre, kw = MODELS[a.model]
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer(name, device=dev, **kw); m.max_seq_length = a.max_len
docs = [json.loads(l) for l in open("data/docs.jsonl")]
t0 = time.time()
if a.docs:
    D = np.load(a.docs).astype(np.float32)
elif a.reuse_docs:
    D = np.load(f"data/doc_emb_{a.model}.npy").astype(np.float32)
else:
    D = m.encode([dpre + d["text"][:3000] for d in docs], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    np.save(f"data/doc_emb_{a.model}.npy", D.astype(np.float16)); print(a.model, "docs embedded", D.shape, round(time.time()-t0), "s", flush=True)
PROJ = torch.from_numpy(np.load(a.proj).astype(np.float32)).to(dev) if a.proj else None
Dt = torch.from_numpy(D).to(dev)
if PROJ is not None and Dt.shape[1] == PROJ.shape[0]: Dt = Dt @ PROJ      # documents given unprojected
for split in a.splits.split(","):
    qa = load_qa("prime", human_generated_eval=(split == "human_generated_eval")); sp = qa.get_idx_split()
    idx = sp[split].tolist() if split != "human_generated_eval" else list(range(len(qa))); rows, out = [], {}
    dev_split = split in ("train", "val")
    QS = json.load(open(a.queries)) if a.queries else None
    assert QS is None or dev_split
    Q = m.encode([qpre + (QS.get(str(int(qa[i][1])), qa[i][0]) if QS else qa[i][0]) for i in idx], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Qt = torch.from_numpy(Q).to(dev)
    if PROJ is not None: Qt = Qt @ PROJ
    top = torch.topk(Qt @ Dt.t(), 100, dim=1).indices.cpu().numpy()
    for j, i in enumerate(idx):
        q, qid, ans, _ = qa[i]; out[int(qid)] = top[j].tolist()
        if dev_split: rows.append(stark_metrics(top[j].tolist(), ans))
    if dev_split: print(f"{a.model} text-only {split}:", {k: round(v, 4) for k, v in summarize(rows).items()}, flush=True)
    json.dump(out, open(f"data/text_{split}_{a.model}{a.tag}.json", "w"))
print("TEXT_DONE")
