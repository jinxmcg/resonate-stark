"""Our own BM25 baseline on STaRK-Prime `val` (all 129k node texts as documents), plus the
exact-name mention coverage of the val queries (how many queries name at least one node)."""
import json, re, time, collections
import numpy as np
import stark_shim  # noqa
from stark_qa import load_qa
from rank_bm25 import BM25Okapi
from metrics import stark_metrics, summarize
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
docs = [json.loads(l) for l in open("data/docs.jsonl")]
t0 = time.time(); bm = BM25Okapi([tok(d["text"]) for d in docs]); print("bm25 built", round(time.time()-t0), "s", flush=True)
qa = load_qa("prime"); val = qa.get_idx_split()["val"].tolist()
names = json.load(open("data/names.json")); low = collections.defaultdict(list)
for i, n in names.items():
    if len(n) >= 4: low[n.lower()].append(int(i))
rows, covered, t0 = [], 0, time.time()
for j, i in enumerate(val):
    q, qid, ans, _ = qa[i]
    s = bm.get_scores(tok(q)); top = np.argpartition(-s, 100)[:100]; top = top[np.argsort(-s[top])]
    rows.append(stark_metrics(top.tolist(), ans))
    ql = q.lower()
    if any(re.search(r"\b" + re.escape(n) + r"\b", ql) for n in low if len(n) >= 6 and n in ql): covered += 1
    if j % 500 == 0: print(j, round(time.time()-t0), "s", flush=True)
print("BM25 val:", {k: round(v, 4) for k, v in summarize(rows).items()}, "n", len(rows))
print("val queries with >=1 exact node-name mention (>=6 chars):", covered, "/", len(val), flush=True)
json.dump({"bm25_val": summarize(rows), "coverage_names6": covered / len(val)}, open("data/bm25_val.json", "w"))
