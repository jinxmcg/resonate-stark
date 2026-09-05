# PLAN_PRIME — STaRK-Prime probe (separate from the OGB work; own numbering)

## P1 pre-registration — the relational half of a semi-structured retrieval benchmark (prime/, jinx 1080 Ti, 2026-09-05)

Question (user): can the same architecture serve as the relational
retriever for STaRK-Prime (PrimeKG as a semi-structured KB: 129,375
entities of 10 types, 8,100,498 edges of 18 relation types; 11,204
synthesized queries with train/val/test splits, 109 human queries)?
Splits this hypothesis READS: the SKB's edges and node text (all; the
KB is the retrieval corpus, there is no link-prediction test split);
STaRK query splits `train` (fusion fit) and `val` (all development
reads). STaRK `test` and `human_generated_eval`: NOT read during
development; at most ONE committed read of each at the end, and only if
val says go (bar below).
Pipeline: (1) ResonatE on all PrimeKG edges through the sparse shell,
both dial settings screened on a 2 % held-out edge slice for link
prediction sanity only (k=12/4x4 and k=8/dense 64; PrimeKG is a typed
biomedical graph, so 4x4 is the prior; the dial is a validation choice,
not a claim). (2) A rule/template parser from the
query text to (entity mention, relation, direction) constraints and an
answer node type, using the SKB's node names and a hand keyword->relation
map for the 18 relations; coverage reported. (3) Relational retriever:
per constraint, the compiled operator + readout over all 129k rows,
z-scored and summed across constraints, answer-type filter. (4) Text
retriever: an offline sentence embedder (MiniLM class) over node text.
(5) Fusion weights fit on STaRK `train` queries only. Baseline computed
by us on `val`: BM25 over node text (rank_bm25).
Metrics: Hit@1, Hit@5, Recall@20, MRR on `val` (STaRK's evaluator),
relational-only / text-only / fused, plus the parser's coverage.
Bar for a test read: fused val Hit@1 >= our BM25 val Hit@1 + 0.10 AND
relational-only alone beats BM25 on the queries it covers; else the
probe is reported on val only. Reference points from the STaRK paper
(their test set): BM25 12.75 Hit@1 / 31.25 R@20, multi-ada-002 15.10 /
38.05 (MRR 23.49), GPT-4 reranker ~18 Hit@1 / 34 R@20.
Hardware: jinx GTX 1080 Ti (torch 2.6.0, own venv in prime/).

### P1 progress (2026-09-05)
Dial screen, 12,500 steps on the 1080 Ti (~2.2 min each), held-out 2 % edges,
MRR over 500 uniform negatives, both directions: k=12 / 4x4 (37.3M params)
0.5601; k=8 / dense 64x64 (16.9M) 0.5526 -> 4x4 kept, as the prior said.
Answer-type detection covers 2,130 / 2,241 val queries (95 %).
First val numbers (12.5k-step k=12/4x4 model; rule parser; MiniLM text; RRF fusion
with w chosen on train = 0.50): text-only 0.0785 / 0.1896 / 0.2188 / 0.1325
(Hit@1 / Hit@5 / R@20 / MRR); relational-only with text fallback for the 24 %
uncovered queries 0.1455 / 0.2874 / 0.3503 / 0.2144; fused 0.1673 / 0.3427 /
0.4185 / 0.2505. Paper reference (their test set): multi-ada-002 0.1510 /
0.3356 / 0.3805 / 0.2349. Parser coverage 76 % (val and train alike).
50k-step model (held-out link MRR 0.557, saturated by 12.5k) + parser v2 (two-hop chains,
gene symbols, "no drugs" filter): coverage 88.2 %; relational-only all-val 16.5 / 29.5 /
36.7 / 22.8 (%), covered 18.7 / 33.4 / 41.6 / 25.9; fused with MiniLM (w=0.5 on train)
18.7 / 37.3 / 45.3 / 27.5. Board top (AvaTaR gpt-4-turbo, TEST): 20.1 / 39.9 / 42.2 / 29.2.
Exact-neighbour boost (candidates reached by exact traversal of the mention's candidate
chains get +beta per satisfied constraint): first-400-val probe, beta=3, covered Hit@1
20.2 -> 24.4. Beta is chosen on train next; val only.
Beta sweep on train (2,000 queries, all-query Hit@1 / Hit@5 / R@20 / MRR, %): beta 0 ~17 /
30 / 37 / 23; 1: 18.9 / 33.4 / 39.6 / 25.6; 3: 20.8 / 35.0 / 41.8 / 27.5; 10: 21.6 /
35.4 / 42.5 / 28.2; 30: 21.7 / 35.4 / 42.6 / 28.2; 100: same as 30 (saturated: exact
graph support dominates, the model orders within). Rule: max train MRR -> beta = 30.
P1 addendum (2026-09-05): text embedder chosen on train among {MiniLM-L6-v2, bge-base-en-v1.5,
Qwen3-Embedding-0.6B} (embed_text2.py; corpus re-embedded per model; query/passage prefixes per
model); fusion weight refit on train per embedder; val reported; test never.
Val with beta = 30 (50k model, parser v2), %: relational-only all queries 22.8 / 36.6 /
43.9 / 29.2 (covered 25.8 / 41.5 / 49.8 / 33.1); + MiniLM fusion (w = 0.5 on train)
23.4 / 44.1 / 52.4 / 33.1. Board top AvaTaR (TEST) 20.1 / 39.9 / 42.2 / 29.2.
Val is not test; the bar (P1) is checked against our own BM25 on val, still running.
Embedder bge-base-en-v1.5 (corpus 94 s on the 5090): text-only val 10.5 / 27.9 / 33.2 /
18.5 (MiniLM 7.9 / 19.0 / 21.9 / 13.3); fused with the beta-30 relational ranking (w = 0.5
on train) val 25.8 / 48.2 / 58.3 / 36.2. Qwen3-Embedding-0.6B next; BM25 val still running.
Qwen3-Embedding-0.6B (corpus 211 s): text-only val 8.9 / 24.8 / 30.4 / 16.6 (below bge here,
probably the prompt/pooling convention through sentence-transformers); fused val 26.1 / 48.2 /
56.3 / 36.3 vs bge-fused 25.8 / 48.2 / 58.3 / 36.2. Rule = max train fused MRR: qwen 0.3629 >
bge 0.3586 -> Qwen3 fusion is the selected configuration (val reported for both).
Text-resolved anchors (bge nearest nodes of a type that can reach the answer type, for queries
with no exact mention; weight 0.7): coverage 88.2 -> 96.2 %; relational-only all-val 23.7 /
39.5 / 47.4 / 31.0; fused val: + Qwen3 26.8 / 49.7 / 58.4 / 37.3, + bge 26.4 / 49.5 / 60.3 /
37.2. Frozen configuration = k12/4x4 50k model, parser v2, beta 30, bge anchors, embedder by
train fused MRR, RRF w = 0.5 (train).

### P1 bar check and FREEZE (2026-09-05 23:0x)
Our BM25 on val: 13.7 / 27.1 / 30.4 / 20.1. Bar: fused val Hit@1 >= 13.7 + 10 = 23.7 -> ours
26.8 (PASS); relational-only on covered queries 24.6 vs BM25 13.7 (PASS). Frozen pipeline:
k=12/4x4 sparse-shell model (models/p_k12b4_50k.pt), parser v2, exact-support beta = 30,
bge-resolved anchors for unparsed queries (top-2, weight 0.7), Qwen3-Embedding-0.6B text
ranking, RRF w = 0.5 (all chosen on train; embedder by train fused MRR 0.3738 vs bge 0.3676).
Committed reads, one each, now: STaRK `test` (the 10 % subset is reported from the same
predictions), `human_generated_eval`. Prediction CSVs in STaRK's eval_results_{split}.csv format.

### P1 RESULT — committed reads (2026-09-05 23:0x, frozen pipeline, one read each)
(The first invocation of predict.py crashed on an import before any answers were read; the
reads below are the first and only ones.) Hit@1 / Hit@5 / R@20 / MRR, %:
  Synthesized (full), STaRK `test`, 2,801 queries, parser coverage 95.9 %: 28.7 / 51.9 / 59.9 / 39.1
  Synthesized (10 %), `test-0.1`, 280 queries (same predictions):           28.2 / 50.7 / 59.9 / 38.5
  Human-generated, `human_generated_eval`, 98 queries, coverage 90.8 %:      20.4 / 41.8 / 48.6 / 29.9
Board top on Synthesized (full), AvaTaR (gpt-4-turbo): 20.1 / 39.9 / 42.2 / 29.2. Paper's Claude-3
reranker on human queries: 28.6 / 46.9 / 41.6 / 36.3. No LLM anywhere in our pipeline.
Prediction CSVs (STaRK format) and all logs: prime/results/.
