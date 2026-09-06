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

### FROZEN as p1-frozen (local git tag in prime/, FROZEN_P1.md lists every component and flag).

## P2 pre-registration — Hit@1 and human-written queries, without a second look at test (2026-09-05)
Reads: `train` (fitting), `val` (all development numbers). `test`, `test-0.1`,
`human_generated_eval`: not loaded during P2; at most one committed read of each at the end,
which would be the SECOND read of test overall — both reads are reported side by side.
Levers, each ablated on val:
 1. A learned reranker over the fused candidate set, fit on STaRK `train` queries (a split
    meant for training): per answer type, a listwise logistic regression on [model z-score,
    exact-support count, number of constraints satisfied, text RRF score, node degree]; the
    wikikg2 combiner's mechanism, on a split where fitting is allowed.
 2. Parser robustness for non-template phrasing, developed BLIND on the synthesized val split
    (the human set is never opened): answer-type synonyms, more relation keywords, three-hop
    chains where the type signatures need them, anchor resolution top-3 with a per-type
    similarity floor, and a relation classifier trained on `train` queries (query text ->
    relation set) to replace keyword hints.
 3. Ranking within the graph-supported set: temperature/calibration of the model score per
    relation, fit on `train`.
Bar for the second committed read: fused val Hit@1 >= 30.0 (P1: 26.8) or val MRR >= 40.0
(P1: 37.3), and no lever kept that does not improve val. Human set: reported from the same
frozen P2 pipeline, one read, next to P1's 20.4 / 41.8 / 48.6 / 29.9.
P2 lever 2a (added before any run): fine-tune bge-base-en-v1.5 on STaRK `train` (query ->
answer node text) pairs with a contrastive (MultipleNegativesRanking) loss, a few epochs on
the 5090; used for BOTH the text ranking and the anchor resolution; evaluated on val
text-only, anchor coverage/quality, and fused. Val decides; the human set stays closed.

### P2 progress (2026-09-05, all numbers on `val`; test / test-0.1 / human not opened)
Lever 2a — finetune_embed.py: bge-base-en-v1.5, MultipleNegativesRankingLoss on `train`
(query -> answer node text), 2 epochs, 130 s on the 5090; corpus re-embedded (embed_text2.py
--model bgeft). Fusion weight re-chosen on train (w = 0.50 again).
  val text only, fine-tuned bge      : 22.5 / 42.7 / 48.9 / 31.7   (Qwen text only: 10.5 / 27.9 / 33.2 / 18.5)
  val fused, P1 relational (bge anc) : 26.9 / 54.1 / 67.2 / 39.7   (P1 frozen: 26.8 / 49.7 / 58.4 / 37.3)
  train fused (fit split, for the gap): 29.8 / 56.6 / 67.4 / 42.2
Hit@1 unchanged; Hit@5 / R@20 / MRR up. Bar (Hit@1 >= 30 or MRR >= 40) not yet met.
Lever 1 — retrieve.py --dump-feats stores per-candidate z-sum and exact-support count; rerank.py
fits a listwise logistic regression on the fused candidate set on `train` (global + per answer
type), reports val. Results below when they land.
  Fine-tuned anchors (retrieve.py --anchor bgeft), val: relational only 24.4 / 40.3 / 48.6 / 31.8
  (bge anchors: 24.6 / 41.6 / 50.0 / 32.6); fused with fine-tuned text 26.2 / 53.2 / 66.4 / 39.0
  (bge anchors: 26.9 / 54.1 / 67.2 / 39.7). Rejected: the query->answer fine-tune points the query
  at the answer, not at the entity it mentions. P2 pipeline keeps bge for anchors, bgeft for text.
Lever 1 result (rerank.py, fit on `train` only; features: z-sum, exact count, exact/mentions,
relational RRF term + flag, text RRF term + flag, log degree, fused RRF score; listwise softmax
cross-entropy, L2 1e-3; per answer type when >= 100 train groups, else the global fit):
  val, bge anchors + fine-tuned text:  fused input 26.9 / 54.1 / 67.2 / 39.7
                                       reranked global   30.8 / 57.5 / 67.4 / 42.9
                                       reranked per-type 33.7 / 58.8 / 68.0 / 45.1
  val, fine-tuned anchors (ablation):  reranked per-type 33.1 / 58.5 / 67.6 / 44.6
predict.py --rerank reproduces the per-type number end to end on val (33.65 / 58.81 / 68.04 / 45.12).
P2 BAR MET on val (Hit@1 33.7 >= 30, MRR 45.1 >= 40) with levers 2a (text only) + 1.
P2 pipeline = P1 retrieval (beta 30, bge anchors top-2 w 0.7) + fine-tuned bge text ranking
+ RRF w 0.5 (train) + per-type reranker (train). Levers 2b (relation classifier) and 3 (per-relation
calibration) NOT yet tried. The second committed read of test / test-0.1 / human has NOT been
made; it is the user's call whether to spend it now or after 2b / 3.
Caveat on the fit: the reranker's text features on `train` come from an embedder fine-tuned on
those same train queries, so the fit sees slightly better text ranks than val does; val is the
honest number and still improves by 6.8 Hit@1 over the fused input.
Ablation (val, per-type reranker, bge anchors + fine-tuned text): dropping the ResonatE z-score
feature from the reranker gives 33.65 / 58.63 / 68.09 / 45.01 vs 33.65 / 58.81 / 68.04 / 45.12 with
it. Once exact traversal and the text ranker are present, the learned vector score adds no ordering
information on this benchmark (it still produces the candidate list). Further drops (exact, all
relational input, text) pending.
Lever 2c (added 2026-09-06, before any run): language normalisation. The human split differs from
the synthesized ones only in phrasing; the facts are the same graph. (a) Proxy human set: a local
instruction model (Qwen2.5-7B-Instruct on the 5090) paraphrases `val` questions into natural,
varied phrasing (no graph information given to it); the P2 pipeline is measured on the paraphrases
vs plain val, and the gap is the parser's phrasing sensitivity. (b) LLM parser: the same local
model, prompted with the PrimeKG schema (10 node types, 18 relations) and few-shot examples from
`train`, maps a question to {answer type, entity mentions, relation hints}; downstream unchanged.
Kept only if it improves paraphrased val without hurting plain val. Human set stays closed.
Lever 2c(a) result — proxy human set (paraphrase.py, Qwen2.5-7B-Instruct, val, 137 s; samples in
logs/p2/para_val.log; parser coverage 94.2% vs 96.2% on plain val):
                              plain val                 paraphrased val
  relational only (P1 path)   24.6 / 41.6 / 50.0 / 32.6   19.3 / 34.4 / 42.1 / 26.4
  fine-tuned text only        22.5 / 42.7 / 48.9 / 31.7   19.4 / 39.3 / 46.1 / 28.7
  P1 pipeline (Qwen text)     26.8 / 49.7 / 58.4 / 37.3   22.5 / 46.1 / 54.3 / 33.3
  P2 without reranker         26.9 / 54.1 / 67.2 / 39.7   24.9 / 49.8 / 62.0 / 36.6
  P2 full                     33.7 / 58.8 / 68.0 / 45.1   30.0 / 53.1 / 63.3 / 41.0
The proxy tracks the real human gap in direction and size (P1: -4.3 Hit@1 on paraphrases, -8.3 on
the 98 real human queries). Under P2 the paraphrase cost is -3.6 Hit@1; both halves of the pipeline
lose about 5 Hit@1 to phrasing. Next: paraphrased `train` as augmentation (embedder + reranker fit
on plain + paraphrased train), then the LLM parser (2c(b)); measured on plain and paraphrased val.
Ablation (cont.): dropping exact traversal from the reranker: 30.8 / 56.0 / 67.9 / 42.5.
Lever 2c(b) smoke (llm_parse.py, Qwen2.5-7B-Instruct, greedy, 3-shot, 64 val questions, 28 s):
all 64 parsed to valid JSON; entities and types look right (e.g. "Mucopolysaccharidosis Type VII
(Sly syndrome)" -> disease; "USH1C" -> gene/protein); relation hints lean on "interacts with".
Retrieval with it (fallback mode): coverage 61 -> 64 of 64, relational-only Hit@1 18.8 vs 20.3
without (64 questions: noise level, no gain visible). Full val + paraphrased val runs queued.
Ablation (cont.): dropping z AND exact: 30.8 / 55.6 / 68.0 / 42.4 (= dropping exact alone).
Lever 4 (added 2026-09-06, before any run) — text-to-latent: put the question INTO the model.
The table was trained on edges only; language was attached next to it (embedder, parser,
reranker), so phrasing never enters the model and the vector score carries no ranking information
beyond traversal + text (ablation above). Lever 4 trains a question encoder into the ResonatE
latent space: z_q = head(bge_ft(question)) with M = 144 complex dims; score(q, t) = Re<z_q, E_t> * e^tau
with the entity table E FROZEN from models/p_k12b4_50k.pt; loss = softmax cross-entropy over the
answer nodes against in-batch + uniform negatives restricted to the answer type when known;
training data = `train` questions + their paraphrases (data/para_train.json). Evaluated on plain
and paraphrased val: text-to-latent alone vs fine-tuned text alone; then as a third candidate
source in fusion and as a reranker feature. Second stage if the first helps: unfreeze the table
and train edges + questions jointly (the model trained "with the language").
Reads: train (fitting), val (all numbers). test / test-0.1 / human: closed.
Lever 2c(b) result — LLM parser (Qwen2.5-7B-Instruct, greedy, 3-shot; 2241 val questions in 560 s,
5 unparsable; paraphrased val 536 s, 3 unparsable). Used for: answer type (fallback = only when the
pattern parser finds none; override = LLM first), relation hints, extra entity names (exact lookup,
then per-name text anchoring when nothing else resolved), exclusion relation.
                                  plain val                    paraphrased val
  relational only, no LLM         24.6 / 41.6 / 50.0 / 32.6      19.3 / 34.4 / 42.1 / 26.4
  relational only, LLM fallback   24.2 / 39.3 / 47.6 / 31.3      19.7 / 35.3 / 43.5 / 27.0
  relational only, LLM override   24.9 / 40.4 / 48.8 / 32.1      20.9 / 36.9 / 45.5 / 28.5
  P2 full (bgeft text + reranker) 33.7 / 58.8 / 68.0 / 45.1      30.0 / 53.1 / 63.3 / 41.0
  P2 full + LLM fallback          33.8 / 58.3 / 68.2 / 45.1      30.1 / 53.7 / 63.9 / 41.3
  P2 full + LLM override          34.6 / 59.2 / 68.9 / 46.0      30.9 / 54.8 / 65.2 / 42.3
Coverage 96.2% -> 100.0% (plain), 94.2% -> 99.8% (paraphrased). Override mode kept: +0.9 Hit@1 on
both, +0.9 / +1.3 MRR; the paraphrase gap is unchanged (~3.7 Hit@1), so the LLM parser adds
coverage rather than phrasing robustness. Reranker here is the plain-train fit on bgeft features.
Lever 4 result, stage 1 (text2latent.py, table frozen, bge_ft2 encoder + linear head into M=144
complex dims, 3 epochs on 12,324 train questions incl. paraphrases, 105 s; loss 7.96 -> 4.38 -> 2.94,
still falling):
                                  plain val                    paraphrased val
  text-to-latent alone            19.6 / 30.9 / 31.3 / 25.2      19.0 / 29.9 / 30.6 / 24.5
  fine-tuned text alone (2a)      22.5 / 42.7 / 48.9 / 31.7      19.4 / 39.3 / 46.1 / 28.7
  relational path alone           24.6 / 41.6 / 50.0 / 32.6      19.3 / 34.4 / 42.1 / 26.4
Phrasing gap 0.6 Hit@1 (vs 3.1 for the text ranker and 5.3 for the parser path): the first
component that is robust to wording, because the question enters the model. Recall@20 is low
(31 vs 49): the frozen table's geometry limits what a single query vector can reach. Next: joint
run (table unfrozen), a longer frozen run (10 epochs), and text-to-latent as a third candidate
source + reranker feature.
Ablation complete (val, per-type reranker, bge anchors + fine-tuned text; feature groups zeroed):
  full                                   33.7 / 58.8 / 68.0 / 45.1
  - ResonatE z-score                     33.7 / 58.6 / 68.1 / 45.0
  - exact traversal                      30.8 / 56.0 / 67.9 / 42.5
  - z-score and exact traversal          30.8 / 55.6 / 68.0 / 42.4
  - all relational features              20.7 / 45.0 / 51.2 / 31.7   (text + degree only, same candidates)
  - text features                        32.9 / 57.8 / 68.0 / 44.5
The graph path carries the ranking (13 Hit@1 without it); text adds 0.7 Hit@1 on top and most of
the recall; the learned vector score adds nothing once traversal is present.
Lever 4 results, stages 2-3:
  joint (table unfrozen, 3 ep, 114 s)   plain 26.2 / 37.8 / 35.2 / 31.7    paraphrased 25.4 / 37.2 / 34.0 / 31.0
  frozen table, 10 ep (593 s)            plain 23.6 / 35.3 / 33.0 / 29.0    paraphrased 23.2 / 33.9 / 32.5 / 28.2
  frozen table, 3 ep                     plain 19.6 / 30.9 / 31.3 / 25.2    paraphrased 19.0 / 29.9 / 30.6 / 24.5
The joint model alone has the best single-source Hit@1 on both wordings (26.2 / 25.4; relational
path 24.6 / 19.3; fine-tuned text 22.5 / 19.4) with a phrasing gap of 0.8. Recall@20 stays ~35:
one query vector reaches few of the multi-answer sets. Caveat for the "same model" story: the
joint run moves the entity table with question supervision; its graph link-prediction quality
after training is not yet re-measured (to do before any claim that one table serves both).
Final combined evaluation queued (final_eval.sh): LLM-override retrieval + bge_ft2 text + joint
text-to-latent as third source; reranker fit plain vs augmented; scored on plain and paraphrased val.
Lever 2c augmentation result (embedder bge_ft2 fine-tuned on plain + paraphrased train, 20,270
pairs, 311 s; reranker fit on plain train vs plain + paraphrased train; plain val):
  bge_ft2 text alone      22.0 / 43.0 / 50.6 / 31.9   (paraphrased val 19.6 / 41.3 / 48.1 / 29.6; bge_ft: 22.5 / 19.4)
  fused (w 0.45 on train) 29.0 / 51.8 / 63.6 / 39.9
  reranker, plain fit     34.4 / 60.8 / 68.3 / 46.0   (bge_ft: 33.7 / 58.8 / 68.0 / 45.1)
  reranker, augmented fit 34.6 / 61.0 / 68.4 / 46.2
The chain's four paraphrased-val scorings crashed on a feature-count mismatch (rerank.py gained
the two text-to-latent features while the chain was running); the final combined evaluation
re-does them with the current code. Ablation table above is complete (ABL_DONE).

### P2 final combined evaluation (2026-09-06; final_eval.sh; val only, test / test-0.1 / human unopened)
Retrieval: P1 parser + LLM parser (override) + beta 30 + bge anchors; text: bge_ft2 (plain + paraphrased
train); RRF w 0.5; reranker fit on train (plain) or train + paraphrased train (aug); optional third
source = joint text-to-latent (t2lj).
                                        plain val                       paraphrased val
  fit plain, no t2l                     34.8 / 59.8 / 69.9 / 46.4         31.6 / 55.6 / 66.4 / 42.6
  fit aug,   no t2l          [P2 FINAL] 35.0 / 60.2 / 70.0 / 46.5         31.6 / 56.0 / 66.7 / 42.7
  fit plain, + t2lj                     32.7 / 52.1 / 63.9 / 42.4         32.1 / 50.4 / 60.6 / 41.2
  fit aug,   + t2lj                     32.8 / 51.7 / 63.3 / 42.2         32.2 / 50.1 / 59.9 / 41.0
Text-to-latent as a third source HURTS (-2.2 Hit@1 plain, -8 Hit@5): its train rankings are
in-sample (the joint model was trained on those questions, loss 0.45), so the reranker learns to
trust it far more than it deserves on val. Proper use needs out-of-fold (k-fold) text-to-latent
rankings on train; not done (box stopped). Same leakage exists mildly for the bge_ft2 text features.
P2 FINAL vs P1 on val:  35.0 / 60.2 / 70.0 / 46.5  vs  26.8 / 49.7 / 58.4 / 37.3
P2 FINAL vs P1 on paraphrased val:  31.6 / 56.0 / 66.7 / 42.7  vs  22.5 / 46.1 / 54.3 / 33.3
Phrasing gap: P1 4.3 Hit@1 -> P2 3.4. Real-human expectation from the proxy ratio (~2x): ~28 Hit@1.
Second committed read: NOT made; the user decides. Components to commit if made: retrieve.py
(--llm-parse override, --beta 30 --anchor bge --dump-feats), llm_parse.py on the split's questions
(the only new per-split step; the LLM sees question text only), embed_text2.py --model bgeft2,
predict.py --rerank data/rerank_llm_ancf_bgeft2_aug.json --w 0.5.
5090 box (vast 49992742) stopped 2026-09-06 after pulling logs/p2, data/*.json, models/ (2.6 GB).
Open items: k-fold text-to-latent; re-measure the joint table's link MRR; fusion w 0.45 vs 0.5.

### Does ResonatE do work, or is it a graph walk + a tuned embedder? (2026-09-06, val, same parser
(LLM override) and bge anchors in all three; model_vs_walk.sh, mvw.py)
                               all val (2241)              answer graph-reachable (1273, 56.8%)   not reachable (968)
  graph walk only (no model)   18.9 / 33.3 / 41.5 / 25.8    33.0 / 57.2 / 71.9 / 44.5              0.3 / 1.9 / 1.6 / 1.2
  ResonatE only (no walk)      19.0 / 33.3 / 40.9 / 25.8    31.0 / 53.7 / 64.8 / 41.6              3.1 / 6.5 / 9.4 / 5.1
  both (P2 retrieval)          24.9 / 40.4 / 48.8 / 32.1    42.1 / 67.1 / 78.9 / 53.5              2.2 / 5.3 / 9.3 / 4.0
Reading: alone, the model equals the exact walk (19.0 vs 18.9 Hit@1). Together they are +6 Hit@1 over
either: on the reachable half the walk says WHICH candidates are graph-supported and the model
says in what ORDER (42.1 vs 33.0). The model is not decoration, and it is not the walk in disguise:
they rank the same candidates differently and the sum is what works. The reranker ablation
("z adds nothing") is consistent: the model's ordering enters through the relational-rank feature
(the candidate list is already model-ordered), not through the raw z feature. The 43% of val
queries with no graph-reachable answer are parser failures (wrong entity / type / relation) — both
paths score ~0 there; text ranking and the reranker recover some of them.

## Lever 5 pre-registration (2026-09-06, before any run) — one table for edges and questions
Question: can the entity table be trained on graph edges AND natural-language questions at once,
so that a question is answered by the same readout as a graph query, without losing link quality?
joint_train.py: start from models/p_k12b4_50k.pt (table + operators); each step = one edge batch
(train_prime.py loss: CE over 4096 uniform negatives + lam 0.1 hop-consistency, 2048 edges) + one
question batch (32 questions incl. paraphrases; z_q = cnorm(head(bge_ft2(q))); softmax over all
entities with multiple positives). Adam: table 1e-3, operators 1e-4, encoder 2e-5, head 1e-3;
3 question epochs (~1155 steps). Table and operators trainable in both losses.
Measured, all on val / held-out edges:
  (a) held-out link MRR (500 uniform negatives, same seed as train_prime.py) for: base table
      (P1 logged 0.557 mean), the questions-only joint table (t2lj), and the edges+questions table;
  (b) text-to-latent QA on plain and paraphrased val (all entities; parser-typed) for t2lj vs joint;
  (c) the P2 relational retrieval (retrieve.py, LLM override, beta 30, bge anchors) with the joint
      table vs the base table (24.9 Hit@1) — does the walk+model path keep working on the new table.
Success = link MRR within 0.02 of base AND QA >= t2lj (26.2 / 25.4). Reads: train, val, held-out
edges. test / test-0.1 / human closed.
Lever 5 result (joint_train.py, 3 question epochs = 1158 steps, each with one 2048-edge batch, 292 s):
  held-out link MRR: base table 0.5574; questions-only table (t2lj) 0.5572; edges+questions table 0.5680
  QA, text-to-latent alone:        plain val 26.6 / 37.9 / 35.4 / 32.0    paraphrased 25.4 / 37.6 / 34.2 / 31.1
  (questions-only t2lj, for reference: 26.2 / 37.8 / 35.2 / 31.7          25.4 / 37.2 / 34.0 / 31.0)
Pre-registered success (link MRR within 0.02 of base AND QA >= t2lj): MET. One table now answers
graph queries (link MRR up 0.011 from the extra edge steps) and language questions (26.6 Hit@1 from
the question alone, phrasing gap 1.2) with the same readout. Retrieval check with the joint table
(walk + model path) pending.
Retrieval check with the joint table (retrieve.py, LLM override, bge anchors), val:
  walk + model   base table 24.9 / 40.4 / 48.8 / 32.1   joint table 24.3 / 40.6 / 49.0 / 32.2   (intact)
  model only     base table 19.0 / 33.3 / 40.9 / 25.8   joint table 17.9 / 32.9 / 40.1 / 25.3   (-1.1 Hit@1)

### Out-of-fold fix, results (2026-09-06; oof_chain.sh; 5-fold text-to-latent (questions-only, t2lj
recipe) and 5-fold bge fine-tune give every train question a ranking from a model that never saw it;
reranker fit on those; val features from the full models; LLM-override retrieval, bge_ft2 text)
                                              plain val                      paraphrased val
  reranker without text-to-latent             35.1 / 59.0 / 68.5 / 46.2        31.6 / 54.6 / 64.4 / 42.3
  reranker WITH out-of-fold text-to-latent    42.4 / 67.3 / 75.2 / 53.9        39.5 / 62.5 / 71.5 / 50.5
(rerank.py's internal val eval of the same fit: 43.2 / 67.5 / 75.4 / 54.5 — it uses the train-chosen
RRF w 0.45 from fusion_bgeft2.json; predict.py above used w 0.5. Use 0.45 from here on.)
With honest features, text-to-latent is the largest single lever in P2: +7.3 Hit@1 plain, +7.9
paraphrased, +7.7 MRR; Recall@20 +6.7. The phrasing gap stays ~3 Hit@1. The in-sample version had
shown -2.2: the failure was the fit, not the source. All val; test / test-0.1 / human unopened.
Pending: the same with the JOINT (edges + questions) model's out-of-fold rankings (joint_oof_chain.sh).
Joint (edges + questions) model as the third source, out-of-fold fit (joint_oof_chain.sh; 5 folds,
51-83 s each; RRF w 0.45 from train); end to end through predict.py, val:
                                              plain val                      paraphrased val
  reranker without text-to-latent             35.6 / 60.0 / 69.8 / 46.8        31.9 / 55.6 / 66.0 / 42.8
  + questions-only text-to-latent (t2lj)      42.4 / 67.3 / 75.2 / 53.9 (w .5) 39.5 / 62.5 / 71.5 / 50.5 (w .5)
  + JOINT text-to-latent (p_joint) [P2 FINAL] 43.5 / 67.4 / 75.7 / 54.7        40.6 / 63.3 / 71.9 / 51.3
P2 FINAL pipeline: retrieve.py (base table, beta 30, bge anchors, LLM-override parse) + bge_ft2 text
+ p_joint text-to-latent (t2l_rank.py) + RRF w 0.45 + per-type reranker data/rerank_llm_ancf_bgeft2_pjoint_aug_oof.json.
vs P1 (26.8 / 49.7 / 58.4 / 37.3 plain; 22.5 / 46.1 / 54.3 / 33.3 paraphrased): +16.7 / +18.1 Hit@1.
Phrasing gap 2.9 Hit@1. Second committed read: NOT made (user's call). Box 50038767 idle, kept up.

## Lever 6 pre-registration (2026-09-06, before any run) — push the standalone readout
Standalone = the joint one-table model answering from the question alone (26.6 / 25.4 Hit@1,
R@20 35). Two tuning levers now, the latent parser later:
 (a) more question data + longer: a second paraphrase set of train (paraphrase.py --seed 2), joint
     training 10 question epochs (edges every step as before): tag p_joint10.
 (b) multi-vector queries: the head emits V=4 latent vectors per question, score = max over V of
     Re<z_v, E_t>; same loss. Tags p_jointv4 (3 epochs, one paraphrase set, isolates the effect
     against p_joint) and p_joint10v4 (10 epochs, two paraphrase sets).
Measured on plain and paraphrased val: standalone all-entities Hit@1 / R@20 / MRR, held-out link
MRR. The best by plain-val standalone MRR then gets out-of-fold rankings and the reranker fit, for
the pipeline number. Reads: train, val, held-out edges. test / test-0.1 / human closed.
Lever 6 results (standalone one-table readout, all entities, full val; link MRR on the 5000-edge check):
                                              plain Hit@1 / Hit@5 / R@20 / MRR   paraphrased                      link MRR
  p_joint     3 ep, 1 vector, 1 para set      26.6 / 37.9 / 35.4 / 32.0          25.4 / 37.6 / 34.2 / 31.1        0.568
  p_jointv4   3 ep, 4 vectors, 1 para set     27.8 / 39.6 / 36.6 / 33.5          27.6 / 38.6 / 35.8 / 32.9        0.570   [BEST]
  p_joint10   10 ep, 1 vector, 2 para sets    27.6 / 37.6 / 32.8 / 32.3          27.5 / 37.3 / 32.4 / 32.1        0.567
  p_joint10v4 10 ep, 4 vectors, 2 para sets   27.1 / 38.4 / 34.3 / 32.3          26.6 / 38.1 / 34.0 / 32.0        0.567
Multi-vector (V=4) helps on every metric (+1.2 Hit@1, +1.2 R@20, +1.5 MRR, phrasing gap 0.2).
Longer training with a second paraphrase set does not: Hit@1 flat, Recall@20 down 2-4 (question
loss ~0.1: the head overfits the train answer sets). The recall ceiling (~36) is not the vector
count or the data; it is the single-readout formulation (no exact AND over constraints) — the
latent parser is the next lever. Pipeline test: out-of-fold p_jointv4 rankings + reranker (running).
Pipeline test of the 4-vector model (out-of-fold, same reranker recipe, w 0.45), val:
  third source = p_joint   (1 vector)   plain 43.5 / 67.4 / 75.7 / 54.7   paraphrased 40.6 / 63.3 / 71.9 / 51.3
  third source = p_jointv4 (4 vectors)  plain 43.2 / 68.0 / 76.1 / 54.8   paraphrased 39.9 / 63.7 / 71.9 / 51.0
Tied within noise (Hit@1 -0.3 / -0.7, Hit@5 and R@20 +0.4 to +0.6): the standalone gain of the
4-vector head does not carry into the pipeline, where the exact walk already supplies what the
extra vectors add. P2 FINAL stays the p_joint pipeline. Lever 6 closed; next lever = latent parser.
Box 50038767 left running per the user (they will use it).

## Lever 7 pre-registration (2026-09-06, before any run) — latent parser
Goal: a LEARNED reader of the question into the model's structured query space, replacing the
regex parser + keyword map + 7B parser; execution stays the model's own machinery (operator
composition for the soft score, adjacency walk for the exact AND). Standalone = table + operators
+ one small head + the graph; no text index, no reranker.
latent_parser.py, on the p_joint encoder (CLS, 768-d):
  heads: answer type (10-way softmax); relation operators (36-way multi-label, sigmoid);
         anchors: K=3 latent vectors in the entity space, scored against the FROZEN p_joint table,
         softmax with multiple positives = the anchor entity ids.
  weak labels from train (+ paraphrases): answer type = majority type of the answers; anchor ids =
         exact-name mentions (Parser.mentions) + LLM-parse entity names resolved exactly; operator
         labels = the (r, d) operators (and both ops of a two-hop chain) whose exact walk from an
         anchor reaches an answer.
  execution (retrieve.py --lparse): answer type from the head; anchors = nearest table entities of
         the K vectors above a train-tuned similarity floor, unioned with exact-name mentions
         (string fallback); operator weights 2.0 for predicted ops, 1.0 otherwise; beta 30, walk +
         model score as in P1/P2. All thresholds tuned on train.
Measured on plain and paraphrased val, relational-only (uncovered = miss):
  references: regex parser 24.6 / 19.3; regex + 7B override 24.9 / 20.9; full pipeline 43.5 / 40.6.
  success = latent parser >= 7B-override path on plain AND a smaller phrasing gap than 4.0.
Reads: train, val. test / test-0.1 / human closed. Box: ask the user before using the NL 5090.
Lever 7 result (latent_parser.py on the p_joint encoder; weak labels 12,324 questions, 11,194 with
anchors, 8,364 with a hitting operator, 174 s; training 230 s; anchor floor tuned on train 14.92,
anchor F1 0.81 on train). Relational path on val (walk + model, beta 30), uncovered = miss:
                                        plain val                     paraphrased val          gap
  regex parser                          24.6 / 41.6 / 50.0 / 32.6      19.3 / 34.4 / 42.1 / 26.4   5.3
  regex + 7B override                   24.9 / 40.4 / 48.8 / 32.1      20.9 / 36.9 / 45.5 / 28.5   4.0
  latent parser alone                   24.5 / 40.3 / 48.0 / 31.8      21.4 / 36.6 / 44.3 / 28.5   3.1
  latent parser + bge anchor fallback   25.1 / 42.2 / 50.4 / 33.0      22.7 / 38.8 / 47.1 / 30.3   2.4
Coverage: 95.2% / 92.5% alone, 100% with the fallback. Pre-registered bar (>= 7B path on plain AND
gap < 4.0): met with the fallback (25.1 >= 24.9, gap 2.4); alone, plain is -0.4 (noise) and the gap
is 3.1. A learned head trained in four minutes replaces the hand-written parser, the keyword map
and the 7B model on the relational path, and reads human-style wording better than all of them.
Not yet done: the latent parser inside the full pipeline (text + reranker), and the same on train
for a reranker refit. Box 50038767 left running (user's).
Seeds (2026-09-06, seeds_chain.sh), val:
  joint model (3 ep, 1 vector), standalone all entities, plain Hit@1 / MRR | paraphrased | link MRR
    s0 26.6 / 32.0 | 25.4 / 31.1 | 0.568;  s1 26.9 / 32.4 | 25.7 / 31.4 | 0.569;  s2 26.6 / 32.3 | 26.1 / 31.5 | 0.568
    mean +- sd: plain 26.7 +- 0.2 / 32.2 +- 0.2; paraphrased 25.7 +- 0.4 / 31.3 +- 0.2; link 0.568 +- 0.001
  latent parser + bge fallback, relational path, plain Hit@1 / MRR | paraphrased
    s0 25.1 / 33.0 | 22.7 / 30.3;  s1 25.3 / 33.2 | 22.0 / 29.8;  s2 26.3 / 33.8 | 22.3 / 30.0
    mean +- sd: plain 25.6 +- 0.6 / 33.3 +- 0.4; paraphrased 22.3 +- 0.4 / 30.0 +- 0.3   (7B path: 24.9 / 20.9)
Lever 7b result — latent parser inside the full pipeline (lp_pipe_chain.sh; out-of-fold parses on
train, out-of-fold text + text-to-latent features, reranker refit; w 0.45), val:
                                          plain Hit@1 / Hit@5 / R@20 / MRR     paraphrased
  pipeline with regex + 7B parser (P2 FINAL)   43.5 / 67.4 / 75.7 / 54.7       40.6 / 63.3 / 71.9 / 51.3
  pipeline with the latent parser (no LLM)     41.3 / 67.3 / 75.3 / 53.1       39.0 / 63.6 / 71.7 / 50.4
Removing the 7B model from query time costs 2.2 Hit@1 plain / 1.6 paraphrased and 1.6 / 0.9 MRR;
Hit@5 and Recall@20 are unchanged. On the graph path alone the latent parser was ahead; inside the
pipeline the 7B parse's extra entity names (text-anchored per name) still buy ~2 points of Hit@1.
Two honest pipelines, the user picks: max score (with the 7B) or no-LLM-at-query-time (-2 Hit@1).
Pulled to jinx: models/lp*.pt, data/lparse_*, data/rel_*lp*, logs. Box 50038767 left running.

## P2 committed read — pre-registration (2026-09-06, before the run)
Decision (user): the paper reports both pipelines; the SUBMISSION is the no-LLM pipeline (latent
parser). This is the SECOND read of test / test-0.1 overall and the second of human_generated_eval;
P1's read (28.7 / 28.2 / 20.4 Hit@1) is reported next to it. Nothing is tuned after this read.
Pipeline (frozen; models/lp.pt, models/p_joint*.{pt,_enc}, models/bge_ft2, models/p_k12b4_50k.pt,
data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json, RRF w 0.45), per split S in {test, test-0.1,
human_generated_eval}, reading only the split's question text and ids:
  1. latent_parser.py --predict S --tag lp                       -> data/lparse_S.json
  2. retrieve.py --split S --beta 30 --anchor bge --dump-feats --lparse data/lparse_S.json -> data/rel_S_lp_ancf.json
  3. embed_text2.py --model bgeft2 --reuse-docs --splits S       -> data/text_S_bgeft2.json
  4. t2l_rank.py --tag p_joint_head --model models/p_joint.pt --split S -> data/text_S_pjoint.json
  5. predict.py --split S --rel ... --text ... --t2l ... --w 0.45 --rerank ... --score   (the read)
Dry run of the identical command sequence on val first (expected 41.3 / 67.3 / 75.3 / 53.1).
Splits read: test, test-0.1, human_generated_eval (questions + ids for prediction; answers only
inside predict.py --score). Expected from val and the proxy: synthesized ~41-43 Hit@1, human ~33-37.

### P2 COMMITTED READ (2026-09-06 10:08-10:12, committed_read_p2.sh, no-LLM pipeline; second read)
  dry run val          41.28 / 67.25 / 75.29 / 53.07   (matches the development number exactly)
  test (2801)          41.81 / 68.30 / 74.77 / 53.66    P1 read: 28.7 / 51.9 / 59.9 / 39.1
  test-0.1 (280)       41.79 / 71.07 / 75.90 / 54.31    P1 read: 28.2 / 50.7 / 59.9 / 38.5
  human (98)           30.61 / 53.06 / 60.58 / 41.74    P1 read: 20.4 / 41.8 / 48.6 / 29.9
Leaderboard (published rows, best per column): synthesized full AvaTaR 20.1 / 39.9 / 42.2 / 29.2;
10% best 18.3 / 37.3 / 41.1 / 26.6; human AvaTaR 33.0 / 51.4 / 53.3 / 41.0.
Standing: first on both synthesized splits on every metric (+21.7 Hit@1 full, +23.5 on 10%); on
human: Hit@1 30.6 vs 33.0 (second, behind AvaTaR, ahead of the Claude-3 / GPT-4 rerankers at
28.6), Hit@5 53.1 vs 51.4 (first), R@20 60.6 vs 53.3 (first), MRR 41.7 vs 41.0 (first).
Prediction files: results_p2/eval_results_{test,test-0.1,human_generated_eval}.csv (idx, query_id,
pred_rank top-100). Nothing is tuned after this read.

### Audit after an external review (2026-09-06; eval_check.py, bench.py)
Wording: "no language model at query time" -> "no generative language model at query time"; two
110M transformer encoders do run (text ranker; encoder under the parser head and the question
readout). Measured (bench.py, RTX 5090, batch 1, 300 val questions): total median 32.0 ms, p90
48.8 ms (parse 2.8, walk+model 21.7, text 4.2, readout 2.7, fusion+rerank 0.3); GPU memory 1.88 GiB
loaded / 1.92 GiB peak; batched encoders + readout < 0.5 ms per question; the committed test read
(2,801 questions, all stages, model loading) took 114 s wall-clock.
Metrics: metrics.py truncates at the top-100 (answer outside -> reciprocal rank 0); the official
Evaluator (top-100 with scores -i, all others tied below) can add at most 1/101 per such answer.
eval_check.py compares both on the val predictions and rescores the three committed prediction
files with the official Evaluator plus 2,000-sample bootstrap CIs over questions. This re-reads
the test answers to score the SAME frozen files a second time; no model or setting changed. Note:
torchmetrics 1.9.0 returns 0 for retrieval_reciprocal_rank / retrieval_recall on valid inputs;
the check pins torchmetrics 1.4.0 (sanity cases correct). Results appended below when done.
Chain scripts moved into scripts/ (repo-relative paths); release v2-no-llm carries SHA-256 sums.
Audit results: metrics.py vs official Evaluator on val — Hit@1 / Hit@5 / R@20 identical per query;
MRR 0.5307 both (max per-query diff 0.0016). Official rescoring of the committed files (2,000-sample
bootstrap 95% CIs over questions): test 41.81 [40.0, 43.7] / 68.30 [66.7, 70.1] / 74.77 [73.3, 76.3]
/ 53.66 [52.2, 55.3]; test-0.1 41.79 [36.1, 47.5] / 71.07 [65.7, 76.4] / 75.90 [71.2, 80.5] / 54.31
[49.5, 59.0]; human 30.61 [21.4, 39.8] / 53.06 [42.9, 63.3] / 60.58 [51.6, 69.4] / 41.74 [33.9, 50.3].
On the human set no metric difference to AvaTaR is outside the interval except Recall@20 (+7.2).

## Lever 8 pre-registration (2026-09-06, before any run) — is "one table serves both" specific to ResonatE?
Reviewer's test: train RotatE on the same graph, freeze its embeddings, train the identical STaRK
projector on top, compare with ResonatE under identical settings.
Setup (rotate_prime.py, t2l_generic.py):
  RotatE: complex entity table (N, M=144) = 288 real numbers per entity, the same width as ResonatE's
  table; relation r = phase vector, reverse direction = conjugate rotation; score gamma - sum_m |h_m
  e^{i theta_m} - t_m| (RotatE's L1-of-moduli distance), gamma 12; trained with the SAME regime as
  train_prime.py (50k steps, batch 2048, 4096 uniform negatives, cross-entropy, Adam, cosine; the same
  2% held-out slice for link MRR).
  Projector: the same T2L module (bge_ft2 encoder, linear head to M complex dims), same loss (softmax
  over all entities, multiple positives), same data (train + paraphrases), same epochs and lrs,
  frozen table; readout = each model's OWN scoring function of a query vector against its table:
  ResonatE Re<z, E_t> * exp(s) with z unit-norm (as in text2latent.py); RotatE gamma - L1 distance
  (z unconstrained). A dot-product readout is also run for RotatE as a control.
  Joint variant for both: table unfrozen, edge loss + question loss (as joint_train.py), 3 epochs.
Measured on val: standalone QA plain / paraphrased (Hit@1, Hit@5, R@20, MRR, all entities); held-out
link MRR before and after joint training. Same seed, same epochs, three seeds if the first differs
by less than 1 Hit@1. Reads: train, val, held-out edges. Test/human closed.
Prediction written down first: RotatE's frozen table will work about as well as ResonatE's frozen
table under the projector (both are geometric tables of the same width); the joint variant is where
they may differ, because ResonatE's unit-norm rows and inner-product readout are what the question
loss trains directly. If RotatE matches on both, the claim in the paper becomes "a KG table can
serve both" rather than "this model's table serves both", and the paper will say so.
Lever 8 amendment (before any result): RotatE's L1-of-moduli distance costs ~15x more per step here
(2000 steps in 174 s vs ~5 min for the whole 50k with a matmul); the run uses RotatE's L2 variant
(distance = ||h e^{i theta} - t||_2, computed through a matmul expansion), same gamma 12, same regime.
Lever 8 result (scripts/lever8_chain.sh; RotatE L2 variant, M=144 complex, 50k steps, same regime,
188 s; identical projector = t2l_generic.py for BOTH tables, 3 epochs, train + paraphrases), val:
                                               plain Hit@1 / Hit@5 / R@20 / MRR   paraphrased                    link MRR (before -> after)
  ResonatE frozen, inner-product readout       19.7 / 30.7 / 31.8 / 25.2          19.3 / 29.8 / 30.6 / 24.6      0.557
  RotatE   frozen, its L2-distance readout      10.1 / 18.6 / 21.7 / 14.6           9.9 / 18.5 / 21.3 / 14.3      0.523
  RotatE   frozen, inner-product readout (ctl)  13.3 / 21.6 / 22.5 / 17.5          12.3 / 20.4 / 21.4 / 16.7      0.523
  ResonatE joint (table unfrozen, edges + q)    26.9 / 37.9 / 35.4 / 32.2          25.4 / 37.4 / 34.2 / 31.1      0.557 -> 0.570
  RotatE   joint (table unfrozen, edges + q)    12.7 / 24.8 / 25.8 / 18.3          12.5 / 24.5 / 25.7 / 18.1      0.523 -> 0.516
The written prediction (frozen tables tie) was WRONG: under the identical projector RotatE's table
takes the projection at about half the Hit@1, with either readout, and joint training helps it
little (+2.6) while costing it link MRR (-0.007); ResonatE gains +7 and +0.013. Caveats, stated
plainly: single seed; RotatE's L2 variant with my initialisation and lr 1e-3 (not tuned; its link
MRR is 0.034 below ResonatE's under the same regime, so part of the gap may be an under-tuned
control); one dataset. A RotatE tuning sweep (lr x gamma, chosen by held-out link MRR, never by QA)
follows before the claim is written into the paper.
Lever 8 sweep (scripts/lever8_sweep.sh; chosen by held-out LINK MRR only): lr 3e-4 -> 0.462; lr 3e-3 ->
0.549; gamma 6 -> 0.522; gamma 24 -> 0.523 (default lr 1e-3 gamma 12: 0.523). Best RotatE table
(lr 3e-3, link MRR 0.549, i.e. 0.008 below ResonatE's 0.557) under the identical frozen projector:
  distance readout       15.5 / 25.4 / 27.2 / 20.6 plain    14.9 / 25.4 / 26.8 / 20.0 paraphrased
  inner-product readout  15.8 / 27.6 / 28.0 / 21.6 plain    15.0 / 27.2 / 27.1 / 20.7 paraphrased
  (ResonatE frozen:      19.7 / 30.7 / 31.8 / 25.2 plain    19.3 / 29.8 / 30.6 / 24.6)
So much of the first gap WAS the under-tuned control: at near-equal link quality the frozen gap is
~4 Hit@1 (20% relative), not half. Joint on the tuned RotatE table: pending (below).
