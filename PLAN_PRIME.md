# PLAN_PRIME — STaRK-Prime probe (separate from the OGB work; own numbering)

> **Submission status — 7 September 2026 (confirmed by the user): nothing has
> been filed for BioKG, WikiKG2, or STaRK-Prime.** Entries are proposed only.
> Historical references below to “submitted” or “filed” describe intended
> candidates, not completed filings.

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
Joint on the tuned RotatE table (lr 3e-3; 3 epochs, edges + questions, same as ResonatE's joint):
  distance readout       16.7 / 29.0 / 30.7 / 22.6 plain    16.5 / 28.0 / 30.4 / 22.2 paraphrased   link 0.549 -> 0.550
  inner-product readout  21.2 / 33.8 / 32.4 / 27.2 plain    20.5 / 32.7 / 30.9 / 26.3 paraphrased   link 0.549 -> 0.552
  (ResonatE joint:       26.9 / 37.9 / 35.4 / 32.2 plain    25.4 / 37.4 / 34.2 / 31.1 paraphrased   link 0.557 -> 0.570)
LEVER 8 CONCLUSION (one seed each, one dataset, RotatE L2 variant tuned by link MRR): with the
identical projector, ResonatE's table takes language better than a RotatE table of the same width
and near-equal link quality: frozen 19.7 vs 15.8 Hit@1 (+3.9), joint 26.9 vs 21.2 (+5.7), and joint
training raises ResonatE's link MRR by 0.013 against 0.003 for RotatE. The first-pass gap of 2x was
mostly an under-tuned control and is retracted; the tuned gap of 4-6 Hit@1 stands. The claim in the
paper is therefore "this table takes a language projection better than a RotatE table under the same
recipe", not "only this table can". RotatE's own distance readout is worse than an inner product on
its own table for this use (16.7 vs 21.2 joint), so the readout matters too.

## Lever 9 pre-registration (2026-09-06, before any run) — an AND in the space
Today score_query sums, over the parsed mentions, the best (max over chains) standardised operator
score: OR-ish. The AND readout keeps every cost the same (one matmul with the k mention vectors, then
a reduction): score(t) = agg_i max_chain z_i(t), with agg in
  sum        (current),
  min        (exact intersection of caps on the sphere),
  softmin    (-tau * logsumexp(-z_i / tau), tau in {0.5, 1, 2}),
  logsig     (sum_i log sigmoid(z_i - c), c in {0, 1}: a calibrated product-of-probabilities AND).
Single-mention questions are unaffected by construction. Measured on val (plain + paraphrased),
MODEL ONLY (beta 0, no walk), LLM-override parse + bge anchors as in the walk-vs-model test, all
queries; also broken down by number of mentions (1 vs >= 2), where the AND can act. References:
model-only sum 19.0 / 19.3 plain / (para from rel_val_para_llm...), walk-only 18.9, both 24.9.
Then the best AND with the walk (beta 30), to see whether it also helps the full relational path.
Success = model-only AND > walk-only (18.9) on plain val. No parameter is fit; tau / c are a
2-3 point grid reported in full. Reads: val. Test / human closed.
Lever 9, first pass (AND over MENTIONS): min collapses on >= 2-mention questions (Hit@1 18.7 -> 8.6,
plain): a name that resolves to several entities contributes one mention per resolution, so the min
demanded all resolutions of one name at once. Corrected before the remaining readouts ran: OR over a
name's resolutions and chains (max), AND across distinct names. The breakdown is now by number of
distinct names. sum result is unchanged by the fix (19.0 / 16.6).
Lever 9, corrected min (AND across names): model only, plain val 13.3 / 22.5 / 27.1 / 17.7 (sum: 19.0);
>= 2-name questions 9.6 Hit@1 vs 18.9 with the sum; 1-name questions identical by construction.
An exact AND over the parsed names hurts: the parsed constraint set is noisy (spurious name
matches, LLM entity names that are context rather than constraints), and one wrong constraint
vetoes the answer. The walk does NOT use a hard AND either: beta * exact is a COUNT of satisfied
constraints. Added agg "count" (soft count of satisfied caps, 30 * sum_i sigmoid((z_i - c)/0.5) +
sum_i z_i; c in {1,2,3}) — the walk's rule, in the space — queued as lever 9b.
Lever 9 result (all val; model only unless noted; AND across names, OR within a name):
  readout                     plain all / >= 2 names / 1 name      paraphrased all
  sum (current)               19.0 / 18.9 / 19.0                     16.6
  min (exact AND)             13.3 /  9.6 / 19.0                     12.3
  softmin tau 0.5 / 1 / 2     13.8 / 15.0 / 15.9                     12.7 / 13.4 / 14.1
  logsig c 0 / 1              15.0 / 14.9                            13.4 / 13.4
  soft count c 1 / 2 / 3      15.8 / 15.4 / 16.5  (>= 2 names: 13.8 / 13.0 / 14.8)   13.9 / 14.0 / 15.2
  with the walk (beta 30): sum 24.9, min 22.8, softmin 23.4 (paraphrased 20.9 / 19.5 / 19.8)
CONCLUSION: no inference-time AND readout beats the sum on this constraint set; every AND-like
combination loses on multi-name questions and is neutral on single-name ones by construction. Two
causes, separable: (i) the single-hop table was never trained for conjunctions, so per-constraint
scores are not calibrated as memberships; (ii) the parsed constraint set is noisy and any AND
lets a wrong constraint veto. The walk's advantage is not an AND either — it is exact membership
plus a count. Next (lever 10, not yet pre-registered in detail): TRAIN the joint model on sampled
conjunctive queries (2-3 anchors + chains, answers = exact intersections from the training graph)
with a soft-AND readout in the loss; judge the geometry on a synthetic exact-AND set (no parser
noise) and on val with the same parse; add a "is this a constraint" head to the parser.

## Lever 10 pre-registration (2026-09-06, before any run) — train the AND
Hypothesis: a table trained on conjunctive queries with a soft-AND readout can intersect in the space;
lever 9's failure was a query-time AND on a single-hop-trained table plus noisy parsed constraints.
Data (conj_sample.py, TRAIN edges only): conjunctions of 2 or 3 constraints (anchor a_i, chain c_i
from the type signatures, 1- or 2-hop) with a common answer type; answers = exact intersection of
the walks, kept if 1 <= |intersection| <= 50 and every single constraint's set is > 2x larger than
the intersection (so the AND matters). 60k train conjunctions, 2k held-out conjunctions (disjoint
anchors) as a SYNTHETIC exact-AND validation set with no parser in the loop.
Model (and_train.py): start from models/p_joint.pt (edges + questions); each step = one edge batch
(train_prime loss) + one conjunction batch (per constraint z_i = model.out(hop(E[a_i], c_i)),
s_i(t) = Re<z_i, E_t>*exp(tau); combined = -T * logsumexp(-s_i / T) over constraints (soft-AND, T=1);
softmax over all entities with the intersection as positives) + one question batch (as joint_train,
to keep the language readout); table and operators trainable; 3000 steps, batch 256 conjunctions.
Measured:
  (a) synthetic exact-AND val (2k): Hit@1 / R@20 / MRR for base table {sum, min, softmin} and the
      AND-trained table {sum, min, softmin}; the exact walk is the ceiling (1.0 by construction);
  (b) STaRK val, model only (beta 0), same LLM-override parse + bge anchors: AND-trained table with
      sum / min / softmin vs base-table sum 19.0 and walk-only 18.9 (plain; paraphrased too);
  (c) held-out link MRR before / after (must stay within 0.01 of 0.568); QA readout re-fit frozen.
Prediction, written first: (a) will improve a lot (the geometry can learn caps: >= 0.5 Hit@1 with
softmin vs < 0.2 for the base table); (b) will improve little or not at all with min, because the
parsed constraints stay noisy — if (b) beats 19.0 with softmin the constraint noise is smaller
than lever 9 suggested. Reads: train edges/questions, val, held-out edges. Test / human closed.
Lever 10 result (30,000 train / 315 val conjunctions, 2.2 constraints, median 2 answers; 1000 steps
from p_joint, soft-AND loss T=1, 457 s). Synthetic exact-AND val (typed), Hit@1 / Hit@5 / R@20 / MRR:
  base table   sum 16.5 / 31.8 / 28.4 / 23.7    min 11.1 / 25.4 / 20.2 / 18.1    softmin 12.7 / 25.4 / 21.6 / 19.1
  AND-trained  sum 32.7 / 51.1 / 47.5 / 41.5    min 28.6 / 44.8 / 40.5 / 36.7    softmin 27.9 / 45.4 / 42.2 / 37.0
  link MRR 0.568 -> 0.568 (unchanged). Train conjunction loss 3.1 -> 1.35 while held-out plateaued
  from step 400: memorisation of the training conjunctions.
GATE FAILED (softmin 27.9 < 50). Stage 3 (STaRK val) not run, as pre-registered.
Reading: the conjunction loss doubles every readout — the caps get sharper — but the sum stays
ahead of min / softmin on the trained table too, so the table did not acquire an intersection
operator; it acquired tighter constraints. The prediction ("(a) will improve a lot, >= 0.5") was
wrong on the level (0.28) and right on the direction. The exact walk is 100 on this set.
Not tried: longer / colder (T 0.25) training, learned per-constraint widths (Query2Box on the
sphere), a larger conjunction set. Recorded as the second negative result on the AND.
Post-read note (2026-09-06, ask.py demo): Parser.by_name skips names shorter than four characters
(min_len=4), so three-letter gene symbols (GCK, TTR as an anchor, ...) are never anchored by exact
match; the gene-symbol regex finds them but the lookup fails. Found while building the CLI demo,
AFTER the committed read. Left unchanged in the submitted pipeline (frozen); to be fixed in any
future P3, with the expected effect measured on val first.

## P3 pre-registration (2026-09-06, before any run) — one bug fix, then a third read
Bug (found building ask.py, after the second read): Parser.by_name drops names shorter than four
characters, so 2-3 character gene/protein symbols (GCK, TTR, ...) are never anchored by exact match,
although the uppercase-symbol regex finds them. Fix: a separate by_symbol dictionary for 2-3
character gene/protein names, used only through the uppercase-symbol rule (retrieve.py; the old
behaviour is kept behind --legacy-names so the P1/P2 reads stay reproducible). NOTHING ELSE changes.
Plan: (1) count how many val questions mention such a symbol; (2) relational path on val, fixed vs
frozen (25.1 / 22.7 plain / paraphrased with the latent parser + bge fallback); (3) if it helps,
rebuild the latent-parser weak labels and folds, the train/val retrievals and the reranker with the
fix, score val plain / paraphrased; (4) one committed read of test / test-0.1 / human_generated_eval
— the THIRD read of the test splits overall, reported next to the first two. Bar: val Hit@1 not
below P2's 41.3 (a fix must not hurt) and a gain on the questions that mention a short symbol.
Reads: train (fitting), val (all decisions). Test / human closed until step 4.
P3 step 1-2 results (jinx, GTX 1080 Ti): 95 / 2,241 val questions (4.2%) and 203 / 6,162 train
questions mention a 2-3 character gene symbol; 663 such symbols in the graph. Relational path on
val with the fix (latent parser + bge fallback, beta 30), fixed vs frozen:
  plain        26.1 / 43.2 / 51.4 / 34.0   vs  25.1 / 42.2 / 50.4 / 33.0   (+1.0 Hit@1)
  paraphrased  23.5 / 39.8 / 48.2 / 31.1   vs  22.7 / 38.8 / 47.1 / 30.3   (+0.8 Hit@1)
Helps, hurts nothing. Step 3 launched on jinx: weak labels rebuilt, latent parser + 5 folds retrained,
train/val retrievals, reranker refit, val scoring (scripts/lp_chain.sh then scripts/lp_pipe_chain.sh).

### Proxy error analysis (2026-09-06, val only; the human set stays closed)
P2 pipeline, val: 925 plain-wording Hit@1 hits; 190 of them (20.5%) become misses on the
paraphrase, while 138 paraphrase hits are plain misses (net -52 = the -2.3 Hit@1 gap; the churn
is 4x the net). Why the 190 are lost:
   122  64%  an anchor read in plain wording is not read in the paraphrase
                of which  90 (74%) a REAL anchor that reached the answer in the graph — aliases
                          (SLCO1B3 -> OATP1B3), descriptions for names ("low potassium levels" for
                          hypokalemia), shortened pathway names; 23 (19%) only spurious name matches
                          (generic words such as "blood", "growth", "cancer", "Disease"); 9 (7%)
                          the paraphrase itself is corrupted (the 7B model switched to Chinese
                          mid-sentence: 30 / 2,241 val and 93 / 6,162 train paraphrases, 1.3-1.5%)
    46  24%  answer graph-supported in both, lost in ranking (reranker / fusion)
    12   6%  answer only in the text / readout lists: the walk missed it (relation read differently)
     9   5%  wrong answer type
     1   1%  in the walk's candidates but unsupported (negation / relation)
Conclusion: entity resolution under rewording is the dominant failure (aliases, descriptions,
shortened names), ranking second; relations, negation and answer type are minor. A P4 would be
(i) alias / synonym resolution for anchors (PrimeKG node text carries synonyms; the latent anchor
head could be trained on alias mentions), (ii) a parse-confidence feature in the reranker,
(iii) cleaner and harder paraphrases (filter CJK; several styles per question).
Two-reading test (val, P2 pipeline; plain reading vs the 7B paraphrase reading of the same question):
  plain 41.3 / 67.2 / 75.3 / 53.1; paraphrase 39.0 / 63.6 / 71.7 / 50.4; RRF of both 41.2; RRF 2:1 41.1;
  fallback (plain unless no graph-supported candidate, then paraphrase) 41.7 / 67.6 / 75.6 / 53.5;
  oracle best-of-two per question 47.4 / 72.3 / 77.9 / 58.5.
Fusing two readings does not help and the fallback adds 0.4; but the oracle upper bound is +6 Hit@1:
the readings disagree on many questions and SELECTING the right one per question is worth six
points. That needs a confidence signal for a reading (a selector), and, at query time, a second
reading that does not require a generative model (deterministic variants: with / without LLM-style
anchors, alternative answer types, alias-expanded text) — a P4 candidate; not tried.
P3 step 3 result (box 50098239; parser retrained with the fix, 5 folds, retrievals, reranker refit
on out-of-fold features; scripts/lp_chain.sh + scripts/lp_pipe_chain.sh):
                       plain val                       paraphrased val
  P2 (frozen, read)    41.3 / 67.3 / 75.3 / 53.1        39.0 / 63.6 / 71.7 / 50.4
  P3 (fix only)        42.3 / 68.3 / 75.5 / 53.9        39.7 / 63.9 / 71.8 / 51.1
Bar met (>= 41.3 plain; gain where short symbols occur). Step 4 (the third committed read) needs the
user's go: it is one frozen pipeline, run once per split, dry run on val first (expected 42.26 /
68.32 / 75.48 / 53.93), reported next to the P1 and P2 reads.
P3 step 4 — committed read authorised by the user on 2026-09-06 ("if we fix a bug is ok"). Pipeline
frozen as above (models/lp.pt = the P3 parser, data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json = the
P3 refit, everything else identical to P2). scripts/committed_read_p3.sh: dry run on val must print
42.26 / 68.32 / 75.48 / 53.93, then test, test-0.1, human_generated_eval once each -> results_p3/.
This is the THIRD read of the test splits; P3 is submitted whatever it says.
P3 read: EXECUTED on the box (2026-09-06, scripts/committed_read_p3.sh) but NOT OPENED, at the
user's request ("do not open, I really want a clean win"). The files stay sealed in results_p3/ on
box 50098239 and the status log is not read. It still counts as a read of the test splits (the
third) and is disclosed as such. Development continues on val only (P4); the sealed P3 files are
either reported next to the final read or discarded unopened, never used to decide anything.

## P4 pre-registration (2026-09-07, before any run) — val only; one read at the very end
From the proxy error analysis: entity resolution under rewording is the dominant loss (aliases,
full names, descriptions), ranking second. Levers, each measured on plain and paraphrased val:
 A. Alias-aware anchoring. The node text carries, for genes, an alias list (14,213 genes) and the
    full gene name ("glucokinase" for GCK); build data/aliases.json {alias -> ids} from them
    (build_aliases.py; aliases of length >= 3, at most 3 ids per alias, an alias never overrides an
    exact name of a different entity) and let the parser resolve aliases like names. Expected:
    +1-2 Hit@1 on the relational path, more on paraphrased than plain.
 B. Reading selector: several deterministic readings per question, a small classifier fit on train
    + paraphrased train (out-of-fold) choosing among them from parse-confidence features; oracle
    best-of-readings on val measured first, the selector built only if that bound is worth it.
 C. Cleaner and harder paraphrases: drop the 1.3-1.5% CJK-corrupted rewrites; add a terse /
    clinical style; refit the embedder and reranker on them.
Reads: train (fitting), val (all decisions). test / test-0.1 / human_generated_eval: closed; one
committed read of the final P4 pipeline at the end, the fourth read overall, reported next to P1,
P2 and the sealed P3.
P4 lever A, first pass (aliases matched like names, case-insensitive, any alias >= 3 chars): HURT —
val relational path 25.9 / 42.5 / 50.7 / 33.7 plain, 20.8 / 36.7 / 46.7 / 28.4 paraphrased (P3: 26.8 /
22.7): gene alias lists contain ordinary words, and case-insensitive matching turned them into
spurious anchors. Second pass (before any other change): alias SYMBOLS (50,516) match only as
uppercase tokens in the question; FULL GENE NAMES (27,430, >= 5 chars) match like names.
P4 lever A, second pass (symbols as uppercase tokens only; full gene names like names): val relational
path 26.6 / 42.8 / 50.9 / 34.2 plain, 22.5 / 38.6 / 47.6 / 30.1 paraphrased vs P3 26.8 / 42.9 / 51.0 /
34.3 and 22.7 / 38.9 / 47.8 / 30.4: -0.2 everywhere, noise level. LEVER A DROPPED: the alias cases in
the proxy are too few to move the total, and every extra match source adds a little noise. Kept in
the code behind --aliases (off by default). The parser stays as in P3.
Lever B step 1 result (val, P3 pipeline, four deterministic readings; scripts/lever_b_readings.sh,
lever_b_oracle.py):
                                          plain Hit@1 / Hit@5 / R@20 / MRR   uniquely best   paraphrased Hit@1   uniquely best
  r1 learned anchors + exact names (P3)   42.3 / 68.3 / 75.5 / 53.9              37             39.7                 54
  r2 exact names only, pattern type       42.3 / 66.9 / 74.3 / 53.5             164             38.7                164
  r3 learned anchors, floor x0.8          39.9 / 65.8 / 73.3 / 51.5             119             37.3                128
  r4 second-best answer type              20.8 / 40.7 / 47.4 / 30.1             204             19.8                208
  ORACLE best of four                     49.8 / 73.6 / 78.1 / 60.5                             47.0 / 70.8 / 75.2 / 57.7
  top-1 agreement r1-r2 87%, r1-r3 83%, r1-r4 31%
Bar (oracle > 45 on plain) MET: +7.5 Hit@1 of headroom from readings that need no generative model.
r4 is uniquely best on 204 questions: when the answer-type head is wrong (~9%), its second choice is
often right — a selector that detects a wrong type is the largest single piece. Step 2: build the
selector (fit on train + paraphrased train with out-of-fold parses; features of the reading only:
type probabilities, anchor similarities, number of names, walk support, reranker top score and
margin; label = the reading with the best MRR). Val decides.
Lever B step 2, first selectors (features of the reading only + cross-reading agreement; fit on
train + paraphrased train; val):
  linear listwise     plain 43.2 / 67.7 / 75.8 / 54.4   paraphrased 40.6 / 63.5 / 72.1 / 51.4   (P3: 42.3 / 39.7; oracle 49.8 / 47.0)
  MLP (64-32, early   plain 43.4 / 68.3 / 75.9 / 54.6   paraphrased 40.7 / 64.3 / 71.9 / 51.6
   stop on a train slice)
+1.1 / +1.0 Hit@1: the judge captures ~15% of the headroom. It rarely picks the second answer type
(106 of 2,241) although that reading is uniquely best on 204: the type head's own probabilities do
not reveal when it is wrong. Next feature: an independent TYPE VOTE from the parser-free sources
(the type distribution of the question readout's top-10 and the text ranker's top-10) and whether
each reading's top-1 appears in those lists.
With the parser-free type vote (types of the text ranker's and the readout's top-10; top-1 membership):
  linear   plain 43.5 / 68.1 / 76.0 / 54.7   paraphrased 40.8 / 64.1 / 72.1 / 51.7
  MLP      plain 44.4 / 68.1 / 75.3 / 55.1   paraphrased 40.7 / 64.2 / 71.9 / 51.7     (P3: 42.3 / 39.7)
+2.1 plain, +1.0 paraphrased Hit@1; picks the second type 145 times (of 204 where it is uniquely best).
Soft mixture (RRF of the four readings weighted by the selector's softmax): plain 39.3 (T=1) / 41.5
(T=0.5), paraphrased 36.9 / 38.8 — WORSE than either the P3 reading or hard selection: mixing dilutes
a right reading with wrong ones. Hard selection stays. LEVER B CANDIDATE: P3 + the MLP selector over
four deterministic readings: val 44.4 / 68.1 / 75.3 / 55.1 plain, 40.7 / 64.2 / 71.9 / 51.7
paraphrased (P3: 42.3 / 39.7). models/selectorB_mlp.pt; features fit on train + paraphrased train.
Query-time cost: four retrievals instead of one (~4x the walk+model stage, still no generative model).
Lever C (2026-09-07, before any run): (1) drop the CJK-corrupted rewrites from the paraphrase sets
(paraphrase.py now discards them at generation); (2) a second, terse "search-box" style of train
paraphrases (abbreviations, dropped function words, descriptions for some names), English only;
(3) refit the text embedder on plain + cleaned natural + terse (bge_ft3), 5 out-of-fold folds for
the reranker's text features, re-embed; (4) refit the reranker, re-score the four readings, refit the
selector. The augmentation set for the parser and the reranker's paraphrased groups stays
para_train.json (unchanged) so only the embedder changes. Val (plain / paraphrased, the SAME proxy
files as before) decides against P3 + selector = 44.4 / 40.7.
Lever C result (val): text ranker alone bge_ft3 22.0 / 44.0 / 51.1 / 32.3 plain, 20.7 / 41.6 / 48.6 /
30.5 paraphrased (bge_ft2: 21.9 / 43.0 / 50.6 / 31.9 and 19.6 / 41.3 / 48.1 / 29.6): small gains alone.
Single reading with the refit reranker: 42.8 / 67.6 / 75.5 / 54.1 plain (P3 42.3 / 68.3 / 75.5 / 53.9),
39.6 paraphrased (39.7). With the selector, three seeds of the MLP each:
  B  (bge_ft2)  plain 44.4 / 44.2 / 43.8 (mean 44.1)   paraphrased 40.7 / 40.5 / 40.6 (mean 40.6)
  C  (bge_ft3)  plain 43.7 / 43.6 / 43.5 (mean 43.6)   paraphrased 40.2 / 40.4 / 40.3 (mean 40.3)
LEVER C DROPPED: the embedder's own gain does not survive the reranker + selector (-0.5 / -0.3,
consistent across seeds). Terse paraphrases and the CJK filter stay available (data/para_train_terse.json,
data/para_train_clean.json) for future work.
P4 FINAL CANDIDATE = P3 + the reading selector on the bge_ft2 features (models/selectorB_mlp.pt, seed 0):
val 44.4 / 68.1 / 75.3 / 55.1 plain, 40.7 / 64.2 / 71.9 / 51.7 paraphrased. Selector seed variance
~0.3 Hit@1. Awaiting the user's go for the one committed read (the fourth overall; P3's stays sealed).
P4 read script verified (2026-09-07 01:12-01:20, box 50098239, VAL_ONLY): from scratch on val it
reproduces 44.36 / 68.14 / 75.32 / 55.14 with the same reading picks (699 / 874 / 523 / 145) as the
development run; eight minutes per split. Awaiting the user's go for test, test-0.1, human.
P4 committed read AUTHORISED by the user (2026-09-07, "ok do it"). Pipeline frozen: P3 parser (models/lp.pt),
four deterministic readings, bge_ft2 text ranker, p_joint readout, reranker
data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json (w 0.45), selector models/selectorB_mlp.pt (seed 0).
scripts/committed_read_p4.sh: val check, then test, test-0.1, human_generated_eval once each. Fourth
read overall (P1, P2 reported; P3 sealed). Submitted whatever it says.

### P4 COMMITTED READ (2026-09-07 01:26-01:50, box 50098239, scripts/committed_read_p4.sh; fourth read overall)
  val check            44.36 / 68.14 / 75.32 / 55.14  (matches development exactly, same picks)
  test (2801)          43.70 / 69.33 / 75.49 / 55.21    P2 read: 41.81 / 68.30 / 74.77 / 53.66    P1: 28.7 / 51.9 / 59.9 / 39.1
  test-0.1 (280)       41.79 / 72.50 / 76.47 / 54.23    P2 read: 41.79 / 71.07 / 75.90 / 54.31    P1: 28.2 / 50.7 / 59.9 / 38.5
  human (98)           25.51 / 56.12 / 61.29 / 39.18    P2 read: 30.61 / 53.06 / 60.58 / 41.74    P1: 20.4 / 41.8 / 48.6 / 29.9
  selector picks: test [870, 1110, 644, 177]; test-0.1 [77, 118, 65, 20]; human [26, 42, 26, 4]
Reading: synthesized full +1.9 Hit@1 as val predicted; 10% subset flat on Hit@1, up on the rest;
HUMAN Hit@1 DOWN 5.1 points = 5 of 98 questions (P2's 95% CI on this number was [21.4, 39.8], so the
change is inside the noise of the set), while Hit@5 and Recall@20 went up (+3.1, +0.7) and MRR down
2.6. On the human set the selector chose the exact-names reading most (42 of 98) — a reading whose
strength is template wording. P3's read stays sealed. Per the pre-registration P4 is the submission
whatever it says; choosing P2 over P4 on the basis of these numbers would be a test-informed choice
and is recorded here as such if the user takes it.

### P3 read UNSEALED (2026-09-07, at the user's decision to treat P3 and P4 as tested hypotheses)
  P3 (bug fix only)   test 43.09 / 68.80 / 75.50 / 54.73   test-0.1 41.79 / 72.50 / 77.83 / 54.34   human 28.57 / 53.06 / 61.85 / 40.67
  P2 (read before)    test 41.81 / 68.30 / 74.77 / 53.66   test-0.1 41.79 / 71.07 / 75.90 / 54.31   human 30.61 / 53.06 / 60.58 / 41.74
  P4 (P3 + selector)  test 43.70 / 69.33 / 75.49 / 55.21   test-0.1 41.79 / 72.50 / 76.47 / 54.23   human 25.51 / 56.12 / 61.29 / 39.18
Attribution on the human set (98 questions): the bug fix moved Hit@1 by -2 questions and Recall@20
up 1.3; the selector moved Hit@1 by a further -3 questions. Both inside the noise of the set; the
selector's is the larger and is consistent with its choice of the template-strength reading.
### Decision rule (the user, 2026-09-07): publish the best model and architecture, not the best score.
Hypotheses tested on the test splits after P2: P3 = a correctness fix (kept: it belongs to the model
and helps on the synthesized splits, +1.3 Hit@1); P4 = a learned reading selector (dropped: a bolt
that fits the synthesized distribution and does not transfer to human questions). SUBMISSION = P3.
All four reads are reported side by side; P4's files are kept (results_p4/) and not submitted.
Next: the same architecture on STaRK-Amazon and STaRK-MAG with per-dataset learned bolts, and an
out-of-distribution validation protocol (held-out question templates) before any further lever.
Next (planned 2026-09-07, to run tomorrow): P5 = latent-confirmed anchors — every string-matched
mention's candidate entities are scored by the question vector (anchor head / readout) against the
entity rows and kept only above a floor relative to the best (floor set on train); a general
disambiguation rule for name collisions (MS the abbreviation vs MS the gene, Aspirin drug vs exposure,
insulin vs INS). Measured on val plain, paraphrased AND terse paraphrases (the ones with abbreviations);
kept only if it helps the terse set without hurting the others. No read until the OOD protocol says so.
Box 50098239 stopped at the user's request after pulling everything.

## P5 pre-registration (2026-09-07, before any run) — latent-confirmed anchors, under an OOD protocol
Rule: a string in a question is a set of candidate entities; a candidate is kept only if the
question's own vector points at it. Implementation (retrieve.py --confirm): for every string-matched
mention (names, short symbols), score each candidate entity id by the latent parser's anchor vectors
(max over the K=3 vectors of Re<z_k, E_id> * scale, the same score the anchor head is trained on);
keep candidates with score >= max(REL * best score among that name's candidates, ABS * the trained
anchor floor); drop the rest (a name whose candidates all fall below is dropped: no anchor from it).
REL in {0.7, 0.8, 0.9}, ABS in {0.5, 0.75, 1.0}: chosen on TRAIN only (relational path on 2,000 train
questions, plain + terse), by terse Hit@1 subject to plain not dropping.
OOD protocol for the decision (val only): plain val, natural paraphrases (as before), and TERSE
paraphrases of val (to be generated with the same 7B prompt: abbreviations, shorthand — the
wording that produced the human-set losses). P5 is kept only if it improves the terse set without
hurting plain or natural, on the relational path AND on the full pipeline (reranker refit on train
with the confirmed anchors, out-of-fold as before). No read of test / human under P5 until the
protocol passes; if it passes, one read, the fifth, disclosed.

### P5 RESULT (2026-09-07 20:16-20:35, jinx GTX 1080 Ti, scripts/p5_jinx.sh + scripts/p5_jinx2.sh): fail-fast NEGATIVE
Reduced fail-fast version of the pre-registered grid (the user: "use the 1080ti and fail fast"): 1,000
train questions instead of 2,000, plain wording and the terse paraphrases (data/para_train_terse.json),
relational path only (latent parser lp_p3 + bge anchor fallback, --beta 30), confirm model = lp_p3.
Reads: train split only. Hit@1 / Hit@5 / R@20 / MRR, and string-matched candidates kept / dropped.

Grid 1 (the pre-registered thresholds; ABS is a multiple of the parser's trained anchor floor):
| wording | setting            | Hit@1 | Hit@5 | R@20 | MRR  | kept / dropped |
| plain   | no confirm         | 26.2 | 43.1 | 51.2 | 34.2 | 2043 / 0    |
| plain   | rel 0.8 abs 0.75   | 19.9 | 36.5 | 45.6 | 27.8 | 1058 / 985  |
| plain   | rel 0.9 abs 0.75   | 19.9 | 36.5 | 45.6 | 27.8 | 1056 / 987  |
| plain   | rel 0.8 abs 1.0    | 19.6 | 36.0 | 45.2 | 27.2 |  707 / 1336 |
| plain   | rel 0.9 abs 1.0    | 19.6 | 36.0 | 45.2 | 27.2 |  706 / 1337 |
| terse   | no confirm         | 22.0 | 36.6 | 42.9 | 28.8 | 1586 / 0    |
| terse   | rel 0.8 abs 0.75   | 16.6 | 30.1 | 39.1 | 23.2 |  800 / 786  |
| terse   | rel 0.9 abs 0.75   | 16.6 | 30.1 | 39.1 | 23.2 |  796 / 790  |
| terse   | rel 0.8 abs 1.0    | 15.4 | 29.2 | 37.7 | 22.1 |  528 / 1058 |
| terse   | rel 0.9 abs 1.0    | 15.4 | 29.2 | 37.7 | 22.1 |  527 / 1059 |
The absolute floor binds everywhere (REL 0.8 vs 0.9 changes nothing): half or more of all string-matched
candidates score below 0.75x the parser's own floor, correct ones included. Loss of 5-7 Hit@1 on both wordings.

Grid 2 (added after grid 1, still train only; the collision-only reading: no absolute floor, so every
matched name keeps at least its best candidate and only same-name losers can be removed; retrieve.py
gained a guard so a name whose best score is <= 0 keeps all candidates):
| wording | setting            | Hit@1 | Hit@5 | R@20 | MRR  | kept / dropped |
| plain   | rel 0.5  no floor  | 25.1 | 43.0 | 50.5 | 33.4 | 1789 / 254  |
| plain   | rel 0.8  no floor  | 25.1 | 43.2 | 51.0 | 33.5 | 1762 / 281  |
| plain   | rel 0.95 no floor  | 25.0 | 43.2 | 51.0 | 33.5 | 1755 / 288  |
| plain   | rel 0.8  abs 0.5   | 20.7 | 37.8 | 47.2 | 28.8 | 1371 / 672  |
| terse   | rel 0.5  no floor  | 21.5 | 36.9 | 42.9 | 28.6 | 1419 / 167  |
| terse   | rel 0.8  no floor  | 21.5 | 37.0 | 43.1 | 28.6 | 1397 / 189  |
| terse   | rel 0.95 no floor  | 21.5 | 36.9 | 43.1 | 28.6 | 1387 / 199  |
| terse   | rel 0.8  abs 0.5   | 17.7 | 31.8 | 39.9 | 24.5 | 1072 / 514  |
No setting improves terse; the gentlest one (drop only same-name candidates under half the best score)
costs 1.1 plain / 0.5 terse Hit@1. The three relative thresholds are flat: nearly every collision loser
scores far below the winner, i.e. the parser's question vector takes a decisive side, and it sides
against the node the answer path needs often enough to lose. The existing behaviour (keep every
candidate, let exact-support and the z-scored readout sort them) is better than the parser's pick.

DECISION: P5 fails its own pre-registered gate (help terse without hurting plain) at step 1, on train.
Not carried to the val protocol; no terse val paraphrases generated; no read. Recorded as the third
pre-registered negative of this line (after the query-time AND and the trained AND). The name-collision
losses on the human set (MS, CAD, CP, HR) remain open; a rule that fixes them will have to come from
the readout over the candidate's neighbourhood (which candidate has the relation the question asks
for), not from the question vector alone. SUBMISSION stays P3.

### P6 — reverse-operator candidate feature (2026-09-08, written before any run)
Idea, from the wikikg2 line (`/mnt/geocore/resonate/wikikg2/REVERSE_MEMBER.md`, RV1, 2026-09-08):
the table has SEPARATE forward and reverse operators per relation (H is (36, 36, 4, 4) here: 18
relations x 2 directions), so the opposite operator is a second, independent scorer of the same
link. For a candidate c that the walk reached from anchor a through hop op, let r_rev = op ^ n_rel
(the reverse of the LAST hop of the winning chain; a two-hop chain uses its second hop):
  rev_raw(c) = Re< out(hop(embed(c), r_rev), r_rev), row(a) > * tau
  rev_nov(c) = rev_raw(c) - logsumexp over the OTHER candidates of the same shortlist as
               alternative targets (how much c prefers a over other links)
Over several anchors: max and mean, so four reranker columns (rev_raw_max, rev_raw_mean,
rev_nov_max, rev_nov_mean). On wikikg2 the pair added +0.017 MRR to an allowed blend, all of it on
"which entity of this type" questions — the shape of the open STaRK losses (MS / CAD / CP / HR
name collisions).

Implementation (no retraining, no LLM, nothing else changes): `retrieve.py --rev` records, per
candidate and per anchor, which hop won that candidate (argmax over the chains already scored) and
computes the two numbers from the SAME P3 table the pipeline uses (models/p_k12b4_50k.pt); they are
written into the existing `--dump-feats` block. `rev_check.py` is the self-check: for every one-hop
question whose shortlist contains the true answer, rev_raw at that answer is recomputed the long
way — the model's own readout of the reverse-operator query over the whole table, read at the
anchor row — and must agree to ~1e-5 (max absolute difference); for the candidates the walk reached
through the reverse operator the opposite operator is the FORWARD one, so the recomputed number is
literally the table's forward score of the triple (candidate, r, anchor), as in
`wikikg2/reverse_wiki.py`. The check is reported whatever it says; a failure stops the step.

Fail-fast protocol (the P5 protocol, fixed here before running):
* Questions: the first 1,000 of `train`, plain wording and the terse paraphrases
  (`data/para_train_terse.json`). Relational path only: `--anchor bge --beta 30 --lparse`, with the
  lp_p3 parses already produced for P5 (`data/lparse_train_full.json`, `data/lparse_train_terse.json`).
* Reranker: `rev_oof.py`. Groups are the relational top-100 of each question. Features, arm "base":
  z, exact, exact/nm, rel_rrf, logdeg (the relational subset of the P3 reranker's set); arm "rev":
  base + the four columns above. Model: the P3 reranker's listwise logistic regression
  (`rerank.fit`, steps 400, lr 0.05, L2 1e-3, Adam, features standardised on the fitting folds),
  ONE global weight vector (1,000 questions is too few for per-type fits). Out of fold: fold =
  position in the slice % 5; each fold is scored by the model fit on the other four, fit on the
  plain AND terse groups together (the P3 reranker is likewise fit on plain + paraphrased train).
  No question is ever scored by a model fit on it. Both arms use the same folds, the same
  shortlists and the same optimiser settings; only the feature columns differ.
* Metrics: `predict.py --score` on the same 1,000 train questions (`--limit 1000`, added for this),
  i.e. the official scoring path; questions with no relational output count as misses.
  Hit@1 / Hit@5 / Recall@20 / MRR, plus the no-reranker rows for reference.
BAR, fixed before running: plain train Hit@1 with the features at least +1.0 over without, AND
terse train Hit@1 not below the without-rev number. If met: the same comparison on `val`, under a
SEPARATE registration written before that run. If not met: record the negative and stop.
READS: `train` only (question text, and answers inside `predict.py --score`). No read of `test`,
`test-0.1` or `human_generated_eval` under any outcome of this step.
Deviation disclosed: run on a rented RTX 5090 (vast.ai instance 50261550) with the box's torch
2.11.0+cu128, because sm_120 has no torch 2.6.0 kernels; jinx stays pinned at 2.6.0 and the
val step, if it happens, can be reproduced there. Everything else — checkpoint, parses, splits,
scoring path — is the P3 pipeline's.

### P6 RESULT (2026-09-08 12:18-12:24 UTC, vast.ai 50261550, RTX 5090, torch 2.11.0+cu128; scripts/p6_box.sh): fail-fast NEGATIVE
Reads: `train` only, the first 1,000 questions, plain wording and `data/para_train_terse.json`.
Retrieval with `--rev` reproduces the P5 grid's no-confirm rows exactly on a different GPU and a
different torch (plain 26.2 / 43.1 / 51.2 / 34.2, terse 22.0 / 36.6 / 42.9 / 28.8), so the feature
computation changes nothing about the walk.

Self-check (`rev_check.py`, 400 one-hop train questions, at the true answer): max |rev_raw − the
model's own readout of the reverse-operator query over the whole table, at the anchor row| =
**1.14e-05**, mean 2.07e-06, on scores of magnitude ~18 — max RELATIVE difference **3.79e-07**. On
the 205 of those where the opposite operator is the FORWARD one (so the recomputed number is the
table's own forward score of the triple, the wikikg2 check) the max is the same 1.14e-05. A float64
reference of the same number sits 1.30e-05 from the batched value and 7.67e-06 from the full-table
readout: both fp32 paths round, neither is wrong. PASS at the pre-registered ~1e-5.
For information: the forward score of the same link and rev_raw are different numbers (mean 17.52
vs 18.05, Pearson r 0.731) — the two directions are separate learned operators, which is the point.

Out-of-fold reranking of the relational shortlist (5 folds by position, fit on the plain AND terse
groups of the other four folds; all rows scored by `predict.py --score --limit 1000`):

| wording | arm                          | Hit@1 | Hit@5 | R@20 | MRR  |
| plain   | no reranker                  | 26.2 | 43.1 | 51.21 | 34.15 |
| plain   | out-of-fold reranker, base   | 26.3 | 43.4 | 51.68 | 34.37 |
| plain   | out-of-fold reranker, + rev  | 27.0 | 43.5 | 51.98 | 34.73 |
| terse   | no reranker                  | 22.0 | 36.6 | 42.90 | 28.82 |
| terse   | out-of-fold reranker, base   | 22.2 | 36.8 | 42.78 | 28.96 |
| terse   | out-of-fold reranker, + rev  | 22.0 | 36.8 | 43.45 | 29.01 |

rev vs base: plain **+0.7** Hit@1, +0.1 Hit@5, +0.30 R@20, +0.36 MRR; terse **−0.2** Hit@1, +0.0
Hit@5, +0.67 R@20, +0.05 MRR.

Learned weights (mean over the 5 out-of-fold fits, sd in brackets):
  base   z +0.684 (0.038)  exact +0.904 (0.061)  exact/nm +1.000 (0.093)  rel_rrf +0.465 (0.021)  logdeg −0.267 (0.024)
  + rev  z +0.509 (0.045)  exact +0.897 (0.061)  exact/nm +0.778 (0.095)  rel_rrf +0.438 (0.021)  logdeg −0.159 (0.033)
         **rev_raw_max +0.475 (0.046)  rev_raw_mean +0.681 (0.107)**  rev_nov_max −0.389 (0.064)  rev_nov_mean −0.042 (0.112)

DECISION: P6 fails its own pre-registered bar (+1.0 Hit@1 on plain train AND no loss on terse) on
both halves: +0.7 plain, −0.2 terse. Not carried to val; no val paraphrases generated; no read of
test / test-0.1 / human under any part of this step. Recorded as the fourth pre-registered negative
of this line (after the query-time AND, the trained AND, and P5's latent-confirmed anchors).
SUBMISSION stays P3.
What the numbers do say, for whoever picks this up: the reverse operator is NOT noise. The reranker
gives rev_raw_mean a weight of the same size as the walk's own z-score and takes weight away from z
(0.684 → 0.509) and exact/nm — it is a real second opinion on the same link, and it moves Recall@20
and MRR up on BOTH wordings, which is the shape of a useful feature that is simply too small here
to clear a Hit@1 bar. rev_nov is the weak half: it saturates at 0 whenever the anchor dominates the
shortlist's alternative targets (it did so for most candidates of most questions), its weight is
negative and its sd across folds is as large as its mean. On wikikg2 the pair was worth +0.017 MRR
against 500 OGB-supplied decoys per question; here the "alternative targets" are the walk's own
top-100, which are already the anchor's neighbours, so the normalisation has much less to bite on.
A next attempt would score the candidate's reverse link against a set of alternative ANCHORS
(other entities of the anchor's type) rather than other candidates, and would be tested on the
name-collision questions (MS / CAD / CP / HR) directly rather than on the whole slice.

## P7 pre-registration (2026-09-08, before any run) — one anchor encoder, one table (size, not score)
Measured footprint of the SCORED P3 pipeline (not the ask.py demo, which loads three of the four):
| component | on disk | params |
| lp_p3.pt — latent parser (encoder + 0.7M heads) | 420 MB | 110.2M fp32 |
| BAAI/bge-base-en-v1.5 — anchor fallback (retrieve.py --anchor bge) | 419 MB | 109.5M |
| data/doc_emb_bge.npy — its 129,375 x 768 doc matrix | 190 MB | fp16 |
| bge_ft2 — text ranker | 419 MB | 109.5M |
| data/doc_emb_bgeft2.npy — its doc matrix | 190 MB | fp16 |
| p_joint_enc + p_joint_head.pt — text-to-latent trunk (+0.22M head) | 420 MB | 109.7M |
| p_joint.pt — the table the text-to-latent readout scores against | 142 MB | 129,375 x 144 complex |
| p_k12b4_50k.pt — the table the walk scores against, + 36 operators | 142 MB | same |
| TOTAL | ~2.34 GB | four 110M encoders, two tables, two doc matrices |
Two things the record already settles, so they are not tested here: the text-to-latent branch
cannot be dropped (without it 35.6 Hit@1, with it 43.5 — the largest single lever in P2), and one
table can in principle serve both (lever 5: joint table on the walk 24.3 vs 24.9 Hit@1 with Hit@5 /
R@20 / MRR slightly up, and better held-out link MRR 0.5680 vs 0.5574).

The two levers, both configuration only — NO retraining, no new component, no LLM:
 A. The anchor fallback resolves names with an UN-fine-tuned bge-base against its own doc matrix,
    while bge_ft2 — a fine-tune of that same model — is already loaded for the text ranker.
    `--anchor bgeft2` (one line in retrieve.py's model map; data/doc_emb_bgeft2.npy already exists,
    built by embed_text2.py with the identical prefix, pooling and L2 normalisation) removes one
    encoder and one doc matrix: -609 MB.
 C. The walk runs on models/p_joint.pt, the table the text-to-latent readout already uses
    (same shape, same operator set, same loader): -142 MB, and it makes "one table answers graph
    queries and language questions" true of the shipped pipeline rather than of a side experiment.
Together: 2.34 GB -> 1.59 GB (-32%), three encoders instead of four, one table, one doc matrix.

Step 1 — fail-fast, TRAIN only (this step): the first 1,000 train questions, plain wording and the
terse paraphrases (data/para_train_terse.json), relational path only, `--beta 30 --lparse` with the
same lp_p3 parses used by P5 and P6 (data/lparse_train_full.json, data/lparse_train_terse.json),
anchor-top 2, anchor-weight 0.7, k 100. Four arms:
  0.  --model p_k12b4_50k --anchor bge      (the current P3 configuration, the reference)
  A.  --model p_k12b4_50k --anchor bgeft2
  C.  --model p_joint     --anchor bge
  AC. --model p_joint     --anchor bgeft2
Also reported: how many of the 1,000 questions reach the text-anchor fallback at all (the only
questions lever A can move) — retrieve.py now counts it.
BAR, fixed before running: arm AC's Hit@1 not more than 1.0 below arm 0 on plain AND not more than
1.0 below on terse. A size reduction is bought, not free, but more than a point of the relational
path is too much to pay for it.
Step 2 — if the bar is met, on `val`, under a SEPARATE registration written before that run: the
full P3 chain with the AC configuration (train retrievals plain + paraphrased with the out-of-fold
parses, reranker refit out of fold, val scored plain and paraphrased through predict.py --score),
decision bar there: not more than 0.5 Hit@1 below P3's val (42.3 plain / 39.7 paraphrased) on
either wording, for the 750 MB.
READS: `train` only in step 1. No read of test / test-0.1 / human_generated_eval under any outcome.
Same disclosed deviation as P6: the rented RTX 5090 (vast.ai 50261550) with torch 2.11.0+cu128.

### P7 STEP 1 RESULT (2026-09-08 12:42-12:46 UTC, vast.ai 50261550, RTX 5090; scripts/p7_box.sh): BAR MET, and one lever is a gain
1,000 train questions, plain wording and terse paraphrases, relational path only, lp_p3 parses.
Arm 0 reproduces P5's and P6's rows exactly, so the arms are comparable line for line.
The text-anchor fallback — the only thing lever A can touch — fires on 63 of the 1,000 plain
questions and 110 of the terse ones.

| arm | configuration                | plain Hit@1 / Hit@5 / R@20 / MRR | terse Hit@1 / Hit@5 / R@20 / MRR | query-time size |
| 0   | bge anchors + p_k12b4_50k (P3) | 26.2 / 43.1 / 51.21 / 34.15 | 22.0 / 36.6 / 42.90 / 28.82 | 2.34 GB |
| A   | bge_ft2 anchors + p_k12b4_50k  | 25.3 / 42.4 / 49.91 / 33.18 | 21.3 / 35.1 / 41.32 / 27.92 | 1.73 GB |
| C   | bge anchors + p_joint          | **29.2 / 45.9 / 52.73 / 37.29** | **24.1 / 38.5 / 44.90 / 31.10** | 2.20 GB |
| AC  | bge_ft2 anchors + p_joint      | 28.2 / 45.1 / 51.42 / 36.26 | 23.4 / 37.3 / 43.61 / 30.15 | 1.59 GB |

Against arm 0: A −0.9 plain / −0.7 terse; C **+3.0 / +2.1**; AC **+2.0 / +1.4**.
BAR (arm AC not more than 1.0 Hit@1 below arm 0 on either wording): MET — AC is 2.0 and 1.4 ABOVE
it. The 751 MB is not bought at a price; it comes with a gain on the relational path.

Two readings, both worth keeping:
* Lever C is not a compression, it is an improvement. Lever 5 had measured the joint table on the
  walk as −0.6 Hit@1 with the P2-era parser and the LLM override; with the lp_p3 parser it is +3.0
  plain / +2.1 terse. The table trained on edges AND questions is simply a better table for the
  walk once the parser reads the question into that same space — which is the architecture claim of
  this line, now true of the shipped pipeline rather than of a side experiment, and it removes the
  second entity table (−142 MB).
* Lever A does cost something: −0.9 / −0.7 Hit@1, spread over the 63 / 110 questions that reach the
  fallback, i.e. a large relative loss on those. Plausible reason: bge_ft2 was fine-tuned to point a
  QUESTION at its ANSWER documents, so its question vector is a worse resolver for the entity the
  question MENTIONS than the un-fine-tuned bge, which is a generic semantic matcher. The 609 MB is
  therefore a real trade, not a free one, and step 2 measures both C and AC so the trade can be made
  on full-pipeline numbers rather than on the relational path alone.

### P7 step 2 pre-registration (2026-09-08, written before the run) — the two candidates in the FULL pipeline, on val
Step 1's bar was met, so the two size-reducing configurations go through the whole P3 chain and are
decided on val. Configurations (nothing else changes — no retraining, no new component, no LLM):
  ref  bge anchors + p_k12b4_50k table   = P3 as it stands (2.34 GB), re-run here as a control:
       it must reproduce the recorded 42.26 / 68.32 / 75.48 / 53.93 on plain val with the EXISTING
       reranker (data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json); if it does not, the step stops.
  C    bge anchors + p_joint table       (2.20 GB, −142 MB: one table instead of two)
  AC   bge_ft2 anchors + p_joint table   (1.59 GB, −751 MB: one table, and the anchor fallback
                                          reuses the text ranker's encoder and doc matrix)
Chain per candidate, identical to scripts/lp_pipe_chain.sh: retrieve train plain and paraphrased
with the OUT-OF-FOLD lp parses (data/lparse_train_oof{,_para}.json) and --dump-feats; retrieve val
plain and paraphrased (data/lparse_val{,_para}.json); refit the reranker on those out-of-fold
features (rerank.py --text-tag bgeft2 --train-text-tag bgeft2oof --t2l-tag pjoint --train-t2l-tag
pjointoof, augmented with the paraphrased train groups, per-type weights, save-tag _oof); score val
plain and paraphrased through predict.py --score.
FIXED, not tuned per candidate: the RRF weight stays w = 0.45 (data/fusion_bgeft2.json, chosen on
train for P3). Re-picking it per configuration would both overwrite a P3 artefact and add a tuned
degree of freedom to a comparison that is about size; it is deliberately left alone and disclosed.
The text ranker, the text-to-latent rankings and all five out-of-fold folds are the P3 ones,
unchanged by either lever.
DECISION RULE, fixed before the run: a candidate is admissible if its val Hit@1 is not more than
0.5 below P3's on EITHER wording (P3: 42.3 plain, 39.7 paraphrased). Among the admissible ones the
SMALLEST pipeline wins (AC < C < P3) — this step is about size, so a candidate is not preferred for
scoring higher, only rejected for scoring materially lower. If both fail, P3 stands and the size
work moves to the levers that do not touch the anchors (fp16 storage, doc-matrix compression).
READS: `train` (retrievals and the reranker fit) and `val` (the decision). No read of test,
test-0.1 or human_generated_eval under any outcome. Same disclosed deviation: vast.ai 50261550,
RTX 5090, torch 2.11.0+cu128.

### P7 STEP 2 RESULT (2026-09-08 12:50-13:17 UTC, vast.ai 50261550, RTX 5090; scripts/p7_step2.sh): BOTH CANDIDATES INADMISSIBLE by the pre-registered rule
Control first: P3 as it stands, re-run from scratch on this box with the existing reranker,
reproduces the recorded numbers to four decimals — 42.26 / 68.32 / 75.48 / 53.93 plain and
39.71 / 63.94 / 71.80 / 51.09 paraphrased. The step is valid.

Full pipeline on val (reranker refit out of fold per candidate, per-type weights, w 0.45, scored
through predict.py --score):
| candidate | query-time size | plain Hit@1 / Hit@5 / R@20 / MRR | paraphrased Hit@1 / Hit@5 / R@20 / MRR |
| P3 (control)              | 2.34 GB | 42.26 / 68.32 / 75.48 / 53.93 | 39.71 / 63.94 / 71.80 / 51.09 |
| C  (joint table)          | 2.20 GB | 42.21 / 68.50 / 74.92 / 53.73 | 39.13 / 63.86 / 71.10 / 50.48 |
| AC (joint table + shared anchor encoder) | 1.59 GB | 42.12 / 68.18 / 74.72 / 53.68 | 39.00 / 63.05 / 70.94 / 50.22 |
Against P3: C −0.05 plain / −0.58 paraphrased Hit@1; AC −0.14 plain / −0.71 paraphrased.
DECISION RULE (admissible = not more than 0.5 Hit@1 below P3 on EITHER wording): C fails on the
paraphrased set by 0.08, AC by 0.21. Both inadmissible. P3 STANDS; the 751 MB is not taken.
The margins are 13 and 16 questions of 2,241 and well inside the set's noise, but the bar was fixed
before the run and is applied as written. Taking AC anyway would be a choice made after seeing the
numbers; it is available (data/rerank_p7AC_lp_ancf_bgeft2_pjoint_aug_oof.json and the retrievals are
kept) and would have to be recorded as such, not as a bar that was met.

CORRECTION to P7 step 1, which this step exposes: arm C's +3.0 / +2.1 Hit@1 on 1,000 TRAIN questions
was measured IN SAMPLE. models/p_joint.pt is the table joint_train.py fit on PrimeKG's edges AND on
the train questions, so a train question is one the joint table was trained to answer, while the
edges-only table p_k12b4_50k never saw it. On val the same lever is −0.05 plain / −0.58 paraphrased.
The P5/P6-style fail-fast on train is therefore INVALID for any lever that changes the entity table
(and remains valid for levers that do not — P6 used the edges-only table, and arms 0 and A here are
unaffected). Any future table lever must fail-fast on a held-out slice of train that joint_train.py
did not see, or go straight to val. Recorded so the next session does not repeat it.

What the step does establish, for the size work that continues: neither lever is damaging. A 32%
smaller pipeline — one entity table instead of two, and the anchor fallback reusing the text
ranker's own encoder and doc matrix instead of a second, un-fine-tuned copy of the same model —
costs 0.14 Hit@1 on plain val and 0.71 on paraphrased. The remaining size levers (fp16 storage for
the encoders and tables, a compressed doc matrix, one shared trunk with three heads, a smaller
trunk) are untouched by this result, and the shared-trunk lever now carries a warning from step 1:
bge_ft2's question vector is a WORSE anchor resolver than un-fine-tuned bge (−0.9 / −0.7 on the
relational path), because it was fine-tuned to point a question at its answer rather than at the
entity the question mentions. A shared trunk would inherit that conflict and needs a head that
keeps generic mention-matching behaviour.

## P8 pre-registration (2026-09-08, before any run) — fewer PARAMETERS: parser-head anchors, rank-reduced document matrix
Measured parameter inventory of the scored P3 pipeline (not bytes — parameters):
| lp_p3 encoder + heads 110.2M | bge-base anchor fallback 109.5M | bge_ft2 text ranker 109.5M |
| p_joint_enc + head 109.7M | doc_emb_bge 99.4M | doc_emb_bgeft2 99.4M |
| p_k12b4_50k table + operators 37.3M | p_joint table + operators 37.3M | TOTAL 712.1M |
The four encoders are 62% and the two document matrices 28% — the matrices are nearly two encoders'
worth of parameters and were invisible in the "three 110M encoders" framing. fp16 / int8 are NOT in
this plan: they cut bytes, not parameters, and stack on top of whatever count this ends at.

Levers, neither of which retrains anything:
 A'. The anchor fallback (question-level text resolution, used only when neither string matching nor
     the latent parser's own anchors produced a mention: 63 / 1,000 plain and 110 / 1,000 terse
     train questions) is served by the latent parser's anchor head — ALREADY LOADED for --lparse —
     instead of a separate un-fine-tuned bge-base plus its own document matrix. retrieve.py gains
     `--anchor lp` (--lp-anchor models/lp_p3.pt): the question's K=3 anchor vectors score every
     entity through the table, the answer-type-connected types are kept, top-2 at anchor-weight 0.7,
     exactly the shape of the bge fallback it replaces, and with no floor (this IS the last resort).
     −109.5M encoder −99.4M document matrix = −208.9M. This is the version of P7's lever A that does
     not fight its own objective: P7 showed bge_ft2 is a WORSE anchor resolver than stock bge
     (−0.9 / −0.7) because it was fine-tuned to point a question at its answer rather than at the
     entity the question mentions; the parser's anchor head is trained for precisely that task.
 B'. The remaining document matrix is replaced by its rank-d corpus subspace (proj_docs.py, a
     truncated SVD of the 129,375 x 768 document matrix itself — documents only, no query labels,
     no split read): a 129,375 x d matrix plus a 768 x d projection. Documents and queries are both
     projected and ordered by inner product (not renormalised). Fitted energy: d=128 0.9162
     (16.7M, −83%), d=256 0.9709 (33.3M, −66%), d=384 0.9889 (50.0M, −50%).
Step 1 — choose d on TRAIN, no val read: the text ranker alone (embed_text2.py --docs/--proj) on the
train split at d in {128, 256, 384} against the unprojected 768. RULE: the smallest d whose train
Hit@1 is within 0.3 and Recall@20 within 0.5 of 768. (bge_ft2 was fine-tuned on train, so these
train numbers are in-sample in absolute terms; the comparison between d and 768 uses the same
encoder on the same questions, which is what the rule needs.)
Step 2 — ONE val read, two candidates, scored through predict.py --score:
   A'      : --anchor lp, unprojected documents, reranker REFIT out of fold on train retrievals
             (plain + paraphrased, out-of-fold parses) exactly as scripts/lp_pipe_chain.sh does.
   A' + B' : the same retrievals and the same refit reranker, with the val text ranking taken from
             the projected matrix at the chosen d.
   Reference: P3, whose val numbers were reproduced from scratch on this same box under P7 step 2
   (42.26 / 68.32 / 75.48 / 53.93 plain, 39.71 / 63.94 / 71.80 / 51.09 paraphrased) with these
   artefacts; not re-run a third time today.
   DISCLOSED APPROXIMATION: the reranker's train-side text features are precomputed out-of-fold
   rankings from five fold encoders that were not kept, so B' cannot be propagated into them; the
   projection is applied at val only. It is a query-time approximation of the same encoder, not a
   new model, and the bar below carries the risk.
BAR, fixed before running: a candidate is admissible if its val Hit@1 is not more than 0.5 below P3
on EITHER wording (the paraphrased set is the binding half — it is where both P7 candidates failed).
Among the admissible candidates the one with the FEWEST PARAMETERS is adopted. Targets: A' 503.2M,
A' + B' 437.1M at d=256 (from 712.1M).
READS: train (step 1, the retrievals and the reranker fit) and val (step 2, the decision). No read of
test / test-0.1 / human_generated_eval under any outcome. Runs on the rented RTX 5090 (vast.ai
50261550, torch 2.11.0+cu128) only; the 1080 Ti is not used.

### P8 RESULT (2026-09-08 13:40-13:55 UTC, vast.ai 50261550, RTX 5090; scripts/p8_box.sh + scripts/p8_step2.sh): BOTH CANDIDATES ADMISSIBLE — 712.1M -> 453.8M parameters
Step 1, choosing d on train (text ranker alone, same encoder, same questions):
| d   | train Hit@1 / Hit@5 / R@20 / MRR | vs 768        | document-matrix parameters |
| 768 | 24.52 / 48.47 / 55.91 / 35.65 | —             | 99.4M |
| 384 | 24.57 / 48.31 / 55.76 / 35.63 | +0.05 / −0.15 | 50.0M (49.7M + 0.3M projection) |
| 256 | 24.26 / 47.71 / 55.19 / 35.28 | −0.26 / −0.72 | 33.3M |
| 128 | 22.40 / 45.57 / 53.27 / 33.24 | −2.12 / −2.64 | 16.7M |
The rule (smallest d within 0.3 Hit@1 AND 0.5 Recall@20 of 768) selects d=384: d=256 passes on Hit@1
(−0.26) and misses on Recall@20 by 0.22 beyond the allowance. Applied as written. Noted for later: a
Recall@20 allowance of 0.75 would have taken d=256 and another 16.7M; the energy kept is 0.9889 at
384 and 0.9709 at 256.

Step 2, one val read (reranker refit out of fold for A'; P3's row is the control reproduced from
scratch on this box under P7 step 2):
| candidate | parameters | plain Hit@1 / Hit@5 / R@20 / MRR | paraphrased Hit@1 / Hit@5 / R@20 / MRR |
| P3        | 712.1M | 42.26 / 68.32 / 75.48 / 53.93 | 39.71 / 63.94 / 71.80 / 51.09 |
| A'        | 503.2M | 42.48 / 67.92 / 74.72 / 53.81 | 39.49 / 62.78 / 71.07 / 50.55 |
| A' + B'   | 453.8M | 42.44 / 67.69 / 74.57 / 53.74 | 39.49 / 63.05 / 70.87 / 50.51 |
Against P3 on the BAR's metric: A' +0.22 plain / −0.22 paraphrased; A'+B' +0.18 / −0.22. Both are
inside the 0.5 allowance on both wordings, so both are ADMISSIBLE, and the rule adopts the one with
the FEWEST PARAMETERS: **A' + B' — 453.8M, −36.3% of the pipeline, with plain Hit@1 slightly ABOVE
P3 and paraphrased 0.22 below.**
Stated plainly because the bar did not govern them: Hit@5 and Recall@20 do fall — plain −0.63 and
−0.91, paraphrased −0.89 and −0.93. A Recall@20 clause at the same 0.5 allowance would have failed
both candidates. The bar was pre-registered on Hit@1 alone and is applied as written; anyone reading
this should weigh the Recall@20 drop themselves.
On the relational path alone A' costs −0.4 plain / −0.6 paraphrased Hit@1 (26.37 / 22.13 against
P3's 26.8 / 22.7), with the fallback firing on 112 / 2,241 plain and 170 / 2,241 paraphrased
questions; the reranker absorbs it, and on plain wording more than absorbs it. This is the lever P7's
A got wrong: the parser's anchor head is trained to point at the entity a question MENTIONS, which is
what a fallback anchor resolver has to do, whereas bge_ft2 is trained to point at the ANSWER.
The rank-384 document matrix costs almost nothing on top: text ranker alone on val 21.55 / 43.06 /
50.36 / 31.67 plain against the full 768-d ranker's 21.9 / 43.0 / 50.6 / 31.9.

P8 PIPELINE (adopted): latent parser lp_p3 (also serving anchors, --anchor lp), bge_ft2 text ranker
over the rank-384 document matrix, p_joint text-to-latent, both entity tables, reranker
data/rerank_p8A_lp_ancf_bgeft2_pjoint_aug_oof.json. Three encoders instead of four, one document
matrix instead of two. No component was retrained. Nothing is submitted; P3 remains the filed
candidate until a decision says otherwise.
Remaining ladder from 453.8M: one trunk with three heads (−218.7M, needs training and a dedicated
anchor head so it does not re-create P7 lever A's conflict), distillation to a small trunk (−76M),
and the entity tables (74.6M, of which lever C's second table is −37.3M and missed P7's bar by 0.08).
fp16 / int8 remain available as a byte-level 2-4x on top of whatever the parameter count ends at.

### Pipeline speed (2026-09-08, engineering only — no experiment, both changes verified equivalent)
Observed while P8 ran: the retrieval process sits at 101% CPU on a 12-core box with the GPU at
10-15% and 1.2 GB used. The pipeline is CPU-bound and single-threaded, not GPU-bound; the rented
5090 was buying almost nothing for these runs (it will matter for P9, which trains).
Two hot spots, both fixed here, each verified to leave results unchanged:
 1. `Parser.mentions` tested all ~129,000 node names per question with `n in ql` — one Python
    operation each. It now uses an Aho-Corasick automaton over the same names (pyahocorasick,
    built once per Parser; falls back to the old scan if the module is missing). The automaton
    accepts exactly the names `n in ql` accepts, and they are walked in the SAME names_sorted order
    through the same boundary regex. VERIFIED: rerunning P7 arm 0's command over 1,000 train
    questions gives a byte-identical rel json — 0 of 1,000 questions differ. Full train split
    (6,162 questions, out-of-fold parses): 115 s -> 62 s.
 2. `rerank.fit` looped over ~12,000 groups in Python per optimiser step — 4.8M tiny kernel launches
    per fit, which is why an 11-feature logistic regression took ~7 minutes. It now pads the groups
    into one (G, Cmax, F) tensor with a mask and runs the identical loss as a single batched step.
    The old loop is kept behind `--slow-fit`. VERIFIED (fit_equiv.py, 1,500 groups, same data):
    max |weight difference| 7.15e-07 (relative 4.36e-07), 29.7 s -> 0.2 s (168x).
Also adopted for future scripts: independent retrievals run in parallel (12 cores, single-threaded
jobs) instead of serially. A full chain (2 train + 2 val retrievals, refit, 4 scored predicts)
should fall from ~25 minutes to ~5, which is what makes multi-arm val reads affordable.

## P9 pre-registration (2026-09-08, before any run) — one trunk, three heads (-218.7M parameters)
After P8 the pipeline is 453.8M parameters, of which THREE 110M encoders are 329.4M: the latent
parser (which since P8 also resolves anchors), the bge_ft2 text ranker, and the p_joint text-to-latent
trunk. All three are 12-layer/768 BERTs descended from the same BAAI/bge-base-en-v1.5, and their
task-specific parts are tiny (0.7M + 0.22M + 0). One shared trunk with all the heads on top is
109.5M + 1.2M = 110.7M: **-218.7M, taking the pipeline to ~235M**.
The known risk, from P7 lever A: the text and text-to-latent tasks train a question vector to point
at its ANSWER, while the anchor task trains it to point at the entity the question MENTIONS. Lever A
lost 0.9 Hit@1 by using the answer-pointing encoder as an anchor resolver, and P8 gained it back by
using the mention-pointing one. A shared trunk has to serve both, and this is the experiment that
says whether it can.

Model (shared_trunk.py, new): trunk = stock BAAI/bge-base-en-v1.5 (the neutral common ancestor, not
any of the three fine-tunes); CLS pooling. Heads on the CLS vector h:
  text        L2-normalised h, no parameters (the bge convention the doc matrix already uses)
  answer type Linear(768, 10)          ops Linear(768, 36)
  anchors     Linear(768, 3 x 2 x 144) -> 3 unit-norm complex vectors (lp's kanc=3)
  t2l         Linear(768, 2 x 144)     -> 1 unit-norm complex vector
Entity table: models/p_joint.pt, FROZEN — the same table lp_p3's anchor head and the p_joint t2l head
score against today, so one table serves both heads.
Losses, all on the same batch of questions, equal weight 1:1:1:1:1 (fixed, not tuned): answer-type
cross-entropy; operator BCE; anchor listwise over all entities (the latent parser's loss); t2l
listwise over all entities with the ANSWERS as positives; text InfoNCE between the question and its
answer document with in-batch negatives (the MultipleNegativesRanking loss finetune_embed.py uses).
Data: data/lp_labels.json — the same 12,324 weak-labelled rows (plain train questions plus their
para_train paraphrases) the latent parser was trained on; answers and answer documents come from the
train split and data/docs.jsonl. Optimiser AdamW, trunk 2e-5, heads 1e-3, weight decay 0.01,
OneCycle pct_start 0.1, 3 epochs, batch 24, seed 0. Anchor similarity floor tuned exactly as
latent_parser.py tunes it. Nothing here is searched or swept.

Step 1 — SCREEN, one training, NO val read. Train the trunk on train minus fold 0 (5:0, the same
fold convention as every out-of-fold artefact in this tree), then compare each head on fold 0's
questions against the DEDICATED model for the same fold, which already exists and excluded the same
questions: text vs data/oof/bgeft2_f0.json, text-to-latent vs data/oof/p_joint_f0.json, parser vs
data/oof/lparse_train_f0.json (both parses run through retrieve.py --fold 5:0 with the P8
configuration, --anchor lp --beta 30, so the comparison is the relational path each parse produces).
The comparison is exactly matched: same questions, same fold exclusion, dedicated versus shared.
BAR, fixed before running: no head more than 2.0 Hit@1 below its dedicated counterpart on fold 0.
A head that collapses stops P9 here and is recorded, with which head and by how much — that is the
informative outcome either way, because it localises the objective conflict.
Step 2 — only if the screen passes, under a SEPARATE registration: train the four remaining folds
plus a full-train trunk (six trainings in all), re-embed the corpus through the shared trunk, refit
the rank-384 projection on the new document matrix, regenerate every ranking, refit the reranker out
of fold, and take ONE val read against P8's 42.44 / 39.49 under the same 0.5 Hit@1 allowance.
READS: `train` only in step 1 (fold 0 is held out from the model that is screened on it). No read of
test / test-0.1 / human_generated_eval. Runs on the rented RTX 5090 (vast.ai 50261550) only — this
is the first step in this line that is actually GPU-bound.

### P9 STEP 1 RESULT (2026-09-08 14:06-14:12 UTC, vast.ai 50261550, RTX 5090; scripts/p9_screen.sh): SCREEN FAILS on the text head; the text-to-latent comparison is VOID
The shared trunk trained in 2m27s on the 9,858 rows outside fold 0 (3 epochs, all five losses
falling together), anchor floor 15.612 at F1 0.834 against the weak labels, 110.4M parameters saved
as models/st_f0.pt — against the 329.4M of the three encoders it would replace.
Fold 0 of train (n=1,233 questions), shared head vs the DEDICATED model that excluded the same fold:
| head            | shared                        | dedicated (fold 0 excluded)   | Hit@1 |
| text ranker     | 20.19 / 41.52 / 44.58 / 29.68 | 22.30 / 41.77 / 46.61 / 31.42 | **−2.11** |
| text-to-latent  | 44.93 / 70.80 / 69.37 / 56.74 | 26.52 / 39.25 / 36.42 / 32.38 | +18.41 — VOID, see below |
| parser (relational path, --anchor bge, both parses) | 26.85 / 42.17 / 51.63 / 34.39 | 26.36 / 42.17 / 51.03 / 34.05 | +0.49 |
BAR (no head more than 2.0 Hit@1 below its dedicated counterpart): the TEXT head is 2.11 below.
FAILS, by 0.11. P9 stops here under its own pre-registration; step 2 is not run.

The text-to-latent row is not evidence and must not be quoted as a gain. The shared trunk scores its
t2l head against the FULL models/p_joint.pt table, which joint_train.py fit on every train question
INCLUDING fold 0, while data/oof/p_joint_f0.json came from a run that retrained the table without
fold 0 (joint_train.py --fold 5:0; those runs return before saving, so models/p_joint.pt itself is
the clean full-train model — the deployed pipeline is unaffected). The shared arm was therefore
scoring held-out questions with a table that had been trained on their answers. This is the same
in-sample trap recorded two hours earlier for P7 step 1, entered from a different direction: there
the table was the lever, here the table is the fixed scorer and the FOLD MATCH was missed. Written
down again, in the general form: in this pipeline every fold-held-out comparison must fold-match the
ENTITY TABLE as well as the encoder, because the table is trained on train questions too.
The parser row IS fair: both arms score against the same full table (latent_parser.py's fold runs
use models/p_joint.pt as well), so the contamination is identical on both sides and cancels; the
absolute numbers are optimistic but the +0.49 comparison stands.

What the screen does establish, which is the interesting half: the conflict predicted from P7 lever A
and P8 is real and it has a direction. Sharing one trunk HELPS the parser head (+0.49 — the anchor
task is the one that wants a mention-pointing vector, and three of the five losses pull that way) and
HURTS the text ranker (−2.11 — the task that wants an answer-pointing vector). The trunk is being
pulled toward the mention objective, and the text task pays for it. A corrected P9 would have to
(a) rebalance the loss weights toward the text task, or give it its own adapter over the shared
trunk, and (b) redo the t2l screen against a fold-0-excluded table (joint_train.py --fold 5:0, ~5
minutes, no checkpoint of it was kept). Both change a pre-registered design and need a new
registration; neither is run here. P8's 453.8M stands as the smallest measured pipeline.

## P10 pre-registration (2026-09-08, before any run) — the human proxy that was never built, and what the 7B is worth now
Motivation. The human set is 98 questions: one question moves Hit@1 by 1.02 points and P2's
bootstrap CI was [21.4, 39.8], so every difference observed there so far (P1 20.4, P2 30.61,
P3 28.57, P4 25.51) is 2-5 questions and inside noise. Tuning toward that number is chasing noise.
The only defensible route is a mechanism that fixes a NAMED failure class, validated on a proxy.
P5 planned terse val paraphrases — the clinician's search-box shorthand that is the closest stand-in
for human phrasing — and died at step 1 on train, so data/para_val_terse.json was never generated.
This registration builds it and uses it to answer two open questions, then tests one mechanism.
Nothing here changes the submission and nothing reads test or human.

Step 1 — the proxy. paraphrase.py --split val --style terse --seed 3 (Qwen2.5-7B-Instruct, bf16,
the same prompt and seed lever C used for train) -> data/para_val_terse.json. Generation is OFFLINE,
not at query time; the rule this line keeps is no generative model when a question is ANSWERED.
Then the inputs each arm needs for the terse wording: lp_p3 parses, bge_ft2 text ranking (full and
rank-384), p_joint text-to-latent ranking, and llm_parse for the LLM arm.

Step 2 — three candidates x three wordings (plain / natural paraphrase / terse) on val, each scored
through predict.py --score with its OWN existing reranker; nothing is refit, nothing is tuned:
  P3   bge anchors, full document matrix, rerank_lp_ancf_bgeft2_pjoint_aug_oof.json      (712.1M)
  P8   parser-head anchors, rank-384 documents, rerank_p8A_lp_ancf_bgeft2_pjoint_aug_oof (453.8M)
  LLM  the P2-era max-score pipeline: regex parse + Qwen2.5-7B override, bge anchors,
       rerank_llm_ancf_bgeft2_pjoint_aug_oof.json
Why the LLM arm is here: the recorded cost of dropping the 7B is -2.2 Hit@1 plain / -1.6
paraphrased, but the lever-2c(b) block also recorded WHY — "the LLM parser adds COVERAGE rather than
phrasing robustness", taking coverage 96.2% -> 100.0%. Since P8 every val run reports coverage
100.0% WITHOUT the 7B, so that -2.2 is stale and has never been re-measured against the current
pipeline. This measures it, on the wording that matters.
DECISION RULES, fixed before running:
 * The 7B is worth re-opening for the paper's max-score row only if it gains >= 1.0 Hit@1 over P8 on
   TERSE val while losing no more than 0.5 on plain. Otherwise the no-LLM pipeline is confirmed on
   human-like wording as well as on synthesized, and the stale -2.2 is corrected in the record.
 * Between P3 and P8, the higher terse Hit@1 is the more human-robust; within 0.3 they are called
   equal and P8 wins on parameters (453.8M vs 712.1M).
Step 3 — the mechanism P5 named. P5's closing line: the MS / CAD / CP / HR collisions "will have to
come from the readout over the candidate's neighbourhood (which candidate has the relation the
question asks for), not from the question vector alone". That is exactly what P6's rev_raw computes,
and P6 only ever tested it AVERAGED OVER THE WHOLE SLICE, where a fix to a few percent of questions
cannot clear a +1.0 bar. Here it is tested where it was aimed: a val question is a COLLISION question
if some matched name string resolves to two or more distinct entity ids in its parse (computed from
the rel json, no new definition needed). retrieve.py --rev on train (plain + paraphrased) and val
(plain + terse); the relational-shortlist reranker of rev_oof.py fit on train, base vs + the four rev
columns; val scored on the collision subset and on the remainder separately.
BAR, fixed before running: the rev features are carried forward only if they gain >= 2.0 Hit@1 on the
collision subset while losing no more than 0.3 on the remainder.
READS: train (fitting) and val (all decisions). NO read of test, test-0.1 or human_generated_eval
under any outcome. If P10 concludes that a fifth read is worth proposing, that read is a SEPARATE
registration and needs explicit authorisation. Box: vast.ai 50261550 (RTX 5090) only.
FOLLOW-ON, not run here (P11, to be registered separately): the latent parser has never been trained
on terse phrasing — lever C changed only the embedder and said so explicitly — and since P8 the
parser's anchor head is what resolves anchors when no name matches, which is the dominant human
failure. data/lp_labels.json holds 12,324 rows (plain + natural paraphrase) whose labels are per
QUESTION, so terse rows reuse the same answer type, anchors and operators: 18,486 rows, one
retraining. P10's proxy is what makes P11 measurable.

## P11 pre-registration (2026-09-08, before any run) — the parser trained on terse phrasing
The gap lever C left open, stated in its own text: "the augmentation set for the parser and the
reranker's paraphrased groups stays para_train.json (unchanged) so ONLY THE EMBEDDER changes."
Lever C then failed and was dropped, and the terse train paraphrases it generated
(data/para_train_terse.json, all 6,162 questions) have been used by nothing since.
Why it matters more now than it did then: since P8 the latent parser's anchor head is what resolves
the anchor when no name matches — it replaced the fourth encoder — and the proxy error analysis puts
64% of rewording losses on missed anchors, 74% of those real aliases and descriptions. The component
that must survive human phrasing has never been shown any. Terse is that phrasing
("filensin lens support ocular" for a full sentence about filensin).
Design (nothing else changes; ONE thing moves):
 * Labels: data/lp_labels_terse.json = the existing 12,324 rows of data/lp_labels.json (plain +
   natural paraphrase) plus 6,162 terse rows. The weak labels are per QUESTION — answer type, anchor
   ids, operator vector — so a terse row reuses its question's labels verbatim and only the text
   differs; the row's train position is unchanged, so every fold filter keeps working. 18,486 rows.
 * Training: latent_parser.py --train --tag lp_p4t --labels data/lp_labels_terse.json with lp_p3's
   hyperparameters untouched (encoder models/p_joint_enc, table models/p_joint.pt, kanc 3, 3 epochs,
   batch 32, seed 0, the trained anchor floor). Then the five out-of-fold parsers (--fold 5:k, same
   labels) so the reranker's train features stay out of fold, exactly as lp_p3's chain built them.
 * Pipeline: the P8 configuration with the new parser in BOTH of its roles — --lparse from lp_p4t
   and --anchor lp --lp-anchor models/lp_p4t.pt. The reranker is refit on the new train retrievals
   with the SAME group structure as now (plain + natural paraphrase); terse is deliberately NOT added
   to the reranker's groups, so the parser is the only thing that changed.
 * Measured on val in three wordings: plain, natural paraphrase, terse (P10's proxy).
BAR, fixed before running: terse val Hit@1 at least 1.0 ABOVE P8's terse number, AND plain val Hit@1
no more than 0.3 below P8's plain. This is P5's shape — help the human-like wording without hurting
plain — applied to the component that actually reads the words.
READS: train (labels, training, retrievals, the reranker fit) and val (the decision). No read of
test, test-0.1 or human_generated_eval under any outcome. Depends on P10 step 1 for the terse val
proxy and on P10 step 2 for P8's terse number, so it runs after P10. Box: vast.ai 50261550 only.
Cost: six parser trainings (~4-5 minutes each at 18,486 rows), five fold predictions, five
retrievals, one reranker refit, three scored val runs.

### P10 step 1 (2026-09-08 14:5x UTC, vast.ai 50261550): the terse val proxy exists — data/para_val_terse.json
2,241 questions rewritten in 77 s (Qwen2.5-7B-Instruct, bf16, --style terse --seed 3, the prompt and
seed lever C used for train). Median length falls from 21 words to 7:
  "Which drugs specifically target the glucokinase (GCK) gene?" -> "drugs targeting GCK"
QUALITY CHECK, run before any arm was scored, because one sampled rewrite was the model ANSWERING
rather than rewriting ("Which renal condition would preclude ..." -> "chronic kidney disease"), which
would leak answers into the proxy and inflate every terse number:
  wording   median words   questions whose text contains the name of one of their true answers
  plain          21        104 (4.6%)
  natural        17        108 (4.8%)
  terse           7        126 (5.6%)
The baseline is 4.6% — STaRK questions legitimately name an entity that is also among their answers
(multi-answer questions naming one as a constraint). Terse adds ~1.0 percentage point, about 22
questions of 2,241. Recorded rather than filtered: filtering would change the question set for every
arm to remove an effect of ~22 questions, and all three arms share the same text ranker so the
differential between them is smaller still. Terse numbers carry a ~1pp optimistic bias from this
cause, disclosed here.

### P10 RESULT (2026-09-08 14:58-15:15 UTC, vast.ai 50261550; scripts/p10_step23.sh): the 7B COLLAPSES on human-like wording; the reverse features localise to collisions
Step 2 — three candidates x three wordings, val Hit@1 / Hit@5 / R@20 / MRR, each with its OWN
existing reranker, nothing refit:
| arm | parameters | plain | natural paraphrase | TERSE (the human proxy) |
| P3  | 712.1M | 42.26 / 68.32 / 75.48 / 53.93 | 39.71 / 63.94 / 71.80 / 51.09 | 34.09 / 54.89 / 62.47 / 43.75 |
| P8  | 453.8M | 42.44 / 67.69 / 74.57 / 53.74 | 39.49 / 63.05 / 70.87 / 50.51 | 33.96 / 54.35 / 61.99 / 43.35 |
| LLM (P2-era + Qwen2.5-7B) | 712.1M + 7B | **43.78** / 68.01 / 75.80 / 55.07 | **41.05** / 63.99 / 72.26 / 51.83 | **31.50** / 51.14 / 60.26 / 40.91 |
The 7B is the best arm on plain (+1.34 over P8) and on natural paraphrases (+1.56) — and the WORST
on terse, 2.46 BELOW P8 and 2.59 below P3. Its decision rule (>= 1.0 over P8 on terse, losing no
more than 0.5 on plain) fails by a wide margin and in the opposite direction. The no-LLM pipeline is
now confirmed on human-like wording, not just accepted as a -2.2 concession.
This is exactly what lever 2c(b) recorded and nobody followed up: "the LLM parser adds COVERAGE
rather than phrasing robustness", 96.2% -> 100.0%. Since P8 the pipeline reaches 100% coverage
without it, so the -2.2 was stale; and where the 7B has to READ shorthand rather than fill coverage
gaps it degrades faster than the parser it replaced (12 of 2,241 terse questions unparsable).
Recorded so the paper's "two honest pipelines" line can be corrected: the max-score pipeline is max
score ON SYNTHESIZED WORDING ONLY.
P3 vs P8 on terse: 34.09 vs 33.96, a gap of 0.13, inside the pre-registered 0.3 band, so they are
called equal and P8 wins on parameters (453.8M vs 712.1M). Context for the size: the text ranker
alone falls from 21.9 Hit@1 on plain to 15.13 on terse, so every arm is working much harder here.

Step 3 — the reverse-operator features where P5 said the collision fix must come from. A val
question is a COLLISION question if its parse resolves one name string to two or more entity ids:
333 of 2,241 (14.9%). Relational-shortlist reranker fit on train (8,033 groups, plain + paraphrased),
base vs + the four rev columns, scored per subset through predict.py --qids:
| wording | subset      | base  | + rev | delta |
| plain   | collision   | 22.82 | 24.92 | **+2.10** |
| plain   | remainder   | 27.67 | 28.14 | +0.47 |
| terse   | collision   | 18.02 | 18.62 | +0.60 |
| terse   | remainder   | 21.75 | 21.80 | +0.05 |
Learned weights confirm the features are used: rev_raw_max +0.636, rev_raw_mean +0.609 (rev_nov
again near zero: -0.278 / +0.018, as in P6).
BAR AMBIGUITY, disclosed: the pre-registration said "gain >= 2.0 Hit@1 on the collision subset while
losing no more than 0.3 on the remainder" WITHOUT naming the wording. On plain it passes (+2.10,
remainder +0.47). On terse it does not (+0.60). Under the conservative reading — both wordings, terse
binding, as every other bar in this line has been — P10 step 3 FAILS and the rev features are not
carried forward. That is the reading taken. The imprecision was mine and is recorded rather than
resolved in the favourable direction.
What is nonetheless established: P5's prediction was right in mechanism. rev_raw is "the readout over
the candidate's neighbourhood — which candidate has the relation the question asks for", and its
benefit is 4.5x more concentrated on collision questions than on the rest (+2.10 vs +0.47 on plain).
P6 failed because it averaged a fix for 15% of questions over 100% of them. A candidate that acted
ONLY on collisions, or a reranker with a collision indicator interacted with rev_raw, is the shape
this evidence points to — untried, and it would need its own registration.

### P11 RESULT (2026-09-08 15:20-15:33 UTC, vast.ai 50261550; scripts/p11.sh): NEGATIVE by the bar — the best terse score so far, bought with plain
Labels 18,486 rows (+6,162 terse over lp_p3's 12,324), six parser trainings (~1.5 min each), five
out-of-fold predictions, five retrievals, reranker refit, three scored val wordings.
First attempt failed cleanly and was rerun: latent_parser.py --train loads data/llmparse_train.json
unconditionally (before the label-cache check, although the cache makes it unused) and that file had
not been shipped to the box, so all six trainings raised FileNotFoundError, every later step failed
on its missing input, and the script — which had no error checking — still printed P11_DONE. No
partial artefact was written and nothing was contaminated. The script now aborts on a missing
checkpoint after each training; recorded because a silent chain of no-ops that ends in DONE is the
most dangerous shape a run can have.

Val Hit@1 / Hit@5 / R@20 / MRR, against the P8 pipeline the parser was dropped into:
| arm | plain | natural paraphrase | terse |
| P8  | 42.44 / 67.69 / 74.57 / 53.74 | 39.49 / 63.05 / 70.87 / 50.51 | 33.96 / 54.35 / 61.99 / 43.35 |
| P11 | 41.95 / 67.96 / 74.70 / 53.69 | 39.45 / 63.41 / 70.92 / 50.58 | **34.58 / 55.78 / 63.81 / 44.57** |
Against P8: plain −0.49, natural −0.04, terse **+0.62** (Hit@5 +1.43, Recall@20 +1.82, MRR +1.22).
BAR (terse at least +1.0 over P8 AND plain no more than 0.3 below): FAILS on both halves — terse
+0.62 falls 0.38 short, plain −0.49 is 0.19 past the allowance. Not adopted.
On the relational path alone the parser clearly did learn the wording: val terse 22.27 Hit@1, ABOVE
its own natural-paraphrase 22.13, an ordering no previous parser has produced (terse has been the
harder wording for every arm all day). The gain is real at the parse stage and only partly survives
the full pipeline, which is lever C's pattern one component to the left: there the embedder's terse
gain died in the reranker, here the parser's mostly does.
34.58 is nevertheless the highest terse number measured (P3 34.09, P8 33.96), and the trade is
legible: about half a point of plain for about six tenths of terse. That is a robustness/accuracy
exchange, not a win, and on a 98-question human set where one question is 1.02 points it is not the
kind of change that would move the read reliably.
Follow-ons this points at, neither run nor registered: weight the terse rows below the plain ones
instead of 1:1, or add terse only to the ANCHOR loss (the head P8 made load-bearing) rather than to
the answer-type and operator heads as well, so the trade is taken only where the failure is.

### METHODOLOGICAL CAVEAT (2026-09-08, raised by the user after P11): the proxy trains and tests the SAME generated style
P11 trains the parser on data/para_train_terse.json and is judged on data/para_val_terse.json. Split
hygiene is clean — the training text comes from TRAIN questions, the evaluation text from VAL
questions, and neither test nor human is involved — but both files come from the SAME generator, the
SAME prompt and the same style setting (paraphrase.py --style terse, Qwen2.5-7B-Instruct). So part of
P11's +0.62 on terse is not robustness to human shorthand; it is having learned one model's dialect
of shorthand — its particular abbreviations, its particular dropping patterns — and then being tested
on that dialect. How much of the gain transfers to real human phrasing cannot be read off this
measurement.
This is not specific to P11. bge_ft2, the text ranker in every pipeline since P2, was fine-tuned on
data/para_train.json and is evaluated on data/para_val.json — same script, same prompt, same model.
The whole "paraphrased val" column from P2 through P11 shares the property. P11 only makes it
explicit by training on the style deliberately.
The precedent that shows the risk is real and already cost this project a read: P4's selector gained
+2.1 plain / +1.0 paraphrased on val and LOST 5.1 Hit@1 on the human set, and the recorded conclusion
was that it "fits the synthesized distribution and does not transfer to human questions".
FIX for any future proxy work, cheap and not yet done: hold out the STYLE, not only the questions —
generate the evaluation proxy with a different prompt or a different model from the one whose output
trained the component, so that a gain means style-general robustness rather than dialect matching. A
terse number measured that way would be worth something as evidence about humans; the numbers in P10
and P11 are worth less than they look, and are recorded here as such.

## P12 pre-registration (2026-09-08, before any run) — hold out the STYLE, not only the questions
The caveat recorded above: every terse number so far trains on one generator's shorthand and is
graded on the same generator's shorthand (Qwen2.5-7B-Instruct, --style terse, one prompt), so part of
P11's +0.62 is dialect matching rather than robustness, and the same is true of the whole
paraphrased-val column since P2. P4 is the precedent for what that costs: +2.1 on val, −5.1 on human.
This builds an evaluation proxy whose style NO component has been trained on, and re-runs the two
claims that were made from terse numbers.
Held-out proxy (data/para_val_terse_b.json): the same compression intent under a DIFFERENT MODEL
FAMILY and a DIFFERENT PROMPT — microsoft/Phi-3.5-mini-instruct (3.8B, ungated, different
pretraining lineage from Qwen) with paraphrase.py --style terse2, a prompt rewritten from scratch
with different instructions and different examples. Both axes move at once on purpose: the goal is
maximum style independence, not attribution between model and prompt.
Quality check before any arm is scored, the same one P10 step 1 ran: median length, and the share of
questions whose text contains one of their own answer names (the plain-val baseline is 4.6%). If the
held-out proxy is degenerate — for instance if the model answers instead of compressing at a much
higher rate than Qwen did — that is reported and the proxy is not used.
Arms re-scored on the held-out style, each with its own reranker, nothing refit: P3, P8, P11, and
the P2-era Qwen-7B pipeline (which needs its own llm_parse pass over the new proxy).
DECISION RULE, fixed before running: an effect measured on the original terse proxy counts as
STYLE-GENERAL only if, on the held-out proxy, it keeps its SIGN and at least HALF its magnitude.
Applied to the two claims made today:
  * P11's terse advantage over P8 (+0.62) must remain at least +0.31 and positive.
  * The Qwen-7B pipeline's terse deficit against P8 (−2.46) must remain at most −1.23 and negative.
Whatever the outcome, the P10 and P11 result blocks are annotated with what survived. If the two
proxies disagree in sign on either claim, the honest conclusion is that neither proxy supports a
statement about human phrasing, and that is what will be written.
READS: val only (the proxy is generated from val question text, which development already reads).
No read of test, test-0.1 or human_generated_eval. Box: vast.ai 50261550.
