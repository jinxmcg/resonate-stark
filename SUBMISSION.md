# STaRK leaderboard submission — filled form (Prime, all three splits)

- **Method name:** ResonatE (no llm)
- **Dataset:** prime
- **Splits:** test (Synthesized full), test-0.1 (Synthesized 10%), human_generated_eval
- **Team name:** Cristian Malaia
- **Contact email:** cristian.malaia@gmail.com
- **Model type:** Others
- **Code repository:** https://github.com/jinxmcg/resonate-stark
- **Hardware:** 1x NVIDIA RTX 5090 (training and inference)
- **Paper:** ResonatE: Row-Sparse Knowledge-Graph Embeddings with Composable Relation Operators
  (https://github.com/jinxmcg/resonate, paper/resonate.pdf; STaRK section)
- **Prediction files:** results_p2/eval_results_test.csv, results_p2/eval_results_test-0.1.csv,
  results_p2/eval_results_human_generated_eval.csv (columns idx, query_id, pred_rank = top-100 node ids)

**Description (for the form):**
A knowledge-graph embedding (ResonatE: unit-norm complex entity table, composable relation
operators) trained on PrimeKG's edges and then jointly on the STaRK train questions, so that the
same table answers link-prediction queries and natural-language questions with one readout. At
query time no language model runs: a small learned parser head reads the question into the
model's query space (answer type, anchor entities by nearest neighbour in the table, relation
operators); an exact adjacency walk from the anchors supplies graph-supported candidates and the
model's composed operators order them; a fine-tuned bge-base-en-v1.5 ranks node descriptions; the
joint table ranks entities directly from the question text; the three lists are fused by
reciprocal rank and reordered by a per-answer-type logistic reranker fit on train with out-of-fold
features. A 7B instruction model (Qwen2.5-7B-Instruct) was used only offline, to paraphrase train
questions and to name entities for weak labels. Train split used for fitting, validation for all
decisions; the human-generated set was never used for development. This is our second read of
the test splits (first: 28.7 / 28.2 / 20.4 Hit@1, a hand-written-parser pipeline, also reported).
No external data beyond PrimeKG and the STaRK question files.

**Results (our committed read):** test 41.81 / 68.30 / 74.77 / 53.66; test-0.1 41.79 / 71.07 /
75.90 / 54.31; human 30.61 / 53.06 / 60.58 / 41.74 (Hit@1 / Hit@5 / Recall@20 / MRR).
