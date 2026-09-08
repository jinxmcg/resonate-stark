# STaRK leaderboard submission — Prime, all three splits (P3; prepared, not filed)

> **Submission status — 8 September 2026: nothing has been filed for BioKG,
> WikiKG2, or STaRK-Prime.** Entries are proposed only. **P3 is the selected
> candidate** (`PLAN_PRIME.md`, decision rule of 2026-09-07). This document was
> rewritten from the historical P2 draft; the P2 material it replaced is in git
> history.

- **Method name:** ResonatE (no llm)
- **Dataset:** prime
- **Splits:** test (Synthesized full), test-0.1 (Synthesized 10%), human_generated_eval
- **Team name:** Cristian Malaia
- **Contact email:** cristian.malaia@gmail.com
- **Model type:** Others
- **Code repository:** https://github.com/jinxmcg/resonate-stark
- **Hardware:** 1x NVIDIA RTX 5090 (training and inference)
- **Paper:** Beyond Link Prediction: Compact Knowledge Representations for Prediction, Retrieval, and Direct Access (ResonatE)
  (https://github.com/jinxmcg/resonate, paper/resonate.pdf; STaRK section)
- **Prediction files:** results_p3/eval_results_test.csv, results_p3/eval_results_test-0.1.csv,
  results_p3/eval_results_human_generated_eval.csv (columns idx, query_id, pred_rank = top-100 node ids)

**Description (for the form):**
A knowledge-graph embedding (ResonatE: unit-norm complex entity table, composable relation
operators) trained on PrimeKG's edges and then jointly on the STaRK train questions, so that the
same table answers link-prediction queries and natural-language questions with one readout. At
query time no generative language model is run (two 110M transformer encoders are: a fine-tuned bge-base text ranker and the parser / question encoder): a small learned parser head reads the question into the
model's query space (answer type, anchor entities by nearest neighbour in the table, relation
operators); an exact adjacency walk from the anchors supplies graph-supported candidates and the
model's composed operators order them; a fine-tuned bge-base-en-v1.5 ranks node descriptions; the
joint table ranks entities directly from the question text; the three lists are fused by
reciprocal rank and reordered by a per-answer-type logistic reranker fit on train with out-of-fold
features. A 7B instruction model (Qwen2.5-7B-Instruct) was used only offline, to paraphrase train
questions and to name entities for weak labels. Relative to the second read (P2), the submitted pipeline changes in exactly one respect: a
correctness fix. `Parser.by_name` dropped names shorter than four characters, so 2-3 character
gene/protein symbols (GCK, TTR, ...) were never anchored by exact match even though the
uppercase-symbol regex found them; a separate by_symbol dictionary now handles them, used only
through that rule. It affects 95 of 2,241 validation questions (4.2%). The old behaviour is kept
behind `--legacy-names` so the earlier reads stay reproducible. Nothing else differs.

Train split used for fitting, validation for all decisions; the human-generated set was never used
for development. **This is our third read of the test splits, and four reads have been taken over
the project; all four are reported below.** No external data beyond PrimeKG and the STaRK question
files.

**Results (the submitted read, P3):** test 43.09 / 68.80 / 75.50 / 54.73; test-0.1 41.79 / 72.50 /
77.83 / 54.34; human_generated_eval 28.57 / 53.06 / 61.85 / 40.67 (Hit@1 / Hit@5 / Recall@20 / MRR).

**Every test read taken on this project, in order, disclosed in full:**

| # | pipeline | test | test-0.1 | human | submitted |
|---|---|---|---|---|---|
| 1 | hand-written parser | 28.7 | 28.2 | 20.4 | no |
| 2 | P2 | 41.81 | 41.79 | 30.61 | no |
| 3 | **P3 — the short-symbol fix** | **43.09** | **41.79** | **28.57** | **yes** |
| 4 | P4 — P3 + a learned reading selector | 43.70 | 41.79 | 25.51 | no |

**P4 scored higher on test than the entry we are submitting and was dropped anyway.** The rule
fixed on 2026-09-07 was "publish the best model and architecture, not the best score": P4's selector
is a bolt that fits the synthesized distribution and does not transfer — it costs 3 questions on the
human-generated set (25.51 vs 28.57) while gaining 0.61 on synthesized test. P3 is a correctness fix
that belongs to the model. P4's prediction files are kept in `results_p4/` and are not submitted.

Parameter count: 712.1M. A shared-trunk variant at ~235M was built and measured (P15) and is NOT
submitted: it loses 6.69 Hit@1 on validation, because one encoder feeding all three rankers
correlates their errors and the fusion loses the diversity it depends on. Reported as a negative
result in `PLAN_PRIME.md`.
