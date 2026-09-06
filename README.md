# ResonatE on STaRK-Prime: one entity table that answers in graph terms and in language

This repository is the STaRK-Prime probe of [ResonatE](https://github.com/jinxmcg/resonate),
a knowledge-graph embedding with a unit-norm complex entity table and composable relation
operators. It is separate from the paper's OGB work on purpose: its own numbering (P1, P2),
its own plan file, its own reads of the test split, all in [`PLAN_PRIME.md`](PLAN_PRIME.md).

**Submitted system: "ResonatE (no llm)".** No generative language model is run when a question is
answered: nothing decodes tokens at query time. What does run is the entity table trained on
PrimeKG's edges *and* on the training questions, its 36 relation operators, the graph adjacency,
two 110M transformer encoders (a fine-tuned `bge-base-en-v1.5` text ranker, and the encoder under
the parser head and the question readout) and a logistic reranker: 32 ms median per question at
batch size 1 on one RTX 5090, 1.9 GiB of GPU memory (measured below). A 7B instruction model
(Qwen2.5-7B-Instruct) was used once, offline, to paraphrase training questions and to name
entities for weak labels; it is not needed to run or to reproduce anything here.

## Results

Committed reads of the STaRK-Prime test splits: one pipeline, frozen before the read, run once per
split after a dry run on validation reproduced the development number to four decimals.
`results_p2/eval_results_*.csv` are the files submitted. Metrics are Hit@1 / Hit@5 / Recall@20 /
MRR over the whole node set. Baseline rows are the STaRK leaderboard's own
(https://stark.stanford.edu, fetched from the leaderboard's source on 2026-09-06), i.e. the rows of
the STaRK paper (Wu et al., NeurIPS D&B 2024) and the AvaTaR paper (Wu et al., NeurIPS 2024). The
"regime" column is our reading of those papers: *zero-shot* = a pretrained retriever with no
STaRK-specific training; *trained* = fine-tuned or trained on the STaRK train split; *LLM at query
time* = a generative model runs on each query (reranking, or an agent). Ours is trained on the
train split (embedder fine-tune, joint table, parser head, reranker) with offline LLM augmentation
(paraphrases and weak labels from a local 7B model) and no generative model at query time.

**Synthesized (full), 2,801 questions**

| method | regime | Hit@1 | Hit@5 | R@20 | MRR |
|---|---|---|---|---|---|
| BM25 | zero-shot | 12.75 | 27.92 | 31.25 | 19.84 |
| DPR (roberta) | trained | 4.46 | 21.85 | 30.13 | 12.38 |
| ANCE (roberta) | trained | 6.53 | 15.67 | 16.52 | 11.05 |
| QAGNN (roberta) | trained | 8.85 | 21.35 | 29.63 | 14.73 |
| ada-002 | zero-shot | 12.63 | 31.49 | 36.00 | 21.41 |
| voyage-l2-instruct | zero-shot | 10.85 | 30.23 | 37.83 | 19.99 |
| LLM2Vec | zero-shot | 10.10 | 22.49 | 26.34 | 16.12 |
| GritLM-7b | zero-shot | 15.57 | 33.42 | 39.09 | 24.11 |
| multi-ada-002 | zero-shot | 15.10 | 33.56 | 38.05 | 23.49 |
| ColBERTv2 | zero-shot | 11.75 | 23.85 | 25.04 | 17.39 |
| AvaTaR (claude-3-opus) | LLM at query time, prompts optimised on train | 18.44 | 36.73 | 39.31 | 26.73 |
| AvaTaR (gpt-4-turbo) | LLM at query time, prompts optimised on train | 20.10 | 39.89 | 42.23 | 29.18 |
| **ResonatE (no llm)** | trained + offline LLM augmentation | **41.81** | **68.30** | **74.77** | **53.66** |

**Synthesized (10%), 280 questions**

| method | regime | Hit@1 | Hit@5 | R@20 | MRR |
|---|---|---|---|---|---|
| BM25 | zero-shot | 13.93 | 31.07 | 32.84 | 21.68 |
| DPR (roberta) | trained | 5.00 | 23.57 | 30.50 | 13.50 |
| ANCE (roberta) | trained | 6.78 | 16.15 | 17.07 | 11.42 |
| QAGNN (roberta) | trained | 7.14 | 17.14 | 32.95 | 16.27 |
| ada-002 | zero-shot | 15.36 | 31.07 | 37.88 | 23.50 |
| voyage-l2-instruct | zero-shot | 12.14 | 31.42 | 37.34 | 21.23 |
| LLM2Vec | zero-shot | 9.29 | 20.70 | 25.54 | 15.00 |
| GritLM-7b | zero-shot | 16.79 | 34.29 | 41.11 | 24.99 |
| multi-ada-002 | zero-shot | 15.36 | 32.86 | 40.99 | 23.70 |
| ColBERTv2 | zero-shot | 15.00 | 26.07 | 27.78 | 19.98 |
| Claude3 Reranker | LLM at query time (reranks ada-002 top-20) | 17.79 | 36.90 | 35.57 | 26.27 |
| GPT4 Reranker | LLM at query time (reranks ada-002 top-20) | 18.28 | 37.28 | 34.05 | 26.55 |
| **ResonatE (no llm)** | trained + offline LLM augmentation | **41.79** | **71.07** | **75.90** | **54.31** |

**Human-generated, 98 questions** (our row with 95% bootstrap confidence intervals over questions;
the baselines are single numbers from the leaderboard, whose intervals on 98 questions are of the
same width, so differences of a few points are not significant)

| method | regime | Hit@1 | Hit@5 | R@20 | MRR |
|---|---|---|---|---|---|
| BM25 | zero-shot | 22.45 | 41.84 | 42.32 | 30.37 |
| DPR (roberta) | trained | 2.04 | 9.18 | 10.69 | 7.05 |
| ANCE (roberta) | trained | 7.14 | 13.27 | 11.72 | 10.07 |
| QAGNN (roberta) | trained | 6.12 | 13.27 | 17.62 | 9.39 |
| ada-002 | zero-shot | 17.35 | 34.69 | 41.09 | 26.35 |
| voyage-l2-instruct | zero-shot | 16.33 | 32.65 | 39.01 | 24.33 |
| LLM2Vec | zero-shot | 9.18 | 21.43 | 26.77 | 15.24 |
| GritLM-7b | zero-shot | 25.51 | 41.84 | 48.10 | 34.28 |
| multi-ada-002 | zero-shot | 24.49 | 39.80 | 47.21 | 32.98 |
| ColBERTv2 | zero-shot | 15.31 | 26.53 | 25.56 | 19.67 |
| Claude3 Reranker | LLM at query time | 28.57 | 46.94 | 41.61 | 36.32 |
| GPT4 Reranker | LLM at query time | 28.57 | 44.90 | 41.61 | 34.82 |
| AvaTaR (gpt-4-turbo) | LLM at query time, prompts optimised on train | 33.03 | 51.37 | 53.34 | 41.00 |
| **ResonatE (no llm)** | trained + offline LLM augmentation | 30.61 [21.4, 39.8] | 53.06 [42.9, 63.3] | 60.58 [51.6, 69.4] | 41.74 [33.9, 50.3] |

Reading the human table honestly: our Hit@1 is 2.4 points below AvaTaR, our Recall@20 is 7 points
above it, and Hit@5 and MRR are within the interval width. On the two synthesized splits the margin
over every row is far outside any interval.

Our first read (P1, tag `p1-frozen`: hand-written parser, off-the-shelf embedder) was 28.7 / 28.2 /
20.4 Hit@1. Both reads are reported; nothing was tuned after the second. Development used the
train split for fitting and the validation split for every decision; a paraphrased copy of the
validation questions, written by a local 7B model from the question text alone, served as a proxy
for human phrasing.

**Metric definition and the official evaluator.** `metrics.py` computes the four metrics over the
top-100 list and counts an answer outside the top-100 as reciprocal rank 0. The leaderboard scores
the same CSV with `stark_qa`'s `Evaluator` (top-100 ids with scores −i, every other candidate tied
below), under which an answer outside the top-100 can only add up to 1/101 by tie order.
`eval_check.py` runs both on the validation predictions of the submitted pipeline: Hit@1, Hit@5 and
Recall@20 agree on every query; MRR agrees to four decimals in the mean (0.5307 both), with a
largest per-query difference of 0.0016 from that tie effect. Rescoring the three committed files
with the official `Evaluator` gives the same numbers as above to two decimals (test 41.81 / 68.30 /
74.77 / 53.66; 95% bootstrap CIs over questions: Hit@1 [40.0, 43.7], MRR [52.2, 55.3]; test-0.1
Hit@1 [36.1, 47.5], MRR [49.5, 59.0]; human as in the table).
Note: `torchmetrics` 1.9.0 returns 0 for `retrieval_reciprocal_rank` and `retrieval_recall` on
valid inputs (the functional retrieval imports are deprecated there); the check pins
`torchmetrics==1.4.0`, under which the tiny sanity cases give the expected values.

**Query-time cost** (`bench.py`, one RTX 5090, batch size 1, 300 validation questions, CUDA-synchronised, first query excluded):

| stage | what runs | median | p90 |
|---|---|---|---|
| parse | 110M encoder + three heads, nearest-entity anchors | 2.8 ms | 2.9 ms |
| walk + model | adjacency walk from the anchors (CPU) + operator scoring over the table | 21.7 ms | 38.5 ms |
| text ranker | 110M encoder + top-100 over 129,375 node embeddings | 4.2 ms | 4.6 ms |
| question readout | 110M encoder + head + readout against the table | 2.7 ms | 2.8 ms |
| fusion + reranker | numpy | 0.3 ms | 0.4 ms |
| **total** | | **32.0 ms** | **48.8 ms** |

GPU memory: 1.88 GiB after loading all components, 1.92 GiB peak; models load in 1.8 s. Parameters:
graph table + operators 37.3M, joint table + operators 37.3M, parser encoder + heads 110.2M,
question encoder + head 109.7M, text ranker 109.5M. Batched, the encoders and the readout cost under
0.5 ms per question (2,241 questions in 1.0 s and 0.8 s); the walk is per query and CPU-side and is
the cost that remains. The whole committed test read (2,801 questions, all stages, including model
loading) took 114 s of wall-clock.

## Why this is interesting

**The same table serves three tasks.** The entity table was trained on the graph's 8.1M edges
(held-out link MRR 0.557), then continued on edges and 12,324 questions together. It ended at link
MRR 0.568 (three seeds, sd 0.001), better than before, and it now answers a written question from
the text alone, with the same readout used for link prediction:

| standalone readout, question in, entity out (val, 3 seeds) | Hit@1 | MRR |
|---|---|---|
| synthesized wording | 26.7 ± 0.2 | 32.2 ± 0.2 |
| human-style paraphrases | 25.7 ± 0.4 | 31.3 ± 0.2 |

No parser, no index, one matrix product; its sensitivity to phrasing is under one point.

**A learned reader instead of a language model.** A three-headed parser (answer type, anchor
entities found by nearest neighbour in the table, relation operators) trained in four minutes on
weak labels derived from the graph reads questions into the model's query space. On the graph
path alone it scores 25.6 ± 0.6 Hit@1 on synthesized and 22.3 ± 0.4 on human-style wording,
against 24.9 / 20.9 for a hand-written parser backed by a 7B instruction model. Inside the full
pipeline the 7B model is worth about two points of Hit@1 and nothing on Hit@5 or Recall@20; the
submitted system leaves it out.

**Where the model does and does not do work, measured.** With the same parser and anchors:
the exact graph walk alone gives 18.9 Hit@1 on val, ResonatE's operator composition alone 19.0,
and the two together 24.9. The walk decides which candidates are graph-supported; the model
decides their order. Inside the final reranker the model's raw score adds nothing once the walk
and the text ranker are present; its contribution enters through the candidate order and, above
all, through the language readout, which is the largest single lever in the pipeline
(+8 Hit@1 with honest, out-of-fold features).

**Is it the table, or would any table do?** (a reviewer's test) RotatE was trained on the same graph
at the same width (288 numbers per entity) under the same regime, tuned by held-out link MRR only
(0.549 vs ResonatE's 0.557), and the identical projector was trained on top of both tables, frozen
and jointly. Frozen: RotatE 15.8 Hit@1 (inner-product readout; 15.5 with its own distance readout)
vs ResonatE 19.7. Joint: RotatE 21.2 vs ResonatE 26.9, and joint training raised RotatE's link MRR by
0.003 against 0.013 for ResonatE. One seed each, one dataset, RotatE's L2 variant; our first pass
with an untuned RotatE showed a 2x gap, which was mostly the control and is retracted in
`PLAN_PRIME.md`. The honest claim: this table takes a language projection better than a RotatE
table under the same recipe, not that only this table can.

**What did not work**, all on validation: fine-tuning the text encoder for anchor resolution
(worse: the fine-tune points a question at its answer, not at the entity it names); the model's
raw score as a reranker feature (no gain); text-to-latent with in-sample training features
(a 2-point loss, fixed by 5-fold out-of-fold features); longer joint training with more
paraphrases (recall drops as the head memorises answer sets); four query vectors instead of one
(+1 standalone, no gain in the pipeline).

## The pipeline, in order

1. `latent_parser.py` reads the question: answer type, up to three anchor entities, relation hints.
2. `retrieve.py` walks the adjacency from each anchor along the operator chains the type
   signatures allow (exact support, weight 30) and orders candidates with the model's composed
   operators; questions without an anchor fall back to a text-resolved anchor.
3. `embed_text2.py` ranks node descriptions with `bge-base-en-v1.5` fine-tuned on train pairs.
4. `t2l_rank.py` ranks entities directly from the question with the joint table's readout.
5. `predict.py` fuses the three lists by reciprocal rank (weight chosen on train) and reorders
   the candidates with a per-answer-type logistic reranker (`rerank.py`) fit on train with
   out-of-fold features.

The dry run of this exact command sequence on validation reproduces the development number
(41.28 / 67.25 / 75.29 / 53.07) before any test question is read (`committed_read_p2.sh`).

## Reproduce

Everything below is in this repository; the chain scripts are in `scripts/` and run from any
checkout (`cd "$(dirname "$0")/.."`). Two ways in.

**From the released weights** (release `v2-no-llm`, checksums in `CHECKSUMS.txt`):

```bash
git clone https://github.com/jinxmcg/resonate-stark && cd resonate-stark
uv sync                                   # pyproject pins torch 2.6.0 (cu124) for a Pascal card; on a Blackwell card use torch>=2.7
uv run python build_kg.py                 # PrimeKG + STaRK-Prime via stark_qa -> data/kg.npz, names.json, docs.jsonl, dicts.json, splits.json, signatures.json
REL=https://github.com/jinxmcg/resonate-stark/releases/download/v2-no-llm
curl -L -o models_p2.tar.gz $REL/models_p2.tar.gz && curl -L -o data/doc_emb_bgeft2.npy $REL/doc_emb_bgeft2.npy
sha256sum -c CHECKSUMS.txt                # 2a5ccf66… models_p2.tar.gz, eeaa88a5… doc_emb_bgeft2.npy
tar xzf models_p2.tar.gz                  # -> models/{p_k12b4_50k.pt, bge_ft2/, p_joint.pt, p_joint_head.pt, p_joint_enc/, lp.pt}
ln -s p_joint_enc models/p_joint_head_enc
scripts/committed_read_p2.sh              # dry run on val (must print 41.28 / 67.25 / 75.29 / 53.07), then test, test-0.1, human -> results_p2/
```

**From scratch** (one RTX 5090; times measured):

```bash
uv run python train_prime.py --k 12 --block-size 4 --steps 50000 --save models/p_k12b4_50k.pt    # graph model, ~4 min
# offline 7B uses (outputs are committed in data/, so these two are optional):
uv run python paraphrase.py --split train                                 # -> data/para_train.json   (5 min)
uv run python paraphrase.py --split val                                   # -> data/para_val.json     (2 min)
uv run python llm_parse.py --split train --batch 12 --tag train           # -> data/llmparse_train.json (weak-label entity names)
uv run python llm_parse.py --split train --queries data/para_train.json --batch 12 --tag train_para
uv run python finetune_embed.py --extra data/para_train.json --out models/bge_ft2                # 5 min
uv run python embed_text2.py --model bgeft2 --splits train,val            # corpus embedding + train/val text rankings
uv run python joint_train.py --tag p_joint --extra data/para_train.json   # one table for edges + questions, 5 min
uv run python latent_parser.py --train --tag lp                           # parser head, 4 min (+3 min of weak labels)
uv run python latent_parser.py --predict val --tag lp --out data/lparse_val.json
uv run python latent_parser.py --predict val --tag lp --queries data/para_val.json --out data/lparse_val_para.json
scripts/oof_chain.sh          # 5-fold out-of-fold text rankings (bge fine-tunes) and questions-only text-to-latent, ~30 min
scripts/joint_oof_chain.sh    # 5-fold out-of-fold rankings from the joint model, ~10 min
scripts/lp_pipe_chain.sh      # 5-fold out-of-fold parsers, retrieval on train/val, reranker fit -> data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json, ~25 min
scripts/committed_read_p2.sh  # the read
```

`PYTHONPATH=lib` is set by the scripts; set it yourself for single commands. Training is seeded but
CUDA kernels are not bit-deterministic, so a retrained pipeline reproduces the numbers to seed
noise (three seeds of the joint model and the parser are in `PLAN_PRIME.md`), not bit-for-bit; the
released weights reproduce the prediction files exactly. The shared model code (`resonate.py`,
`resonate_wiki.py`, `rowadagrad.py`) is vendored in `lib/` from the main repository. Other scripts
in `scripts/` are the ablations and negative results recorded in `PLAN_PRIME.md`
(`model_vs_walk.sh`, `lever6_chain.sh`, `seeds_chain.sh`, `final_eval.sh`, …).

## Protocol

- The test splits were read twice in total: once by P1 (28.7 / 28.2 / 20.4 Hit@1) and once by
  the submitted P2 pipeline. Both are in `PLAN_PRIME.md` with the pre-registration written before
  each read. The test splits' answers are used only inside `predict.py --score`; their question
  text is read only to produce predictions.
- The human-generated set was never used for development; the paraphrase proxy stands in for it.
- Every reranker feature is out-of-fold on train. The in-sample variant, and its cost, are recorded.
- No leaderboard rule was bent: train for training, validation for tuning, test for the reported
  numbers, no external data beyond PrimeKG and the STaRK question files.

## Layout

`build_kg.py` (data) · `train_prime.py` (graph model) · `retrieve.py` (parser, walk, model scoring)
· `llm_parse.py`, `paraphrase.py` (offline 7B uses) · `finetune_embed.py`, `embed_text2.py` (text)
· `text2latent.py`, `joint_train.py`, `t2l_rank.py` (language readout) · `latent_parser.py` ·
`rerank.py`, `predict.py`, `fuse.py` · `oof_embed.py` · `scripts/*.sh` (chains) · `eval_check.py`, `bench.py` (audit) ·
`results/` (P1 read) · `results_p2/` (submitted files) · `logs/` · `PLAN_PRIME.md` (everything).

License: MIT. Author: Cristian Malaia, with Claude Fable 5 (Anthropic) as pair programmer.
