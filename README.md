# ResonatE on STaRK-Prime: one entity table that answers in graph terms and in language

This repository is the STaRK-Prime probe of [ResonatE](https://github.com/jinxmcg/resonate),
a knowledge-graph embedding with a unit-norm complex entity table and composable relation
operators. It is separate from the paper's OGB work on purpose: its own numbering (P1, P2),
its own plan file, its own reads of the test split, all in [`PLAN_PRIME.md`](PLAN_PRIME.md).

**Submitted system: "ResonatE (no llm)".** No language model runs when a question is answered.
At query time the system is the entity table trained on PrimeKG's edges *and* on the training
questions, its 36 relation operators, one small parser head, the graph adjacency, a fine-tuned
110M text encoder and a logistic reranker. A 7B instruction model was used once, offline, to
paraphrase training questions and to name entities for weak labels.

## Results

Committed reads of the STaRK-Prime test splits (one pipeline, frozen before the read, run once per
split; `results_p2/eval_results_*.csv` are the files submitted):

| split | Hit@1 | Hit@5 | Recall@20 | MRR | best published row (per column) |
|---|---|---|---|---|---|
| Synthesized (full, 2,801) | **41.81** | **68.30** | **74.77** | **53.66** | 20.10 / 39.89 / 42.23 / 29.18 |
| Synthesized (10%, 280) | **41.79** | **71.07** | **75.90** | **54.31** | 18.28 / 37.28 / 41.11 / 26.55 |
| Human-generated (98) | 30.61 | **53.06** | **60.58** | **41.74** | 33.03 / 51.37 / 53.34 / 41.00 |

On the human-written set Hit@1 is 2.4 points (about two questions) behind AvaTaR (gpt-4-turbo),
and ahead on the other three metrics. Reference rows are the STaRK leaderboard's own
(`stark.stanford.edu`, September 2026).

Our first read, with a hand-written parser and an off-the-shelf text embedder (P1, frozen tag
`p1-frozen`), was 28.7 / 28.2 / 20.4 Hit@1. Both reads are reported; nothing was tuned after the
second. Development used the train split for fitting and the validation split for every decision;
a paraphrased copy of the validation questions, written by a local 7B model from the question
text alone, served as a proxy for human phrasing.

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

```bash
uv sync                                      # torch 2.6 (cu124) on jinx; torch>=2.7 on Blackwell cards
uv run python build_kg.py                    # PrimeKG + STaRK-Prime via stark_qa -> data/ (type signatures: data/signatures.json)
uv run python train_prime.py --k 12 --block-size 4 --steps 50000 --save models/p_k12b4_50k.pt
uv run python finetune_embed.py --extra data/para_train.json --out models/bge_ft2
uv run python joint_train.py --tag p_joint --extra data/para_train.json
uv run python latent_parser.py --train --tag lp
./oof_chain.sh && ./joint_oof_chain.sh && ./lp_pipe_chain.sh     # out-of-fold features + reranker
./committed_read_p2.sh                                             # dry run on val, then the reads
```

Trained weights (`models/`: `p_k12b4_50k.pt`, `bge_ft2/`, `p_joint.pt` + `p_joint_enc/`,
`lp.pt`) and the corpus embedding are in the GitHub release; with them, the last command alone
reproduces the prediction files. Paraphrases and LLM parses (`data/para_*.json`,
`data/llmparse_*.json`) are committed so the 7B model is not needed to reproduce anything.
The shared model code (`resonate.py`, `resonate_wiki.py`, `rowadagrad.py`) is vendored in `lib/`
from the main repository. Hardware: one RTX 5090; the submitted pipeline trains in well under an
hour of GPU time (graph model 50k steps, embedder fine-tune 5 min, joint model 5 min, parser 4 min,
out-of-fold folds ~15 min); the ablations, seeds and paraphrase generation add a few GPU-hours.

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
`rerank.py`, `predict.py`, `fuse.py` · `oof_embed.py` and the `*_chain.sh` scripts ·
`results/` (P1 read) · `results_p2/` (submitted files) · `logs/` · `PLAN_PRIME.md` (everything).

License: MIT. Author: Cristian Malaia, with Claude Fable 5 (Anthropic) as pair programmer.
