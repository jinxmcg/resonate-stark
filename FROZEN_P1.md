# P1 frozen pipeline (2026-09-05)
model:     models/p_k12b4_50k.pt  (train_prime.py --k 12 --block-size 4 --steps 50000, seed 0, sparse shell)
parser:    retrieve.py v2 (exact name mentions + gene symbols, answer type from the question head,
           one- and two-hop operator chains from PrimeKG type signatures, keyword hints, "no drugs" filter)
ranking:   retrieve.py --beta 30 --anchor bge --anchor-top 2 --anchor-weight 0.7
text:      embed_text2.py --model qwen (Qwen3-Embedding-0.6B), corpus embeddings data/doc_emb_qwen.npy
fusion:    RRF, w = 0.5 (fuse.py, chosen on train), predict.py
reads:     test 28.7 / 51.9 / 59.9 / 39.1; test-0.1 28.2 / 50.7 / 59.9 / 38.5; human 20.4 / 41.8 / 48.6 / 29.9
           (results/eval_results_*.csv, results/committed_*.log)
