"""P8: reduce the document matrix from 768 dimensions to d by a truncated SVD of the corpus.

The 129,375 x 768 fp16 document matrix is 99.4M of the pipeline's parameters — nearly a whole
encoder. A rank-d basis of the corpus itself (no query labels, no split is read) replaces it with a
129,375 x d matrix plus a 768 x d projection: at d=256 that is 33.3M instead of 99.4M. Documents and
queries are both projected by the same basis; inner products, not renormalised cosines, order the
candidates (the projection approximates the original inner product).

Usage: PYTHONPATH=lib uv run python proj_docs.py --emb data/doc_emb_bgeft2.npy --dims 128,256,384
"""
import argparse, numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/doc_emb_bgeft2.npy")
    p.add_argument("--dims", default="128,256,384")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(); dev = torch.device(a.device)
    D = torch.from_numpy(np.load(a.emb).astype(np.float32)).to(dev)
    print(f"docs {tuple(D.shape)} ({D.numel()/1e6:.1f}M parameters)", flush=True)
    # right singular vectors of the (uncentred) document matrix: the rank-d subspace that best
    # preserves the documents themselves, hence their inner products with any query
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    tag = a.emb.split("doc_emb_")[-1].replace(".npy", "")
    for d in (int(x) for x in a.dims.split(",")):
        V = Vh[:d].t().contiguous()                                    # (768, d)
        P = D @ V
        np.save(f"data/docproj_{tag}_{d}.npy", V.cpu().numpy().astype(np.float32))
        np.save(f"data/doc_emb_{tag}_p{d}.npy", P.cpu().numpy().astype(np.float16))
        n = P.numel() + V.numel()
        print(f"d={d}: energy kept {energy[d-1]:.4f}  params {n/1e6:.1f}M (from {D.numel()/1e6:.1f}M, -{100*(1-n/D.numel()):.0f}%)"
              f"  -> data/doc_emb_{tag}_p{d}.npy + data/docproj_{tag}_{d}.npy", flush=True)
    print("PROJ_DONE")


if __name__ == "__main__":
    main()
