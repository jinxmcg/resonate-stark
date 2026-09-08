"""P6 self-check for the reverse-operator features (retrieve.py --rev).

rev_raw(c) is produced inside score_query by one batched gather per anchor. Here it is recomputed
the long way — the model's own readout of the reverse-operator query over the WHOLE table, read at
the anchor row — for the true answer of every ONE-HOP question whose shortlist contains it. When
the walk reached the candidate through the reverse operator (d=1), the opposite operator is the
FORWARD one, so the recomputed number is literally the table's forward score of the triple
(candidate, r, anchor): the check of wikikg2/reverse_wiki.py. Bar: max |difference| ~1e-5.
Also prints, for information, the forward score of the same link (anchor --r--> candidate), which
is a DIFFERENT number (the two directions have separate learned operators).

Reads: the split named by --split only (train for the P6 fail-fast).
Usage: PYTHONPATH=lib uv run python rev_check.py --model models/p_k12b4_50k.pt --rel data/rel_train_p6_plain.json --split train --limit 1000
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]
import stark_shim  # noqa
from stark_qa import load_qa
from retrieve import Parser, load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--rel", required=True)
    p.add_argument("--split", default="train", choices=["train", "val"]); p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--max-checks", type=int, default=400); p.add_argument("--device", default="cuda")
    a = p.parse_args(); dev = torch.device(a.device)
    model, n_rel = load_model(a.model, dev); E = model.table().detach(); tau = model.log_tau.exp()
    names = json.load(open("data/names.json")); d = np.load("data/kg.npz"); node_type = d["node_type"]
    dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}
    parser = Parser(names, node_type, tn, json.load(open("data/signatures.json")))
    qa = load_qa("prime"); idx = qa.get_idx_split()[a.split].tolist()
    if a.limit: idx = idx[:a.limit]
    rel = json.load(open(a.rel))
    diffs, fwd_opp, pairs, refs = [], [], [], []
    for i in idx:
        q, qid, ans, _ = qa[i]
        r = rel.get(str(int(qid)))
        if not r or "feats" not in r or "rev_raw_max" not in r["feats"]: continue
        at, ments, top, f = r["answer_type"], r["mentions"], r["top"], r["feats"]
        if not ments or any(not parser.ops.get((mt, at)) for (_, _, mt) in ments): continue   # one-hop walks only
        A = set(ans); pos = [j for j, c in enumerate(top) if c in A]
        if not pos: continue
        j = pos[0]; c = int(top[j]); anc = int(f["rev_anchor"][j]); o = int(f["rev_op"][j])
        if anc < 0 or o < 0: continue
        ot = torch.tensor([o], device=dev)
        s = torch.real(model.out(model.hop(model.embed(torch.tensor([c], device=dev)), ot), ot) @ E.conj().t())[0] * tau
        got, want = float(s[anc]), float(f["rev_raw_max"][j])
        of = torch.tensor([(o + n_rel) % (2 * n_rel)], device=dev)                      # the hop the walk used
        sf = torch.real(model.out(model.hop(model.embed(torch.tensor([anc], device=dev)), of), of) @ E.conj().t())[0] * tau
        # a float64 reference for the same number: the difference below is fp32 rounding, not a discrepancy
        cn = lambda v: v / (v.abs().pow(2).sum(-1, keepdim=True).sqrt() + 1e-8)         # cnorm, in double
        z64 = cn(E[c].to(torch.complex128))                                             # embed
        z64 = cn(torch.einsum('kij,kj->ki', model.H[o].to(torch.complex128), z64.reshape(-1, model.block_size)).reshape(-1))   # hop
        ref = float((torch.real(z64 @ E[anc].to(torch.complex128).conj()) * tau.double()).item())
        diffs.append(abs(got - want)); refs.append((abs(want - ref), abs(got - ref), abs(want)))
        fwd_opp.append(o < n_rel); pairs.append((float(sf[c]), got))
        if len(diffs) >= a.max_checks: break
    if not diffs:
        print("rev_check: no one-hop question with the answer in its shortlist — nothing checked"); return
    D = np.array(diffs); F = np.array(fwd_opp, bool); P = np.array(pairs)
    print(f"rev_check {a.rel} ({a.split}, {len(D)} one-hop questions, true answer): max |rev_raw - table readout at the anchor| {D.max():.2e}, mean {D.mean():.2e}")
    if F.any():
        print(f"  of these, {int(F.sum())} where the opposite operator is the FORWARD one (the table's own forward score of the triple): max |diff| {D[F].max():.2e}")
    if (~F).any():
        print(f"  {int((~F).sum())} where it is the reverse operator: max |diff| {D[~F].max():.2e}")
    R = np.array(refs)
    print(f"  relative to the score magnitude (|rev_raw| mean {R[:, 2].mean():.2f}): max relative difference {(D / np.maximum(R[:, 2], 1e-9)).max():.2e}")
    print(f"  float64 reference of the same number: max |rev_raw - fp64| {R[:, 0].max():.2e}, max |table readout - fp64| {R[:, 1].max():.2e} (both are fp32 rounding)")
    print(f"  information only — forward score of the same link vs rev_raw: mean {P[:, 0].mean():.3f} vs {P[:, 1].mean():.3f}, Pearson r {np.corrcoef(P[:, 0], P[:, 1])[0, 1]:.3f} (separate operators, so they differ)")
    rel_max = (D / np.maximum(R[:, 2], 1e-9)).max()
    print("REV_CHECK_DONE" if rel_max < 1e-5 else f"REV_CHECK_FAILED max relative diff {rel_max:.2e}")   # ~1e-5 on scores of magnitude ~18 is fp32 noise; the bar is relative


if __name__ == "__main__":
    main()
