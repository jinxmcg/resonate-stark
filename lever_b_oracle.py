"""Lever B, step 1 analysis: oracle best-of-readings on val, pairwise agreement, and how often each
reading is the unique best. Reads results_b/val_{plain,para}_r{1..4}.csv. Val only."""
import csv, ast, itertools, json, stark_shim
from stark_qa import load_qa
from metrics import stark_metrics, summarize
qa = load_qa("prime"); idx = qa.get_idx_split()["val"].tolist()
NAMES = {"r1": "learned anchors + exact names (P3)", "r2": "exact names only (regex type)", "r3": "learned anchors, low floor", "r4": "second-best answer type"}
for V in ("plain", "para"):
    P = {r: {int(x["query_id"]): ast.literal_eval(x["pred_rank"]) for x in csv.DictReader(open(f"results_b/val_{V}_{r}.csv"))} for r in NAMES}
    rows = {r: [] for r in NAMES}; oracle = []; best_count = {r: 0 for r in NAMES}; agree = {}
    for i in idx:
        q, qid, ans, _ = qa[i]; ms = {r: stark_metrics(P[r][int(qid)], ans) for r in NAMES}
        for r in NAMES: rows[r].append(ms[r])
        b = max(NAMES, key=lambda r: ms[r]["mrr"]); oracle.append(ms[b])
        if sum(1 for r in NAMES if ms[r]["mrr"] == ms[b]["mrr"]) == 1: best_count[b] += 1
    for a, b in itertools.combinations(NAMES, 2):
        agree[(a, b)] = sum(1 for i in idx if P[a][int(qa[i][1])][0] == P[b][int(qa[i][1])][0]) / len(idx)
    print(f"== {V} val")
    for r in NAMES: print(f"  {r} {NAMES[r]:38s}", {k: round(v * 100, 1) for k, v in summarize(rows[r]).items()}, f"| uniquely best on {best_count[r]}")
    print("  ORACLE best of four        ", {k: round(v * 100, 1) for k, v in summarize(oracle).items()})
    print("  top-1 agreement:", {f"{a}-{b}": round(v * 100) for (a, b), v in agree.items()})
