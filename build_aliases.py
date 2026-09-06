"""P4 lever A: alias -> entity ids from the node text. Genes: the NCBI alias list and the full gene
name; other types: nothing yet (no synonym field in their text). Writes data/aliases.json
{alias_lower: [ids]} with aliases of length >= 3, at most 3 ids each, skipping any alias that is
already an exact node name (names win). Usage: uv run python build_aliases.py"""
import json, re, ast, collections, numpy as np
docs = {d["id"]: d["text"] for d in map(json.loads, open("data/docs.jsonl"))}
names = json.load(open("data/names.json")); nt = np.load("data/kg.npz")["node_type"]
tn = {int(k): v for k, v in json.load(open("data/dicts.json"))["node_type_dict"].items()}
exact = {n.strip().lower() for n in names.values()}
al = collections.defaultdict(set); fn = collections.defaultdict(set); n_gene = 0
for i, t in docs.items():
    if tn[int(nt[i])] != "gene/protein": continue
    m = re.search(r"alias \(other gene names\): (\[.*?\])", t)
    cands = []
    if m:
        try: cands += [str(x) for x in ast.literal_eval(m.group(1))]
        except Exception: pass
    m2 = re.search(r"name \(gene name\): ([^\n]+?)(?:\s+- |\n|$)", t)
    if m2:
        c = m2.group(1).strip().lower()
        if len(c) >= 5 and c not in exact: fn[c].add(int(i))          # full gene names: matched like names (case-insensitive, word-bounded)
    for c in cands:
        c = c.strip().lower()
        if len(c) >= 2 and c not in exact and not c.isdigit(): al[c].add(int(i))   # alias symbols: matched ONLY as uppercase tokens in the question
    n_gene += bool(cands) or bool(m2)
out = {"symbols": {k: sorted(v)[:3] for k, v in al.items() if len(v) <= 3}, "names": {k: sorted(v)[:3] for k, v in fn.items() if len(v) <= 3}}
json.dump(out, open("data/aliases.json", "w"))
print(f"genes with alias text: {n_gene}; alias symbols kept: {len(out['symbols'])}; full gene names kept: {len(out['names'])}")
