"""Lever B at inference: run the four deterministic readings of a split through the P3 pipeline, score each
with the MLP selector, pick one reading per question, write eval_results_{split}.csv. Also used for the
val dry run. Expects the readings' rankings and feature files produced by scripts/readings_for_split.sh.
Usage: PYTHONPATH=lib uv run python select_readings.py --split val [--score]"""
import argparse, json, csv, ast, numpy as np, torch, stark_shim
from stark_qa import load_qa
from metrics import stark_metrics, summarize
from lever_b_select import READ, feats, NT

def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", required=True); p.add_argument("--score", action="store_true"); p.add_argument("--tag", default="", help="reading file tag (e.g. plain / para for val)")
    a = p.parse_args(); human = a.split == "human_generated_eval"; qa = load_qa("prime", human_generated_eval=human)
    idx = qa.get_idx_split()[a.split].tolist() if not human else list(range(len(qa))); tag = a.tag or a.split
    P = {r: {int(x["query_id"]): ast.literal_eval(x["pred_rank"]) for x in csv.DictReader(open(f"results_b/{tag}_{r}.csv"))} for r in READ}
    SC = {r: json.load(open(f"data/scB_{tag}_{r}.json")) for r in READ}; LP = {r: (json.load(open(f"data/lpB_{tag}_{r}.json")) if r != "r2" else {}) for r in READ}
    TX = json.load(open(f"data/text_{tag}_bgeft2.json")); U = json.load(open(f"data/text_{tag}_pjoint.json"))
    ck = torch.load("models/selectorB_mlp.pt", map_location="cpu", weights_only=False); mu, sd = np.array(ck["mu"], np.float32), np.array(ck["sd"], np.float32)
    mlp = torch.nn.Sequential(torch.nn.Linear(len(mu), 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)); mlp.load_state_dict(ck["state"]); mlp.eval()
    rows, out, picks = [], [], [0] * 4
    for i in idx:
        q, qid, ans, _ = qa[i]; qid = int(qid); k = str(qid)
        F = [feats(qid, r, SC, LP) for r in READ]; tops = [P[r][qid][0] if P[r][qid] else -1 for r in READ]; t1 = [f[14] for f in F]; base_type_p1 = F[0][4]
        tx10 = TX.get(k, [])[:10]; u10 = U.get(k, [])[:10]
        for j, r in enumerate(READ):
            F[j] += [1.0 if tops[j] == tops[0] else 0.0, float(sum(1 for t in tops if t == tops[j]) - 1), t1[j] - t1[0], base_type_p1, 1.0 if tops[j] == tops[1] else 0.0]
            tt = int(NT[tops[j]]) if tops[j] >= 0 else -1
            F[j] += [sum(1 for c in tx10 if int(NT[c]) == tt) / 10.0, sum(1 for c in u10 if int(NT[c]) == tt) / 10.0, 1.0 if tops[j] in tx10 else 0.0, 1.0 if tops[j] in u10 else 0.0]
        with torch.no_grad(): s = mlp(torch.tensor((np.array(F, np.float32) - mu) / sd)).squeeze(-1).numpy()
        j = int(s.argmax()); picks[j] += 1; ranked = P[READ[j]][qid][:100]; out.append((i, qid, ranked))
        if a.score: rows.append(stark_metrics(ranked, ans))
    with open(f"eval_results_{a.split}.csv", "w") as f:
        f.write("idx,query_id,pred_rank\n")
        for i, qid, ranked in out: f.write(f'{i},{qid},"{ranked}"\n')
    print(f"wrote eval_results_{a.split}.csv with {len(out)} rows; picks {picks}")
    if a.score: print(f"COMMITTED {a.split} (selector):", {k: round(v, 4) for k, v in summarize(rows).items()})

if __name__ == "__main__":
    main()
