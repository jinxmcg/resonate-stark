import numpy as np
def stark_metrics(ranked, answers):
    """ranked: list of node ids (best first); answers: list of true ids. Returns dict."""
    a = set(answers); top = ranked[:20]
    hit1 = float(ranked[0] in a) if ranked else 0.0
    hit5 = float(any(x in a for x in ranked[:5]))
    rec20 = len([x for x in top if x in a]) / max(1, len(a))
    mrr = 0.0
    for i, x in enumerate(ranked[:100]):
        if x in a: mrr = 1.0 / (i + 1); break
    return {"hit1": hit1, "hit5": hit5, "recall20": rec20, "mrr": mrr}
def summarize(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ("hit1", "hit5", "recall20", "mrr")}
