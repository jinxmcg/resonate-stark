"""P1 relational retriever for STaRK-Prime queries.

parse(query) -> answer node type + a list of anchor mentions (node ids found by exact name
match in the query text, longest match first) + for each mention the candidate operators:
every (relation, direction) whose PrimeKG type signature connects the mention's type to the
answer type (data/signatures.json), weighted 2x when a relation keyword is present in the
query (K below), else 1x.

score(query) -> for each mention and each candidate operator, the compiled operator applied
to the mention row, one readout over all 129,375 rows, z-scored over the answer-type rows;
weighted max over operators per mention, summed over mentions; candidates outside the
answer type get -inf. Returns the top-100 node ids.

Evaluated on STaRK `val` only (P1); writes data/rel_val.json with per-query rankings so the
fusion step can combine them with the text retriever without re-scoring.
Usage: uv run python retrieve.py --model models/p_k12b4_12k.pt [--split val] [--limit N]
"""
import os, argparse, json, re, sys, time, collections
import numpy as np, torch, torch.nn.functional as F
import scipy.sparse as sps
sys.path[:0] = ["/mnt/geocore/resonate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")]   # shared modules on jinx, or prime/lib on a rented box
from resonate_wiki import SparseTableResonatE
import stark_shim  # noqa
from stark_qa import load_qa
from metrics import stark_metrics, summarize

# answer type from the question head
TYPE_PATTERNS = [
    (r"\b(gene|genes|protein|proteins|gene/protein|genes/proteins|genes or proteins|genetic targets?|targets?|receptors?|enzymes?)\b", "gene/protein"),
    (r"\b(drug|drugs|medication|medications|medicine|medicines|compound|compounds|treatment options|supplements?|agents?|therapies|therapy)\b", "drug"),
    (r"\b(disease|diseases|condition|conditions|illness|illnesses|disorder|disorders|syndrome|syndromes)\b", "disease"),
    (r"\b(anatomical|anatomy|tissue|tissues|organ|organs|body part|body parts|structures)\b", "anatomy"),
    (r"\b(phenotyp\w*|effect|effects|symptom|symptoms|manifestation\w*|clinical feature\w*)\b", "effect/phenotype"),
    (r"\b(pathway|pathways|subprocess\w*|signaling cascades?)\b", "pathway"),
    (r"\b(biological process\w*|process|processes)\b", "biological_process"),
    (r"\b(cellular component\w*|cell structures?|cellular structures?|component\w*|organelle\w*)\b", "cellular_component"),
    (r"\b(molecular function\w*|function|functions|activity|activities)\b", "molecular_function"),
    (r"\b(exposure\w*|environmental\w*)\b", "exposure"),
]
# relation keyword hints (relation id -> patterns); a hit doubles that relation's weight
K = {
    0: r"interact\w* with|ppi|protein.protein|binds?\b|binding",
    1: r"carrier",
    2: r"enzym\w*|metaboli\w*",
    3: r"target\w*|inhibit\w*|agonist|antagonist|modulat\w*",
    4: r"transport\w*",
    5: r"contraindicat\w*|should not be (treated|used|taken)|not be prescribed|avoid\w*",
    6: r"indicat\w*|treat\w*|therap\w*|medication for|drugs? for|prescribed for|manage\w*",
    7: r"off.label",
    8: r"synergi\w*|combin\w* with|taken (together|with)|complement\w*",
    9: r"associated with|association|linked to|related to|involved in .* (disease|disorder)",
    10: r"subtype|sub-type|parent|child|categori[sz]ed|classif\w*|variant of|form of|type of|broader|narrower|specific",
    11: r"absent|lack\w* the phenotype|without .* phenotype|does not present",
    12: r"phenotyp\w*|symptom\w*|present\w* with|manifest\w*|feature\w*|characteri[sz]ed by",
    13: r"side effect\w*|adverse|reaction\w*",
    14: r"interact\w* with|involved in|participat\w*|part of|function\w* in|role in|located in|found in|component of|activity",
    15: r"linked to|exposure|exposed",
    16: r"express\w*|present in|found in|located in",
    17: r"lack\w* the expression|not express\w*|absent|no expression|do not express|does not express|don't express",
}


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    m = SparseTableResonatE(ck["N"], 2 * ck["n_rel"], k=a["k"], block_size=a["block_size"], sparse_grad=False, device=dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["n_rel"]


class Parser:
    def __init__(self, names, node_type, type_names, sigs, min_len=4, short_symbols=True, aliases=None):
        self.node_type = node_type; self.tn = type_names
        self.by_name = collections.defaultdict(list)
        self.by_symbol = collections.defaultdict(list)      # P3 fix: 2-3 character gene/protein symbols (GCK, TTR, ...) — matched only by the uppercase-symbol rule
        gene_t = {i for i, t in type_names.items() if t == "gene/protein"}
        for i, n in names.items():
            n = n.strip().lower()
            if len(n) >= min_len and not n.isdigit():
                self.by_name[n].append(int(i))
            elif short_symbols and 2 <= len(n) < min_len and n.isalnum() and not n.isdigit() and int(node_type[int(i)]) in gene_t:
                self.by_symbol[n].append(int(i))
        self.alias_symbols = {}
        if aliases:                                          # P4 lever A: full gene names resolve like names; alias symbols only as uppercase tokens
            for a_, ids in aliases.get("names", {}).items():
                if a_ not in self.by_name: self.by_name[a_] = list(ids)
            for a_, ids in aliases.get("symbols", {}).items():
                if a_ not in self.by_name and a_ not in self.by_symbol: self.alias_symbols[a_] = list(ids)
        self.names_sorted = sorted(self.by_name, key=len, reverse=True)
        # substring index over the names: mentions() used to test all ~129k names per question with
        # `n in ql`, one Python operation each, and that dominated retrieval wall time. The automaton
        # returns exactly the names that `n in ql` would accept; they are then walked in the SAME
        # order (names_sorted) through the same boundary regex, so the output is unchanged.
        self.name_rank = {n: i for i, n in enumerate(self.names_sorted)}
        self.aut = None
        try:
            import ahocorasick
            A = ahocorasick.Automaton()
            for n in self.names_sorted:
                if len(n) >= min_len: A.add_word(n, n)
            A.make_automaton(); self.aut = A
        except Exception:
            self.aut = None                                   # falls back to the scan; same results, slower
        # (mention type, answer type) -> set of (rel, direction)   direction 0: mention --r--> answer
        self.ops = collections.defaultdict(set)
        for r, a, b, c in sigs:                       # signatures carry type ids; parse() uses names
            self.ops[(type_names[a], type_names[b])].add((r, 0))
            self.ops[(type_names[b], type_names[a])].add((r, 1))
        self.ktype = {t: i for i, t in type_names.items()}
        # two-hop paths (mention type -> X -> answer type) for pairs without a direct relation
        self.ops2 = collections.defaultdict(set)
        types = list(type_names.values())
        for a in types:
            for b in types:
                if self.ops.get((a, b)):
                    continue
                for x in types:
                    for op1 in self.ops.get((a, x), ()):
                        for op2 in self.ops.get((x, b), ()):
                            self.ops2[(a, b)].add((op1, op2))

    def answer_type(self, q):
        head = q[:120].lower()
        best = None
        for pat, t in TYPE_PATTERNS:
            m = re.search(pat, head)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), t)
        return best[1] if best else None

    def mentions(self, q):
        ql = " " + q.lower() + " "
        found, used = [], []
        for m in re.finditer(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,7})(?![A-Za-z0-9])", q):   # gene symbols like HGD, TP53, GCK
            n = m.group(1).lower()
            if n in self.by_name and all(self.node_type[i] == self.ktype["gene/protein"] for i in self.by_name[n][:1]):
                found.append((n, self.by_name[n])); used.append((m.start() + 1, m.end() + 1))
            elif n in self.by_symbol:
                found.append((n, self.by_symbol[n])); used.append((m.start() + 1, m.end() + 1))
            elif n in self.alias_symbols:
                found.append((n, self.alias_symbols[n])); used.append((m.start() + 1, m.end() + 1))
        pool = self.names_sorted
        if self.aut is not None:
            pool = sorted({v for _, v in self.aut.iter(ql)}, key=self.name_rank.__getitem__)
        for n in pool:
            if len(n) < 4 or n not in ql:
                continue
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", ql):
                s, e = m.span()
                if any(not (e <= us or s >= ue) for us, ue in used):
                    continue
                used.append((s, e)); found.append((n, self.by_name[n]))
            if len(found) >= 6:
                break
        return found

    def chain_weights(self, mt, at, q, rel_hints=(), scale=1.0):
        """operator chains (mention type -> answer type) with keyword / hint weights"""
        ops = self.ops.get((mt, at), set()) if at else set()
        w = {}
        for r, d in ops:
            hit = bool(re.search(K.get(r, r"$^"), q.lower())) or r in rel_hints
            w[((r, d),)] = scale * (2.0 if hit else 1.0)
        if not ops and at:
            for (op1, op2) in self.ops2.get((mt, at), ()):
                hits = sum(bool(re.search(K.get(op[0], r"$^"), q.lower())) or op[0] in rel_hints for op in (op1, op2))
                w[(op1, op2)] = scale * (1.0 + 0.5 * hits)
        return w

    def parse(self, q, at=None, rel_hints=(), extra_names=()):
        at = at or self.answer_type(q)
        ments = self.mentions(q)
        seen = {n for n, _ in ments}
        for n in extra_names:                          # names proposed by the LLM parser, exact lookup
            n = n.strip().lower()
            if n in self.by_name and n not in seen:
                ments.append((n, self.by_name[n])); seen.add(n)
        out = []
        for n, ids in ments:
            for i in ids[:3]:
                mt = self.tn[int(self.node_type[i])]
                w = self.chain_weights(mt, at, q, rel_hints)
                if not w:
                    continue
                out.append((i, n, mt, w))
        return at, out

    @staticmethod
    def negation(q):
        """'diseases with no drugs indicated / lacking treatment' -> exclude candidates that have an
        indication edge; returns the relation id to require zero degree in, or None."""
        ql = q.lower()
        if re.search(r"(no|without|lack\w*|not have|lacking) (any )?(associated |approved |available |known )?(drugs?|medications?|treatments?|pharmacolog\w*)", ql):
            return 6
        return None


class Adjacency:
    """Exact neighbour sets per directed operator (r, d): op r,0 = h -> t rows of relation r; r,1 = t -> h."""
    def __init__(self, kg, n_rel, N):
        h, r, t = kg["h"], kg["r"], kg["t"]
        self.A = {}
        for rr in range(n_rel):
            m = r == rr
            self.A[(rr, 0)] = sps.csr_matrix((np.ones(m.sum(), np.float32), (h[m], t[m])), shape=(N, N))
            self.A[(rr, 1)] = self.A[(rr, 0)].T.tocsr()
    def reach(self, i, chain, cap=200000):
        """node ids reached from i along the chain of (r, d) operators (exact graph traversal)."""
        front = np.array([i])
        for (r, d) in chain:
            M = self.A[(r, d)]
            nxt = M[front].sum(0)
            front = np.asarray(nxt).ravel().nonzero()[0] if front.size else front
            if front.size > cap:
                break
        return front


@torch.no_grad()
def rev_feats(model, n_rel, E, tk, rev_track, dev):
    """P6: the OPPOSITE operator's score of the same link, per shortlist candidate.

    The table has separate forward and reverse operators per relation, so for an anchor a and a
    candidate c the reverse of the hop the walk used to reach c is a second, independent scorer of
    the same link. r_rev = op ^ n_rel of the LAST hop of the chain that won c from a;
      rev_raw(c) = Re<out(hop(embed(c), r_rev), r_rev), row(a)> * tau
      rev_nov(c) = rev_raw(c) - logsumexp of the same score over the OTHER shortlist candidates as
                   alternative targets (how much c prefers a over other links).
    Max and mean over the anchors; rev_anchor / rev_op record which anchor and operator gave the
    max, for rev_check.py. (wikikg2/REVERSE_MEMBER.md)"""
    K = len(tk)
    if not rev_track:
        return {k: [0.0] * K for k in ("rev_raw_max", "rev_raw_mean", "rev_nov_max", "rev_nov_mean")} | {"rev_anchor": [-1] * K, "rev_op": [-1] * K}
    tau = model.log_tau.exp(); ar = torch.arange(K, device=dev)
    raws, novs, ops = [], [], []
    for (i, best_op) in rev_track:
        op_rev = (best_op[tk] + n_rel) % (2 * n_rel)
        zc = model.out(model.hop(model.embed(tk), op_rev), op_rev)                                  # (K, M)
        S = torch.real(zc @ E[torch.cat([torch.tensor([i], device=dev), tk])].conj().t()) * tau     # (K, 1 + K): anchor, then the shortlist
        S[ar, ar + 1] = -1e9                                                                        # a candidate is not its own alternative target
        raws.append(S[:, 0]); novs.append(S[:, 0] - torch.logsumexp(S, 1)); ops.append(op_rev)
    R = torch.stack(raws); V = torch.stack(novs); O = torch.stack(ops); b = R.argmax(0)
    return {"rev_raw_max": R.max(0).values.tolist(), "rev_raw_mean": R.mean(0).tolist(),
            "rev_nov_max": V.max(0).values.tolist(), "rev_nov_mean": V.mean(0).tolist(),
            "rev_anchor": [int(rev_track[j][0]) for j in b.tolist()], "rev_op": O[b, ar].tolist()}


@torch.no_grad()
def score_query(model, n_rel, parsed, at, type_mask, dev, k=100, exclude=None, adj=None, beta=0.0, feats=False, no_model=False, logdeg=None, agg="sum", agg_p=1.0, rev=False):
    at_idx, ments = parsed
    if at is None or not ments:
        return ([], None) if feats else []
    E = model.table()                                    # (N, M) complex
    mask = type_mask[at].clone()
    if exclude is not None:
        mask &= ~exclude
    total = torch.zeros(E.shape[0], device=dev)
    exact = torch.zeros(E.shape[0], device=dev)
    per_name = {}                                         # best chain score per NAME: OR over a name's resolutions and chains, AND across names
    rev_track = []                                        # P6: (anchor id, per-candidate last hop of the winning chain)
    for (i, n, mt, w) in ments:
        best = None; best_op = None
        if adj is not None and beta > 0:
            hit = np.zeros(E.shape[0], dtype=bool)
            for chain, wt in w.items():
                hit[adj.reach(i, chain)] = True
            exact += torch.from_numpy(hit).to(dev).float()
        if no_model:
            continue
        for chain, wt in w.items():
            z = model.embed(torch.tensor([i], device=dev))
            for (r, d) in chain:                          # compiled chain: apply the hops in order
                op = torch.tensor([r + (0 if d == 0 else n_rel)], device=dev)
                z = model.hop(z, op)
            s = torch.real(model.out(z, op) @ E.conj().t())[0] * model.log_tau.exp()
            sm = s[mask]
            zs = (s - sm.mean()) / (sm.std() + 1e-6)
            zs = zs * wt
            if rev:                                       # which hop reached each candidate from this anchor
                op_last = int(op[0])
                best_op = torch.full_like(zs, op_last, dtype=torch.long) if best is None else torch.where(zs > best, op_last, best_op)
            best = zs if best is None else torch.maximum(best, zs)
        total += best
        if best is not None: per_name[n] = best if n not in per_name else torch.maximum(per_name[n], best)
        if rev and best is not None: rev_track.append((i, best_op))
    if agg != "sum" and len(per_name) >= 2:              # AND over names of (OR over resolutions and chains): one stacked tensor, one reduction
        S = torch.stack(list(per_name.values()))          # (k_names, N)
        if agg == "min": total = S.min(0).values
        elif agg == "softmin": total = -agg_p * torch.logsumexp(-S / agg_p, 0)
        elif agg == "logsig": total = F.logsigmoid(S - agg_p).sum(0)
        elif agg == "count":                              # soft count of satisfied caps (what the walk's exact-support count does, in the space) + the sum for ordering
            total = 30.0 * torch.sigmoid((S - agg_p) / 0.5).sum(0) + S.sum(0)
    total = total + beta * exact
    if no_model and logdeg is not None:
        total = total + 1e-3 * logdeg
    total[~mask] = -1e9
    tk = torch.topk(total, k).indices
    top = tk.tolist()
    if feats:   # per-candidate parts of the score, for a reranker fit on train: z-score sum and exact-support count
        f = {"z": (total[tk] - beta * exact[tk]).tolist(), "exact": exact[tk].tolist()}
        if rev: f.update(rev_feats(model, n_rel, E, tk, rev_track, dev))
        return top, f
    return top


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--no-2hop", action="store_true", help="ablation: one-hop operators only (parser v1 coverage)")
    p.add_argument("--beta", type=float, default=0.0, help="weight of the exact-neighbour count (graph-supported constraints)")
    p.add_argument("--anchor", default=None, help="text-resolve anchors for queries without an exact mention: bge | bgeft | bgeft2 (bgeft2 reuses the text ranker's encoder and data/doc_emb_bgeft2.npy) | lp (P8 lever A': the latent parser's own anchor head, no separate encoder and no doc matrix)")
    p.add_argument("--anchor-top", type=int, default=2)
    p.add_argument("--lp-anchor", default="models/lp_p3.pt", help="checkpoint whose anchor head serves --anchor lp")
    p.add_argument("--anchor-weight", type=float, default=0.7)
    p.add_argument("--queries", default=None, help="json {query_id: text} replacing the question text (paraphrase proxy; train/val only)")
    p.add_argument("--llm-parse", default=None, help="data/llmparse_<tag>.json from llm_parse.py: answer type fallback, relation hints, entity names, exclusion")
    p.add_argument("--lparse", default=None, help="latent parser output (latent_parser.py): answer type, anchor ids, operator ids per query; replaces the regex/LLM parse (exact-name mentions kept as fallback)")
    p.add_argument("--agg", default="sum", choices=["sum", "min", "softmin", "logsig", "count"], help="how mention scores combine: sum (OR-ish) or an AND readout")
    p.add_argument("--agg-p", type=float, default=1.0, help="tau for softmin, c for logsig")
    p.add_argument("--confirm", default=None, help="P5: latent-confirmed anchors — path of the latent parser checkpoint (models/lp.pt); string-matched candidates must be pointed at by the question vector")
    p.add_argument("--confirm-rel", type=float, default=0.8); p.add_argument("--confirm-abs", type=float, default=0.75)
    p.add_argument("--aliases", default=None, help="data/aliases.json from build_aliases.py (P4 lever A)")
    p.add_argument("--legacy-names", action="store_true", help="P1/P2 behaviour: no 2-3 character gene symbols (reproduces the committed reads)")
    p.add_argument("--no-model", action="store_true", help="ablation: drop the ResonatE score entirely; rank by exact-support count only (ties by node degree)")
    p.add_argument("--llm-override-type", action="store_true", help="let the LLM answer type override the pattern one (default: fallback only)")
    p.add_argument("--dump-feats", action="store_true", help="also store per-candidate z-sum and exact count (reranker features)")
    p.add_argument("--rev", action="store_true", help="P6: also store the reverse-operator features rev_raw / rev_nov per candidate (needs --dump-feats)")
    a = p.parse_args()
    predict_only = a.split in ("test", "test-0.1", "human_generated_eval")   # committed read: rankings only, no metrics here
    dev = torch.device(a.device)
    model, n_rel = load_model(a.model, dev)
    names = json.load(open("data/names.json")); d = np.load("data/kg.npz"); node_type = d["node_type"]
    dicts = json.load(open("data/dicts.json")); tn = {int(k): v for k, v in dicts["node_type_dict"].items()}
    sigs = json.load(open("data/signatures.json"))
    parser = Parser(names, node_type, tn, sigs, short_symbols=not a.legacy_names, aliases=json.load(open(a.aliases)) if a.aliases else None)
    if a.no_2hop:
        parser.ops2 = {}
    type_mask = {t: torch.from_numpy(node_type == i).to(dev) for i, t in tn.items()}
    deg = {rr: np.bincount(d["h"][d["r"] == rr], minlength=len(node_type)) for rr in range(int(d["n_rel"]))}
    logdeg_all = torch.from_numpy(np.log1p(np.bincount(d["h"], minlength=len(node_type)) + np.bincount(d["t"], minlength=len(node_type))).astype(np.float32)).to(dev)
    adj = Adjacency(d, int(d["n_rel"]), len(node_type)) if a.beta > 0 else None
    anchor = None
    if a.anchor == "lp":                                   # P8 lever A': reuse the parser head already needed for --lparse
        from latent_parser import LP as LPNET, QPRE as LP_QPRE
        from transformers import AutoTokenizer
        cka = torch.load(a.lp_anchor, map_location="cpu", weights_only=False)
        atok = AutoTokenizer.from_pretrained(cka["encoder"])
        anet = LPNET(cka["encoder"], model.m, len(tn), 2 * n_rel, cka["kanc"]).to(dev); anet.load_state_dict(cka["state"]); anet.eval()
        anchor = ("lp", atok, anet, model.table().detach(), LP_QPRE)
    elif a.anchor:
        from sentence_transformers import SentenceTransformer
        emb = torch.from_numpy(np.load(f"data/doc_emb_{a.anchor}.npy").astype(np.float32)).to(dev)
        st = SentenceTransformer({"bge": "BAAI/bge-base-en-v1.5", "bgeft": "models/bge_ft", "bgeft2": "models/bge_ft2"}[a.anchor], device=str(dev))   # P7: bgeft2 is the text ranker's own encoder — one encoder and one doc matrix instead of two
        qpre = "Represent this sentence for searching relevant passages: "
        anchor = (emb, st, qpre)
    qa = load_qa("prime", human_generated_eval=(a.split == "human_generated_eval"))
    idx = qa.get_idx_split()[a.split].tolist() if a.split != "human_generated_eval" else list(range(len(qa)))
    if a.limit: idx = idx[:a.limit]
    QS = json.load(open(a.queries)) if a.queries else None
    assert QS is None or not predict_only
    CONF = None
    if a.confirm:                                          # P5: load the anchor head once; confirm string-matched candidates with the question's vectors
        from latent_parser import LP, QPRE as LP_QPRE
        from transformers import AutoTokenizer
        ckc = torch.load(a.confirm, map_location="cpu", weights_only=False)
        lp_tok = AutoTokenizer.from_pretrained(ckc["encoder"]); lpm = LP(ckc["encoder"], model.m, len(tn), 2 * n_rel, ckc["kanc"]).to(dev); lpm.load_state_dict(ckc["state"]); lpm.eval()
        E_all = model.table().detach(); CONF = (lp_tok, lpm, E_all, ckc["sim_floor"], LP_QPRE); n_conf_kept = n_conf_dropped = 0
    LP = json.load(open(a.llm_parse)) if a.llm_parse else None
    LPZ = json.load(open(a.lparse)) if a.lparse else None
    rel_ids = {v: int(k) for k, v in dicts["edge_type_dict"].items()}
    type_ok = set(tn.values()); n_llm_type = n_llm_ent = 0
    rows_all, rows_cov, out, n_cov, n_type, t0 = [], [], {}, 0, 0, time.time(); rows_multi, rows_single = [], []; n_anc_fb = 0
    for j, i in enumerate(idx):
        q, qid, ans, _ = qa[i]
        if QS is not None: q = QS.get(str(int(qid)), q)
        lp = (LP or {}).get(str(int(qid))) or {}
        l_at = lp.get("answer_type") if lp.get("answer_type") in type_ok else None
        l_rels = {rel_ids[r] for r in (lp.get("relations") or []) if isinstance(r, str) and r in rel_ids}
        l_ents = [e for e in (lp.get("entities") or []) if isinstance(e, dict) and isinstance(e.get("name"), str)]
        at0 = parser.answer_type(q)
        at_use = (l_at or at0) if a.llm_override_type else (at0 or l_at)
        n_llm_type += (at0 is None and l_at is not None)
        if LPZ is not None:                                   # latent parser: learned answer type, anchors and operator hints
            lz = LPZ.get(str(int(qid))) or {}
            at_use = lz.get("answer_type") if lz.get("answer_type") in type_ok else at0
            l_rels = {int(o) % n_rel for o in lz.get("ops", [])}
        at, ments = parser.parse(q, at=at_use, rel_hints=l_rels, extra_names=[e["name"] for e in l_ents])
        if CONF is not None and ments:
            lp_tok, lpm, E_all, floor_abs, lp_qpre = CONF
            with torch.no_grad():
                enc = lp_tok([lp_qpre + q], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                _, _, zq = lpm(enc); ids = torch.tensor([i_ for i_, _, _, _ in ments], device=dev)
                sc = (torch.real(zq[0] @ E_all[ids].conj().t()) * lpm.scale.exp()).max(0).values.tolist()   # best of the K vectors per candidate
            by_name = {}
            for (i_, n_, mt, w), s_ in zip(ments, sc): by_name.setdefault(n_, []).append((i_, mt, w, s_))
            kept = []
            for n_, lst in by_name.items():
                best = max(s_ for *_, s_ in lst); thr = max(a.confirm_rel * best if best > 0 else -1e9, a.confirm_abs * floor_abs)
                for (i_, mt, w, s_) in lst:
                    if s_ >= thr: kept.append((i_, n_, mt, w)); n_conf_kept += 1
                    else: n_conf_dropped += 1
            ments = kept
        if LPZ is not None and at is not None:
            have = {i_ for i_, _, _, _ in ments}
            for i_ in lz.get("anchors", []):
                if i_ in have: continue
                mt = tn[int(node_type[i_])]; w = parser.chain_weights(mt, at, q, l_rels)
                if w: ments.append((i_, names[str(i_)].lower(), mt, w)); n_llm_ent += 1
        n_type += at is not None
        if anchor is not None and anchor[0] != "lp" and at is not None and not ments and l_ents:      # resolve the LLM's entity names by text, per name
            emb, st, qpre = anchor
            found = {n_ for _, n_, _, _ in ments}
            for e in l_ents:
                et = e.get("type") if e.get("type") in type_ok else None
                qv = torch.from_numpy(st.encode([qpre + e["name"]], convert_to_numpy=True, normalize_embeddings=True)).to(dev)[0]
                sims = emb @ qv
                ok = torch.zeros_like(sims, dtype=torch.bool)
                for mt_name in parser.ktype:
                    if (et is None or mt_name == et) and (parser.ops.get((mt_name, at)) or parser.ops2.get((mt_name, at))):
                        ok |= type_mask[mt_name]
                sims[~ok] = -1e9
                i_ = int(torch.argmax(sims)); mt = tn[int(node_type[i_])]
                w = parser.chain_weights(mt, at, q, l_rels, scale=a.anchor_weight)
                if w and float(sims[i_]) > -1e8:
                    ments.append((i_, names[str(i_)].lower(), mt, w)); n_llm_ent += 1
        if anchor is not None and at is not None and not ments:
            n_anc_fb += 1
            if anchor[0] == "lp":                          # P8 lever A': the parser's anchor vectors over the entity table, no floor (this IS the fallback)
                _, atok, anet, E_anc, aqpre = anchor
                with torch.no_grad():
                    aenc = atok([aqpre + q], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                    _, _, zq = anet(aenc)
                    sims = (torch.real(zq[0] @ E_anc.conj().t()) * anet.scale.exp()).max(0).values
            else:
                emb, st, qpre = anchor
                qv = torch.from_numpy(st.encode([qpre + q], convert_to_numpy=True, normalize_embeddings=True)).to(dev)[0]
                sims = emb @ qv
            ok = torch.zeros_like(sims, dtype=torch.bool)
            for mt_name, ti in parser.ktype.items():
                if parser.ops.get((mt_name, at)) or parser.ops2.get((mt_name, at)):
                    ok |= type_mask[mt_name]
            sims[~ok] = -1e9
            for i_ in torch.topk(sims, a.anchor_top).indices.tolist():
                mt = tn[int(node_type[i_])]
                w = {}
                for r, dd in parser.ops.get((mt, at), ()):
                    w[((r, dd),)] = a.anchor_weight * (2.0 if re.search(K.get(r, r"$^"), q.lower()) else 1.0)
                if not w:
                    for (op1, op2) in parser.ops2.get((mt, at), ()):
                        hits = sum(bool(re.search(K.get(op[0], r"$^"), q.lower())) for op in (op1, op2))
                        w[(op1, op2)] = a.anchor_weight * (1.0 + 0.5 * hits)
                if w:
                    ments.append((i_, names[str(i_)].lower(), mt, w))
        neg = parser.negation(q)
        if neg is None and lp.get("exclude_relation") in rel_ids:
            neg = rel_ids[lp["exclude_relation"]]
        excl = torch.from_numpy(deg[neg] > 0).to(dev) if neg is not None else None
        top = score_query(model, n_rel, (at, ments), at, type_mask, dev, exclude=excl, adj=adj, beta=a.beta, feats=a.dump_feats, no_model=a.no_model, logdeg=logdeg_all, agg=a.agg, agg_p=a.agg_p, rev=a.rev)
        fe = None
        if a.dump_feats:
            top, fe = top
        if top:
            n_cov += 1
        if not predict_only:
            m = stark_metrics(top, ans)
            rows_all.append(m)
            if top:
                rows_cov.append(m)
            (rows_multi if len({n_ for _, n_, _, _ in ments}) >= 2 else rows_single).append(m)
        out[int(qid)] = {"top": top[:100], "answer_type": at, "mentions": [(i_, n_, mt) for (i_, n_, mt, w) in ments], "negation": neg}
        if fe is not None:
            out[int(qid)]["feats"] = fe
        if j % 500 == 0:
            print(j, round(time.time() - t0), "s", flush=True)
    print(f"{a.split}: n={len(idx)}  answer type found {n_type}  covered (>=1 usable mention) {n_cov} ({n_cov/len(idx):.1%})" + (f"  LLM: type fallback used {n_llm_type}, entity anchors {n_llm_ent}" if LP else "") + (f"  confirm: kept {n_conf_kept}, dropped {n_conf_dropped}" if CONF is not None else "") + (f"  text-anchor fallback used on {n_anc_fb}" if anchor is not None else ""))
    if rows_all:
        print("relational-only, all queries (uncovered count as misses):", {k: round(v, 4) for k, v in summarize(rows_all).items()})
    if rows_cov:
        print("relational-only, covered queries:", {k: round(v, 4) for k, v in summarize(rows_cov).items()})
    if rows_multi:
        print(f"  >= 2 names (n={len(rows_multi)}):", {k: round(v, 4) for k, v in summarize(rows_multi).items()}, f"| 1 name (n={len(rows_single)}):", {k: round(v, 4) for k, v in summarize(rows_single).items()})
    json.dump(out, open(a.out or f"data/rel_{a.split}.json", "w"))


if __name__ == "__main__":
    main()
