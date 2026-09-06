"""P2 lever 2c(b): a local instruction model as the question parser. Given the PrimeKG schema and a
few train examples, it maps a question to {answer_type, entities [{name, type}], relations,
exclude_relation}. The graph answers; the model only reads. Writes data/llmparse_{tag}.json keyed by
query id. Runs on train/val questions or a paraphrase file; never on test or the human set.
Usage: uv run python llm_parse.py --split val [--queries data/para_val.json --tag val_para]"""
import argparse, json, re, time, torch
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModelForCausalLM

TYPES = ["disease", "gene/protein", "molecular_function", "drug", "pathway", "anatomy", "effect/phenotype",
         "biological_process", "cellular_component", "exposure"]
RELS = {"ppi": "gene/protein interacts with gene/protein (protein-protein interaction, binding)",
        "carrier": "gene/protein is a carrier of drug", "enzyme": "gene/protein is an enzyme metabolising drug",
        "target": "gene/protein is a target of drug (drug targets / inhibits / activates gene)",
        "transporter": "gene/protein transports drug", "contraindication": "drug is contraindicated for disease",
        "indication": "drug is indicated for / treats disease", "off-label use": "drug is used off-label for disease",
        "synergistic interaction": "drug interacts synergistically with drug (used together)",
        "associated with": "gene/protein is associated with disease or with effect/phenotype",
        "parent-child": "a broader category and its subtype (disease, phenotype, pathway, process, function, component, anatomy, exposure)",
        "phenotype absent": "disease does NOT present phenotype", "phenotype present": "disease presents phenotype / symptom",
        "side effect": "drug has side effect (phenotype)",
        "interacts with": "gene/protein participates in pathway / biological process / molecular function / cellular component; exposure interacts with gene, process, disease",
        "linked to": "exposure is linked to disease", "expression present": "gene/protein is expressed in anatomy (tissue, organ)",
        "expression absent": "gene/protein is NOT expressed in anatomy"}
SYS = ("You convert biomedical questions about a knowledge graph into a structured query. The graph has node types: "
       + ", ".join(TYPES) + ". Relations: " + "; ".join(f"'{k}': {v}" for k, v in RELS.items()) + ". "
       "Return ONLY a JSON object with keys: \"answer_type\" (one node type: the kind of thing the question asks for), "
       "\"entities\" (list of {\"name\", \"type\"}: every specific named entity the question mentions, with the name exactly as "
       "written in the question, expanding nothing), \"relations\" (list of relation names the question uses, from the list), "
       "\"exclude_relation\" (a relation name if the question asks for things that LACK such a relation, else null).")
FEW = [("Which drugs specifically target the glucokinase (GCK) gene?",
        {"answer_type": "drug", "entities": [{"name": "glucokinase (GCK)", "type": "gene/protein"}], "relations": ["target"], "exclude_relation": None}),
       ("What gene or protein linked to Noonan syndrome is expressed in the dorsolateral prefrontal cortex?",
        {"answer_type": "gene/protein", "entities": [{"name": "Noonan syndrome", "type": "disease"}, {"name": "dorsolateral prefrontal cortex", "type": "anatomy"}],
         "relations": ["associated with", "expression present"], "exclude_relation": None}),
       ("Which conditions lack any approved medication and present with hearing loss?",
        {"answer_type": "disease", "entities": [{"name": "hearing loss", "type": "effect/phenotype"}], "relations": ["phenotype present"], "exclude_relation": "indication"})]

p = argparse.ArgumentParser(); p.add_argument("--split", default="val"); p.add_argument("--queries", default=None); p.add_argument("--tag", default=None)
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); p.add_argument("--batch", type=int, default=32); p.add_argument("--limit", type=int, default=0)
a = p.parse_args(); assert a.split in ("train", "val")
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
qa = load_qa("prime"); idx = qa.get_idx_split()[a.split].tolist()
if a.limit: idx = idx[:a.limit]
QS = json.load(open(a.queries)) if a.queries else {}
msgs0 = [{"role": "system", "content": SYS}]
for q_, j_ in FEW: msgs0 += [{"role": "user", "content": q_}, {"role": "assistant", "content": json.dumps(j_)}]
out, bad, t0 = {}, 0, time.time()
for b in range(0, len(idx), a.batch):
    chunk = idx[b:b + a.batch]
    qs = [QS.get(str(int(qa[i][1])), qa[i][0]) for i in chunk]
    prompts = [tok.apply_chat_template(msgs0 + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = m.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
    for i, g in zip(chunk, gen):
        txt = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        mm = re.search(r"\{.*\}", txt, re.S)
        try:
            j = json.loads(mm.group(0)) if mm else None
        except Exception:
            j = None
        if not isinstance(j, dict): j = None; bad += 1
        out[int(qa[i][1])] = j
    if b % (a.batch * 10) == 0:
        print(b, round(time.time() - t0), "s |", qs[0][:80], "->", json.dumps(out[int(qa[chunk[0]][1])])[:160], flush=True)
tag = a.tag or a.split
json.dump(out, open(f"data/llmparse_{tag}.json", "w"))
print("LLMPARSE_DONE", len(out), "unparsable", bad, round(time.time() - t0), "s")
