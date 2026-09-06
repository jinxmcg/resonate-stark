"""P2 lever 2c(a): a proxy "human" set. A local instruction model rewrites `val` (or `train`)
questions the way a person would type them: different wording, synonyms, abbreviations, looser
structure, same meaning and same constraints. The model sees the question only (no graph, no
answers). Writes data/para_{split}.json {query_id: paraphrase}. Never run on test or human.
Usage: uv run python paraphrase.py --split val [--model Qwen/Qwen2.5-7B-Instruct]"""
import argparse, json, time, re, torch
import stark_shim  # noqa
from stark_qa import load_qa
from transformers import AutoTokenizer, AutoModelForCausalLM

SYS = ("You rewrite biomedical search questions the way a real researcher or clinician would type them "
       "into a search box. Change the wording substantially: use synonyms, common abbreviations, a different "
       "sentence structure, sometimes a terse keyword style, sometimes a longer conversational style. Keep "
       "exactly the same meaning: every constraint in the original must remain, none may be added, and the "
       "kind of thing being asked for must not change. Do not answer the question. Output only the rewritten "
       "question, nothing else.")

p = argparse.ArgumentParser(); p.add_argument("--split", default="val"); p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--batch", type=int, default=48); p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default=None)
p.add_argument("--style", default="natural", choices=["natural", "terse"], help="terse = clinician's search-box shorthand: abbreviations, dropped function words, descriptions instead of some names")
a = p.parse_args()
assert a.split in ("train", "val")
if a.style == "terse":
    SYS = ("You rewrite biomedical search questions the way a busy clinician or researcher types them into a search box: "
           "short, keyword-like, with common abbreviations (e.g. T2D, HTN, CKD, BRCA1), function words dropped, and sometimes a short "
           "description in place of a name (e.g. 'low potassium' instead of 'hypokalemia'). Keep exactly the same meaning: every "
           "constraint stays, none is added, and the kind of thing asked for does not change. Write in English only. Output only the "
           "rewritten query, nothing else.")
torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
qa = load_qa("prime"); idx = qa.get_idx_split()[a.split].tolist()
if a.limit: idx = idx[:a.limit]
out, t0 = {}, time.time()
for b in range(0, len(idx), a.batch):
    chunk = idx[b:b + a.batch]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": qa[i][0]}],
                                       tokenize=False, add_generation_prompt=True) for i in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = m.generate(**enc, max_new_tokens=96, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
    for i, g in zip(chunk, gen):
        txt = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().strip('"')
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", txt): txt = ""      # drop rewrites that drifted into another script (lever C)
        out[int(qa[i][1])] = txt or qa[i][0]
    if b % (a.batch * 10) == 0:
        print(b, round(time.time() - t0), "s |", qa[chunk[0]][0][:90], "->", out[int(qa[chunk[0]][1])][:90], flush=True)
json.dump(out, open(a.out or f"data/para_{a.split}.json", "w"), indent=0)
print("PARA_DONE", len(out), round(time.time() - t0), "s")
