"""
FACET — Robustness Perturbation Evaluation
===========================================
Adds a third evaluation condition alongside zero_shot and cot:
  - surface:    rename entities, swap numbers, rephrase (semantics preserved)
  - distractor: inject an irrelevant sentence into the question stem
  - reframe:    wrap the question in a scenario/story context

Generates perturbed variants of every task in facet_v2_dataset.json,
evaluates them under zero-shot prompting, and saves results to:
  facet_perturb_results_{model}_seed{seed}.json

The robustness score per task is:
  robust = 1 if (original correct AND perturbed correct), else 0
  fragile = 1 if (original correct AND perturbed wrong)  ← key metric

Usage:
    python facet_perturbation.py --model groq --key gsk_xxx
    python facet_perturbation.py --model openai --key sk-proj-xxx
    python facet_perturbation.py --model deepseek --key xxx
    python facet_perturbation.py --model gemini --key xxx
    python facet_perturbation.py --model qwen --key xxx
    python facet_perturbation.py --results --file facet_perturb_results_groq_seed42.json

Optional flags:
    --perturbation surface|distractor|reframe|all  (default: all)
    --seed 42
    --spotcheck   Preview 10 perturbed tasks without running eval
"""

import json
import random
import argparse
import os
import re
import time
from collections import defaultdict
from datetime import datetime

DATASET_FILE = "facet_v2_dataset.json"

CATEGORIES = [
    "reversal_curse", "compositional_reasoning", "syllogistic_reasoning",
    "working_memory", "inhibitory_control", "counting",
    "anchoring_bias", "theory_of_mind",
]

# ─────────────────────────────────────────────────────────────
# PERTURBATION ENGINE
# ─────────────────────────────────────────────────────────────

# Name pools for surface perturbations
MALE_NAMES   = ["James", "Oliver", "Ethan", "Lucas", "Noah", "Liam", "Henry", "Felix", "Hugo", "Leo"]
FEMALE_NAMES = ["Emma", "Sofia", "Aria", "Clara", "Nora", "Zoe", "Maya", "Iris", "Luna", "Eva"]
ALL_NAMES    = MALE_NAMES + FEMALE_NAMES

# Common name patterns to detect and replace
NAME_PATTERN = re.compile(
    r'\b(Alice|Bob|Carol|David|Eve|Frank|Grace|Henry|Iris|Jack|Kate|'
    r'Liam|Mary|Noah|Olivia|Paul|Quinn|Rose|Sam|Tom|Uma|Victor|Wendy|'
    r'Xander|Yara|Zoe|John|Jane|Mike|Sarah|Emma|James|Lisa|Anna|Chris|'
    r'Dan|Pat|Alex|Jordan|Morgan|Taylor|Casey|Jamie|Robin|Blair)\b'
)

# Distractor sentences — generic, topic-neutral, semantically irrelevant
DISTRACTORS = [
    "Note that the weather outside is partly cloudy today.",
    "For context, this question was originally written in English.",
    "Keep in mind that the building has four floors.",
    "Incidentally, the event took place on a Tuesday.",
    "As a reminder, all measurements here use the metric system.",
    "Note that there are seven days in a week.",
    "For reference, the room contains exactly twelve chairs.",
    "As background, the participants had all eaten breakfast that morning.",
    "By the way, the document was printed on standard A4 paper.",
    "Note that the conversation happened in the afternoon.",
]

# Reframe templates — wrap question in a neutral scenario
REFRAME_TEMPLATES = [
    "You are helping a student prepare for an exam. They ask you the following question:\n\n{question}",
    "A researcher studying cognitive tasks presents you with this problem:\n\n{question}",
    "During a study session, the following question comes up:\n\n{question}",
    "Imagine you are tutoring someone who asks:\n\n{question}",
    "A colleague poses this reasoning challenge to you:\n\n{question}",
    "In a logic puzzle book, you encounter the following:\n\n{question}",
    "While reviewing practice problems, you read:\n\n{question}",
    "You are playing a quiz game and receive this question:\n\n{question}",
]

# Number pattern — match integers 1-99 in text
NUMBER_PATTERN = re.compile(r'\b([2-9]|[1-9][0-9])\b')

# Number words to swap
NUMBER_WORDS = {
    "two": 3, "three": 4, "four": 5, "five": 6, "six": 7,
    "seven": 8, "eight": 9, "nine": 10, "ten": 12,
    "twice": "three times", "double": "triple",
    "first": "second", "second": "third", "third": "fourth",
}


def perturb_surface(task, rng):
    """
    Rename entities and shift numbers by a small offset.
    Semantics are preserved — the correct answer letter stays the same.
    """
    question = task["question"]
    options  = dict(task["options"])

    # ── Rename detected people ──────────────────────────────
    found_names = list(dict.fromkeys(NAME_PATTERN.findall(question)))  # preserve order, dedup
    if found_names:
        # Build a name replacement map — pick different names of any gender
        available = [n for n in ALL_NAMES if n not in found_names]
        rng.shuffle(available)
        name_map = {}
        for i, name in enumerate(found_names):
            name_map[name] = available[i % len(available)]

        def replace_name(m):
            return name_map.get(m.group(0), m.group(0))

        question = NAME_PATTERN.sub(replace_name, question)
        for k in options:
            options[k] = NAME_PATTERN.sub(replace_name, options[k])

    # ── Shift small integers by +1 or +2 ───────────────────
    # Only shift numbers that appear in isolation (not years, IDs)
    def shift_number(m):
        n = int(m.group(1))
        delta = rng.choice([1, 2])
        return str(n + delta)

    # Only apply if question has standalone small numbers
    if NUMBER_PATTERN.search(question):
        question = NUMBER_PATTERN.sub(shift_number, question)
        for k in options:
            # Only shift numbers in options too — preserves relative ordering
            options[k] = NUMBER_PATTERN.sub(shift_number, options[k])

    return question, options


def perturb_distractor(task, rng):
    """
    Inject an irrelevant sentence at a random position in the question stem.
    Correct answer is unchanged.
    """
    question   = task["question"]
    distractor = rng.choice(DISTRACTORS)

    # Split into sentences and inject
    sentences = re.split(r'(?<=[.!?])\s+', question.strip())
    if len(sentences) <= 1:
        # Just prepend
        new_question = distractor + " " + question
    else:
        insert_pos = rng.randint(0, len(sentences) - 1)
        sentences.insert(insert_pos, distractor)
        new_question = " ".join(sentences)

    return new_question, dict(task["options"])


def perturb_reframe(task, rng):
    """
    Wrap the question in a story/scenario framing.
    Correct answer is unchanged.
    """
    template    = rng.choice(REFRAME_TEMPLATES)
    new_question = template.format(question=task["question"])
    return new_question, dict(task["options"])


PERTURBATION_FNS = {
    "surface":    perturb_surface,
    "distractor": perturb_distractor,
    "reframe":    perturb_reframe,
}


def apply_perturbations(task, perturbation_types, rng):
    """
    Returns a dict of {perturbation_type: (perturbed_question, perturbed_options)}.
    """
    results = {}
    for ptype in perturbation_types:
        fn = PERTURBATION_FNS[ptype]
        results[ptype] = fn(task, rng)
    return results


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER (zero-shot only for perturbation eval)
# ─────────────────────────────────────────────────────────────

def build_prompt(question, options):
    options_text = "\n".join(f"({k}) {v}" for k, v in options.items())
    return (
        f"{question}\n\n"
        f"{options_text}\n\n"
        f"Answer with a single letter only: A, B, C, or D."
    )


# ─────────────────────────────────────────────────────────────
# ANSWER EXTRACTION (copied from facet_eval.py for consistency)
# ─────────────────────────────────────────────────────────────

def extract_answer(response_text):
    if not response_text:
        return None
    text = response_text.strip().upper()
    patterns = [
        r"(?:MY\s+)?(?:FINAL\s+)?ANSWER(?:\s+IS)?[\s:]+([ABCD])\b",
        r"(?:THEREFORE|SO|THUS|HENCE)[,\s]+(?:THE\s+ANSWER\s+IS\s+)?([ABCD])\b",
        r"CORRECT\s+ANSWER[\s:]+([ABCD])\b",
        r"(?:I\s+)?(?:CHOOSE|SELECT|PICK|GO\s+WITH)\s+([ABCD])\b",
        r"OPTION\s*\(?([ABCD])\)?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    matches = re.findall(r"\(([ABCD])\)", text)
    if matches:
        return matches[-1]
    tail = text[-100:]
    m = re.search(r"\b([ABCD])\b\.?\s*$", tail)
    if m:
        return m.group(1)
    m = re.search(r"\\BOXED\{([ABCD])\}", text)
    if m:
        return m.group(1)
    m = re.search(r"\*\*([ABCD])\*\*", text)
    if m:
        return m.group(1)
    if len(text.split()) <= 5:
        m = re.search(r"\b([ABCD])\b", text)
        if m:
            return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────
# MODEL CALLERS (mirrors facet_eval.py exactly)
# ─────────────────────────────────────────────────────────────

def call_openai(prompt, api_key, model="gpt-4o-mini"):
    try:
        import openai
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def call_groq(prompt, api_key, model="llama-3.1-8b-instant"):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 200, "temperature": 0}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

def call_deepseek(prompt, api_key, model="deepseek-chat"):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 200, "temperature": 0}
        r = requests.post("https://api.deepseek.com/chat/completions",
                         headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

def call_deepseek_r1(prompt, api_key, model="deepseek-reasoner"):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 1000, "temperature": 0}
        r = requests.post("https://api.deepseek.com/chat/completions",
                         headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

def _call_gemini_base(prompt, api_key, model, max_output_tokens=500):
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_output_tokens}
    }
    for attempt in range(4):
        try:
            r = requests.post(f"{url}?key={api_key}", headers=headers,
                             json=payload, timeout=60)
            if r.status_code in [503, 429]:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            text_parts = [p["text"] for p in parts if "text" in p]
            return text_parts[-1] if text_parts else "ERROR: no text"
        except Exception as e:
            if attempt < 3:
                time.sleep(30)
                continue
            return f"ERROR: {e}"
    return "ERROR: max retries"

def call_gemini(prompt, api_key):
    return _call_gemini_base(prompt, api_key, "gemini-2.5-flash")

def call_qwen(prompt, api_key, model="qwen/qwen3-32b"):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 500, "temperature": 0}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        return f"ERROR: {e}"

MODEL_CALLERS = {
    "openai":      call_openai,
    "groq":        call_groq,
    "deepseek":    call_deepseek,
    "deepseek_r1": call_deepseek_r1,
    "gemini":      call_gemini,
    "qwen":        call_qwen,
}


# ─────────────────────────────────────────────────────────────
# SPOTCHECK — preview perturbations without running eval
# ─────────────────────────────────────────────────────────────

def run_spotcheck(tasks, perturbation_types, seed=42, n=10):
    rng = random.Random(seed)
    sample = rng.sample(tasks, min(n, len(tasks)))
    print(f"\n{'='*70}")
    print(f"FACET Perturbation Spotcheck — {n} tasks")
    print(f"{'='*70}")
    for i, task in enumerate(sample, 1):
        print(f"\n{'─'*70}")
        print(f"Task {i:02d} | {task['category']} | {task['difficulty']}")
        print(f"ORIGINAL: {task['question'][:200]}")
        print(f"ANSWER:   ({task['answer']}) {task['options'][task['answer']]}")
        for ptype in perturbation_types:
            fn = PERTURBATION_FNS[ptype]
            pq, po = fn(task, random.Random(seed + i))
            print(f"\n[{ptype.upper()}]")
            print(f"  Q: {pq[:200]}")
            print(f"  A still: ({task['answer']}) {po[task['answer']]}")
    print(f"\n{'='*70}\n")


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────

def run_eval(tasks, model_name, api_key, perturbation_types, seed=42):
    rng = random.Random(seed)
    caller = MODEL_CALLERS[model_name]

    # Each task gets: original zero-shot + one call per perturbation type
    conditions = ["original"] + perturbation_types
    total_calls = len(tasks) * len(conditions)

    print(f"\n{'='*65}")
    print(f"FACET Perturbation Evaluation")
    print(f"Model       : {model_name}")
    print(f"Tasks       : {len(tasks)}")
    print(f"Perturbations: {', '.join(perturbation_types)}")
    print(f"Total calls : {total_calls}")
    print(f"{'='*65}\n")

    results = []
    call_num = 0
    errors = 0

    for task in tasks:
        task_result = {
            "id":             task["id"],
            "category":       task["category"],
            "difficulty":     task["difficulty"],
            "correct_answer": task["answer"],
            "conditions":     {},
        }

        # Generate all perturbed variants for this task upfront
        perturbed = {}
        for ptype in perturbation_types:
            pq, po = PERTURBATION_FNS[ptype](task, random.Random(seed + hash(task["id"]) % 10000))
            perturbed[ptype] = (pq, po)

        for condition in conditions:
            call_num += 1

            if condition == "original":
                question = task["question"]
                options  = task["options"]
            else:
                question, options = perturbed[condition]

            prompt = build_prompt(question, options)

            label = f"{task['category'][:16]}/{task['difficulty'][:4]}"
            print(f"[{call_num:04d}/{total_calls}] {label:<22} {condition:<12}", end="", flush=True)

            # Retry logic for rate limits
            response = None
            for attempt in range(3):
                response = caller(prompt, api_key)
                if response and "429" in str(response):
                    wait = 60 * (attempt + 1)
                    print(f" ⏳ rate limit, wait {wait}s...", end="", flush=True)
                    time.sleep(wait)
                    continue
                break

            if response and (str(response).startswith("ERROR:") or "429" in str(response)):
                extracted, correct = None, False
                errors += 1
                print(f" ⚠ {str(response)[:50]}")
            else:
                extracted = extract_answer(response)
                correct   = (extracted == task["answer"]) if extracted else False
                status    = "✓" if correct else ("?" if not extracted else "✗")
                print(f" {status} (got {extracted or '?'}, expected {task['answer']})")

            task_result["conditions"][condition] = {
                "question":          question if condition != "original" else None,
                "extracted_answer":  extracted,
                "correct":           correct,
            }
            time.sleep(0.5)

        results.append(task_result)

    # ── Compute robustness metrics ──────────────────────────
    output = {
        "model":             model_name,
        "run_date":          datetime.now().isoformat(),
        "seed":              seed,
        "perturbation_types": perturbation_types,
        "total_tasks":       len(tasks),
        "api_errors":        errors,
        "results":           results,
    }

    filename = f"facet_perturb_results_{model_name}_seed{seed}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {filename}")
    print_summary(results, model_name, perturbation_types)


# ─────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(results, model_name, perturbation_types):
    """
    For each perturbation type, report:
      - original accuracy (baseline)
      - perturbed accuracy
      - accuracy drop
      - fragility rate: % of originally-correct tasks that break under perturbation
      - robustness rate: % of originally-correct tasks that stay correct
    """
    print(f"\n{'='*75}")
    print(f"FACET Robustness Results — {model_name}")
    print(f"{'='*75}")

    for ptype in perturbation_types:
        orig_correct   = []
        perturb_correct = []
        fragile        = []  # was correct, now wrong
        robust         = []  # was correct, still correct

        cat_stats = defaultdict(lambda: {"orig": [], "perturb": [], "fragile": []})

        for r in results:
            o = r["conditions"].get("original", {})
            p = r["conditions"].get(ptype, {})
            if not o or not p:
                continue
            oc = o["correct"]
            pc = p["correct"]
            orig_correct.append(oc)
            perturb_correct.append(pc)
            cat = r["category"]
            cat_stats[cat]["orig"].append(oc)
            cat_stats[cat]["perturb"].append(pc)
            if oc:
                robust.append(pc)
                fragile.append(not pc)
                cat_stats[cat]["fragile"].append(not pc)

        def pct(lst):
            return f"{sum(lst)/len(lst)*100:.1f}%" if lst else "  -  "

        orig_acc    = sum(orig_correct)/len(orig_correct)*100    if orig_correct    else 0
        perturb_acc = sum(perturb_correct)/len(perturb_correct)*100 if perturb_correct else 0
        frag_rate   = sum(fragile)/len(fragile)*100              if fragile         else 0
        rob_rate    = sum(robust)/len(robust)*100                if robust          else 0
        drop        = orig_acc - perturb_acc

        sign = "+" if drop < 0 else "-"
        print(f"\n── Perturbation: {ptype.upper()} ──────────────────────────────────")
        print(f"  Original accuracy  : {orig_acc:.1f}%")
        print(f"  Perturbed accuracy : {perturb_acc:.1f}%  (Δ {sign}{abs(drop):.1f}pp)")
        print(f"  Fragility rate     : {frag_rate:.1f}%  (correct → wrong)")
        print(f"  Robustness rate    : {rob_rate:.1f}%  (correct → still correct)")

        print(f"\n  {'Category':<28} {'Orig':>6} {'Perturb':>8} {'Drop':>6} {'Fragile':>8}")
        print(f"  {'─'*60}")
        for cat in CATEGORIES:
            cs = cat_stats[cat]
            if not cs["orig"]:
                continue
            oa = sum(cs["orig"])/len(cs["orig"])*100
            pa = sum(cs["perturb"])/len(cs["perturb"])*100
            fr = sum(cs["fragile"])/len(cs["fragile"])*100 if cs["fragile"] else 0
            dp = oa - pa
            label = cat.replace("_", " ").title()[:27]
            sign2 = "+" if dp < 0 else "-"
            print(f"  {label:<28} {oa:5.1f}%  {pa:6.1f}%  {sign2}{abs(dp):4.1f}pp  {fr:6.1f}%")

    print(f"\n{'='*75}")
    print(f"\nKey metric: FRAGILITY RATE = % of originally-correct answers")
    print(f"that the model gets WRONG after a semantics-preserving perturbation.")
    print(f"High fragility → model is pattern-matching, not reasoning.")
    print(f"{'='*75}\n")


def load_and_print_results(filepath, perturbation_types=None):
    with open(filepath) as f:
        data = json.load(f)
    ptypes = perturbation_types or data.get("perturbation_types", ["surface", "distractor", "reframe"])
    print_summary(data["results"], data["model"], ptypes)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACET Perturbation Evaluation")
    parser.add_argument("--eval",     action="store_true", help="Run perturbation evaluation")
    parser.add_argument("--spotcheck",action="store_true", help="Preview perturbed tasks")
    parser.add_argument("--results",  action="store_true", help="Print results from saved file")
    parser.add_argument("--model",    choices=list(MODEL_CALLERS.keys()), default="openai")
    parser.add_argument("--key",      type=str, default=None)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--file",     type=str, default=None, help="Results file for --results")
    parser.add_argument(
        "--perturbation",
        choices=["surface", "distractor", "reframe", "all"],
        default="all",
        help="Which perturbation type(s) to apply (default: all)"
    )
    args = parser.parse_args()

    perturbation_types = (
        ["surface", "distractor", "reframe"]
        if args.perturbation == "all"
        else [args.perturbation]
    )

    if args.results:
        filepath = args.file or f"facet_perturb_results_{args.model}_seed{args.seed}.json"
        load_and_print_results(filepath, perturbation_types)

    else:
        with open(DATASET_FILE) as f:
            tasks = json.load(f)["tasks"]

        if args.spotcheck:
            run_spotcheck(tasks, perturbation_types, seed=args.seed)

        elif args.eval:
            api_key = args.key or {
                "openai":      os.environ.get("OPENAI_API_KEY"),
                "groq":        os.environ.get("GROQ_API_KEY"),
                "deepseek":    os.environ.get("DEEPSEEK_API_KEY"),
                "deepseek_r1": os.environ.get("DEEPSEEK_API_KEY"),
                "gemini":      os.environ.get("GEMINI_API_KEY"),
                "qwen":        os.environ.get("GROQ_API_KEY"),
            }.get(args.model)

            if not api_key:
                print(f"Error: No API key. Use --key YOUR_KEY or set the environment variable.")
                exit(1)

            run_eval(tasks, args.model, api_key, perturbation_types, seed=args.seed)

        else:
            print("FACET Perturbation Evaluation\n")
            print("Usage:")
            print("  python facet_perturbation.py --eval --model groq --key YOUR_KEY")
            print("  python facet_perturbation.py --spotcheck --model groq --key YOUR_KEY")
            print("  python facet_perturbation.py --results --file facet_perturb_results_groq_seed42.json")
            print("\nPerturbation types (--perturbation):")
            print("  surface    — rename entities, shift numbers (default: all three)")
            print("  distractor — inject irrelevant sentence into question stem")
            print("  reframe    — wrap question in a scenario/story context")
            print("  all        — run all three (default)")