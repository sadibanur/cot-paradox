"""
FACET — Phase 5: Full Evaluation Script
=========================================
Runs the complete 400-task evaluation across all 8 categories
under zero-shot and chain-of-thought prompting conditions.

Supports: openai, groq, deepseek, gemini

Usage:
    python facet_phase5_eval.py --eval --model groq --key gsk_xxx
    python facet_phase5_eval.py --eval --model openai --key sk-proj-xxx
    python facet_phase5_eval.py --eval --model deepseek --key xxx
    python facet_phase5_eval.py --results
    python facet_phase5_eval.py --spotcheck
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
RESULTS_FILE = "facet_full_results.json"

CATEGORIES = [
    "reversal_curse",
    "compositional_reasoning",
    "syllogistic_reasoning",
    "working_memory",
    "inhibitory_control",
    "counting",
    "anchoring_bias",
    "theory_of_mind",
]


def load_dataset():
    with open(DATASET_FILE) as f:
        data = json.load(f)
    return data["tasks"]


def run_spotcheck(tasks, seed=42):
    random.seed(seed)
    groups = defaultdict(list)
    for task in tasks:
        key = (task["category"], task["difficulty"])
        groups[key].append(task)
    sampled = []
    for key, group in groups.items():
        n = max(1, round(len(group) * 0.10))
        sampled.extend(random.sample(group, n))
    random.shuffle(sampled)
    print(f"\n{'='*65}\nFACET Spot-Check — {len(sampled)} tasks (10% sample)\n{'='*65}")
    for i, task in enumerate(sampled, 1):
        cat = task["category"].replace("_", " ").title()
        diff = task["difficulty"].upper()
        print(f"\n--- Task {i:02d}/{len(sampled)} | {cat} | {diff} ---")
        print(f"Q: {task['question']}")
        for letter, text in task["options"].items():
            marker = "  <- CORRECT" if letter == task["answer"] else ""
            print(f"   ({letter}) {text}{marker}")
        print(f"Exp: {task['explanation']}")
    print(f"\nDone.")


def build_prompt(task, cot=False):
    options_text = "\n".join(f"({k}) {v}" for k, v in task["options"].items())
    if cot:
        return (
            f"Think step by step before answering.\n\n"
            f"{task['question']}\n\n"
            f"{options_text}\n\n"
            f"After thinking, give your final answer as a single letter: A, B, C, or D."
        )
    return (
        f"{task['question']}\n\n"
        f"{options_text}\n\n"
        f"Answer with a single letter only: A, B, C, or D."
    )


def extract_answer(response_text):
    """Extract A/B/C/D from model response. Handles long CoT responses."""
    if not response_text:
        return None
    text = response_text.strip().upper()

    # P1: Explicit answer statement
    match = re.search(r"(?:MY\s+)?(?:FINAL\s+)?ANSWER(?:\s+IS)?[\s:]+([ABCD])\b", text)
    if match:
        return match.group(1)

    # P2: "Therefore/So/Thus the answer is X"
    match = re.search(r"(?:THEREFORE|SO|THUS|HENCE)[,\s]+(?:THE\s+ANSWER\s+IS\s+)?([ABCD])\b", text)
    if match:
        return match.group(1)

    # P3: "Correct answer: X"
    match = re.search(r"CORRECT\s+ANSWER[\s:]+([ABCD])\b", text)
    if match:
        return match.group(1)

    # P4: "I choose/select/pick X"
    match = re.search(r"(?:I\s+)?(?:CHOOSE|SELECT|PICK|GO\s+WITH)\s+([ABCD])\b", text)
    if match:
        return match.group(1)

    # P5: "Option X" or "Option (X)"
    match = re.search(r"OPTION\s*\(?([ABCD])\)?", text)
    if match:
        return match.group(1)

    # P6: Last "(X)" in response — very reliable for MCQ
    matches = re.findall(r"\(([ABCD])\)", text)
    if matches:
        return matches[-1]

    # P7: Last standalone letter in final 100 chars
    tail = text[-100:]
    match = re.search(r"\b([ABCD])\b\.?\s*$", tail)
    if match:
        return match.group(1)

    # P8: Last standalone letter anywhere
    match = re.search(r"\b([ABCD])\b\.?\s*$", text)
    if match:
        return match.group(1)

    # P9: LaTeX boxed answer — DeepSeek-R1 style: \boxed{B}
    match = re.search(r"\\BOXED\{([ABCD])\}", text)
    if match:
        return match.group(1)

    # P10: Bold markdown answer — DeepSeek-V3 style: **B** or Final answer: **B**
    match = re.search(r"\*\*([ABCD])\*\*", text)
    if match:
        return match.group(1)

    # P10: Only letter in short response
    if len(text.split()) <= 5:
        match = re.search(r"\b([ABCD])\b", text)
        if match:
            return match.group(1)

    return None

def call_openai(prompt, api_key, model="gpt-4o-mini"):
    """Call OpenAI API."""
    try:
        import openai
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def call_groq(prompt, api_key, model="llama-3.1-8b-instant"):
    """Call Groq API (free tier) — runs LLaMA 3.1 8B."""
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 500, "temperature": 0}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def call_deepseek(prompt, api_key, model="deepseek-chat"):
    """Call DeepSeek V3 — ~$0.01 for full FACET evaluation."""
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 500, "temperature": 0}
        r = requests.post("https://api.deepseek.com/chat/completions",
                         headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def call_deepseek_r1(prompt, api_key, model="deepseek-reasoner"):
    """Call DeepSeek R1 — reasoning-optimised model (~$0.50 for FACET)."""
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 2000, "temperature": 0}
        r = requests.post("https://api.deepseek.com/chat/completions",
                         headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _call_gemini_base(prompt, api_key, model, max_output_tokens):
    """Shared Gemini REST caller with retry logic and thinking token handling."""
    import requests, time as _time
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
                wait = 30 * (attempt + 1)
                _time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            text_parts = [p["text"] for p in parts if "text" in p]
            if text_parts:
                return text_parts[-1]
            return "ERROR: no text parts in response"
        except Exception as e:
            if attempt < 3:
                _time.sleep(30)
                continue
            return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


def call_gemini(prompt, api_key, model="gemini-2.5-flash"):
    """Gemini 2.5 Flash — fast, stable, free tier available.
    Cost: ~/bin/sh.075/M input tokens. Full FACET 3 runs ~ /bin/sh.05 total."""
    return _call_gemini_base(prompt, api_key, model, max_output_tokens=8192)


def call_gemini_pro(prompt, api_key, model="gemini-2.5-pro"):
    """Gemini 2.5 Pro — Google best reasoning model.
    Cost: ~.25/M input tokens. Full FACET 3 runs ~ -5 total.
    May return 503 under high load — run during off-peak hours."""
    return _call_gemini_base(prompt, api_key, model, max_output_tokens=2000)


def call_qwen(prompt, api_key, model="qwen/qwen3-32b"):
    """Call Qwen3-32B via Groq — free on Developer tier.
    Qwen3 has built-in thinking mode; we strip thinking tokens from output.
    Uses the same Groq API key as LLaMA."""
    try:
        import requests, re
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        # Strip <think>...</think> reasoning tokens if present
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
    "gemini_pro":  call_gemini_pro,
    "qwen":        call_qwen,
}




def run_eval(tasks, model_name, api_key, seed=42, cot_only=False):
    random.seed(seed)
    random.shuffle(tasks)
    total = len(tasks)

    print(f"\n{'='*60}")
    print(f"FACET Phase 5 — Full Evaluation")
    print(f"Model  : {model_name}")
    print(f"Tasks  : {total} (full dataset)")
    mode = "chain-of-thought ONLY (re-run)" if cot_only else "zero-shot + chain-of-thought"
    print(f"Prompts: {mode}")
    print(f"{'='*60}\n")

    caller = MODEL_CALLERS[model_name]
    results = []
    errors = 0

    for i, task in enumerate(tasks, 1):
        task_result = {
            "id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "num_steps": task["num_steps"],
            "correct_answer": task["answer"],
        }

        conditions = ["zero_shot", "cot"] if not cot_only else ["cot"]
        for condition in conditions:
            cot = (condition == "cot")
            prompt = build_prompt(task, cot=cot)
            label = task["category"][:18]
            print(
                f"[{i:03d}/{total}] {label:<18} {task['difficulty']:<6} {condition:<10}",
                end="", flush=True
            )

            # Retry up to 3 times on rate limit errors with exponential backoff
            response = None
            for attempt in range(3):
                response = caller(prompt, api_key)
                if response and "429" in str(response):
                    wait = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f" ⏳ Rate limited — waiting {wait}s...", end="", flush=True)
                    time.sleep(wait)
                    continue
                break

            if response and (response.startswith("ERROR:") or "429" in response):
                extracted = None
                correct = False
                errors += 1
                print(f" ⚠ {response[:50]}")
            else:
                # Extract from FULL response — never truncate before extraction
                extracted = extract_answer(response)
                correct = (extracted == task["answer"]) if extracted else False
                status = "✓" if correct else ("?" if not extracted else "✗")
                print(f" {status} (got {extracted or '?'}, expected {task['answer']})")

            task_result[condition] = {
                # Save truncated version to keep file sizes manageable
                "response": (response or "")[:3000],
                "extracted_answer": extracted,
                "correct": correct,
            }
            time.sleep(0.3)

        results.append(task_result)

    output = {
        "model": model_name,
        "run_date": datetime.now().isoformat(),
        "seed": seed,
        "total_tasks": total,
        "api_errors": errors,
        "results": results,
    }
    # Auto-save with model+seed filename so runs are never overwritten
    suffix = "_cot_only" if cot_only else ""
    auto_filename = f"facet_results_{model_name}_seed{seed}{suffix}.json"
    with open(auto_filename, "w") as f:
        json.dump(output, f, indent=2)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {auto_filename}")


def print_summary(results, model_name):
    stats = defaultdict(lambda: defaultdict(lambda: {"zero_shot": [], "cot": []}))
    unextracted = {"zero_shot": 0, "cot": 0}

    for r in results:
        cat  = r["category"]
        diff = r["difficulty"]
        for cond in ["zero_shot", "cot"]:
            if cond in r:
                stats[cat][diff][cond].append(r[cond]["correct"])
                if r[cond]["extracted_answer"] is None:
                    unextracted[cond] += 1

    def pct(lst):
        if not lst:
            return "  -  "
        return f"{sum(lst)/len(lst)*100:5.1f}%"

    print(f"\n{'='*65}")
    print(f"Pilot Results Summary — {model_name}")
    print(f"{'='*65}")
    print(f"{'Reasoning Type':<28} {'Diff':<8} {'Zero-shot':>10} {'CoT':>10}")
    print(f"{'─'*65}")

    overall_zs, overall_cot = [], []
    for cat in CATEGORIES:
        for diff in ["easy", "medium", "hard"]:
            zs  = stats[cat][diff]["zero_shot"]
            cot = stats[cat][diff]["cot"]
            overall_zs.extend(zs)
            overall_cot.extend(cot)
            label = cat.replace("_", " ").title()[:27]
            print(f"{label:<28} {diff:<8} {pct(zs):>10} {pct(cot):>10}")
        print(f"{'─'*65}")

    print(f"{'OVERALL':<28} {'':8} {pct(overall_zs):>10} {pct(overall_cot):>10}")
    print(f"{'='*65}")

    total_tasks = len(results)
    if unextracted["zero_shot"] > 0 or unextracted["cot"] > 0:
        print(f"\nUnextracted answers (counted as wrong):")
        print(f"  Zero-shot: {unextracted['zero_shot']}/{total_tasks}")
        print(f"  CoT:       {unextracted['cot']}/{total_tasks}")

    print(f"\nDifficulty ordering check (should be easy > medium > hard):")
    print(f"{'─'*50}")
    for cat in CATEGORIES:
        easy   = stats[cat]["easy"]["zero_shot"]
        medium = stats[cat]["medium"]["zero_shot"]
        hard   = stats[cat]["hard"]["zero_shot"]
        e = sum(easy)/len(easy)*100     if easy   else 0
        m = sum(medium)/len(medium)*100 if medium else 0
        h = sum(hard)/len(hard)*100     if hard   else 0
        ok = (e >= m >= h)
        status = "✓ OK" if ok else "✗ ISSUE — difficulty control may need revision"
        label = cat.replace("_", " ").title()[:30]
        print(f"  {label:<32} easy:{e:.0f}% mid:{m:.0f}% hard:{h:.0f}%  {status}")

    print(f"\nCoT gain per category:")
    print(f"{'─'*50}")
    for cat in CATEGORIES:
        zs_all  = (stats[cat]["easy"]["zero_shot"] +
                   stats[cat]["medium"]["zero_shot"] +
                   stats[cat]["hard"]["zero_shot"])
        cot_all = (stats[cat]["easy"]["cot"] +
                   stats[cat]["medium"]["cot"] +
                   stats[cat]["hard"]["cot"])
        if zs_all and cot_all:
            zs_pct  = sum(zs_all)/len(zs_all)*100
            cot_pct = sum(cot_all)/len(cot_all)*100
            gain = cot_pct - zs_pct
            sign = "+" if gain >= 0 else ""
            label = cat.replace("_", " ").title()[:30]
            print(f"  {label:<32} ZS:{zs_pct:.1f}%  CoT:{cot_pct:.1f}%  Gain:{sign}{gain:.1f}%")

    print(f"\nAnswer leakage check (flag if zero-shot > 90%):")
    print(f"{'─'*50}")
    flagged = False
    for cat in CATEGORIES:
        for diff in ["easy", "medium", "hard"]:
            zs = stats[cat][diff]["zero_shot"]
            if zs and sum(zs)/len(zs) > 0.90:
                label = cat.replace("_", " ")
                print(f"  ⚠ {label} / {diff}: {sum(zs)/len(zs)*100:.1f}% — possible answer leakage")
                flagged = True
    if not flagged:
        print("  No leakage detected — no group scored above 90%.")

    print(f"\n{'='*65}")


def load_and_print_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}. Run --eval first.")
        return
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    print_summary(data["results"], data["model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACET Phase 5 — Full Evaluation")
    parser.add_argument("--eval", action="store_true",
                        help="Run full evaluation on all 400 tasks")
    parser.add_argument("--spotcheck", action="store_true",
                        help="Print 40 random tasks for human review")
    parser.add_argument("--results", action="store_true",
                        help="Print results from a previous run")
    parser.add_argument("--model", choices=["openai", "groq", "deepseek", "deepseek_r1", "gemini", "gemini_pro", "qwen"],
                        default="openai", help="Which model to evaluate")
    parser.add_argument("--key", type=str, default=None,
                        help="API key for the chosen model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--cot_only", action="store_true",
                        help="Run CoT condition only (to fix extraction failures)")
    args = parser.parse_args()

    tasks = load_dataset()

    if args.spotcheck:
        run_spotcheck(tasks, seed=args.seed)

    elif args.eval:
        if not args.key:
            env_keys = {
                "openai":      os.environ.get("OPENAI_API_KEY"),
                "groq":        os.environ.get("GROQ_API_KEY"),
                "deepseek":    os.environ.get("DEEPSEEK_API_KEY"),
                "deepseek_r1": os.environ.get("DEEPSEEK_API_KEY"),
                "gemini":      os.environ.get("GEMINI_API_KEY"),
                "gemini_pro":  os.environ.get("GEMINI_API_KEY"),
                "qwen":        os.environ.get("HF_API_KEY"),
            }
            api_key = env_keys.get(args.model)
            if not api_key:
                print(f"Error: No API key. Use --key YOUR_KEY or set environment variable.")
                exit(1)
        else:
            api_key = args.key
        run_eval(tasks, args.model, api_key, seed=args.seed, cot_only=args.cot_only)

    elif args.results:
        load_and_print_results()

    else:
        print("FACET Phase 5 — Full Evaluation")
        print("\nUsage:")
        print("  python facet_eval.py --eval --model groq --key YOUR_KEY")
        print("  python facet_eval.py --eval --model openai --key YOUR_KEY")
        print("  python facet_eval.py --eval --model deepseek --key YOUR_KEY")
        print("  python facet_eval.py --spotcheck")
        print("  python facet_eval.py --results")