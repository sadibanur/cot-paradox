"""
FACET — Confidence Calibration Evaluation
==========================================
Elicits verbalized confidence (0–100%) alongside answers for all tasks,
then computes three standard calibration metrics:

  ECE   — Expected Calibration Error (lower = better calibrated)
  AUROC — Discrimination ability: does confidence rank correct > incorrect?
  OE    — Overconfidence/Underconfidence bias (mean confidence − accuracy)

Prompt design follows literature best practice (Xiong et al. 2024;
Yang et al. 2024): confidence is elicited BEFORE the answer letter,
which yields more conservative, better-calibrated scores.

Output: facet_calib_results_{model}_seed{seed}.json

Usage:
    python facet_calibration.py --eval --model groq --key gsk_xxx
    python facet_calibration.py --eval --model openai --key sk-proj-xxx
    python facet_calibration.py --eval --model deepseek --key xxx
    python facet_calibration.py --eval --model gemini --key xxx
    python facet_calibration.py --eval --model qwen --key xxx
    python facet_calibration.py --eval --model deepseek_r1 --key xxx
    python facet_calibration.py --results --file facet_calib_results_groq_seed42.json
    python facet_calibration.py --spotcheck
"""

import json
import random
import argparse
import os
import re
import time
import math
from collections import defaultdict
from datetime import datetime

DATASET_FILE = "facet_v2_dataset.json"

CATEGORIES = [
    "reversal_curse", "compositional_reasoning", "syllogistic_reasoning",
    "working_memory", "inhibitory_control", "counting",
    "anchoring_bias", "theory_of_mind",
]

# ─────────────────────────────────────────────────────────────
# PROMPT
# Confidence BEFORE answer — per literature best practice
# ─────────────────────────────────────────────────────────────

def build_calibration_prompt(task):
    options_text = "\n".join(f"({k}) {v}" for k, v in task["options"].items())
    return (
        f"{task['question']}\n\n"
        f"{options_text}\n\n"
        f"Before answering, rate your confidence that you know the correct answer.\n"
        f"Respond in this exact format on two lines:\n"
        f"CONFIDENCE: <integer 0-100>\n"
        f"ANSWER: <single letter A, B, C, or D>"
    )


# ─────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_confidence_and_answer(response_text):
    """
    Returns (confidence: float|None, answer: str|None).
    Tries structured format first, falls back to regex search.

    Handles Qwen3 / DeepSeek-R1 think-block format:
      - Searches inside <think>...</think> first for the structured lines
      - Falls back to the text outside think blocks
    """
    if not response_text:
        return None, None

    raw = response_text.strip()

    # Pull out think-block content separately (Qwen3, DeepSeek-R1)
    think_content = ""
    think_matches = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    if think_matches:
        think_content = "\n".join(think_matches)

    # Outside-think text (strip the think blocks)
    outside_text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    # Extract text that appears AFTER the closing </think> tag — this is
    # where Qwen3 puts its final structured answer even when think is long
    post_think = ""
    post_match = re.search(r"</think>(.*)", raw, re.DOTALL | re.IGNORECASE)
    if post_match:
        post_think = post_match.group(1).strip()

    # If no closing </think> tag (response truncated mid-think), use the
    # raw think content itself — CONFIDENCE may appear near the end of it
    has_closing_think = bool(re.search(r"</think>", raw, re.IGNORECASE))
    truncated_think = think_content if (think_content and not has_closing_think) else ""

    # Search order: post-think → outside text → think content → raw truncated think
    search_texts = [t for t in [post_think, outside_text, think_content, truncated_think, raw] if t]
    text = outside_text or raw   # primary text for legacy fallbacks

    # ── Extract confidence (search outside text first, then think block) ───
    confidence = None

    for t in search_texts:
        m = re.search(r"CONFIDENCE\s*:\s*(\d{1,3})", t, re.IGNORECASE)
        if m:
            confidence = min(max(int(m.group(1)), 0), 100)
            break

    if confidence is None:
        for t in search_texts:
            m = re.search(r"(\d{1,3})\s*%\s*(?:confident|sure|certain)", t, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 100:
                    confidence = val
                    break

    if confidence is None:
        for t in search_texts:
            m = re.search(r"\b(\d{1,3})\s*%", t)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 100:
                    confidence = val
                    break

    # ── Extract answer (search outside text first, then think block) ────────
    answer = None

    for t in search_texts:
        m = re.search(r"ANSWER\s*:\s*([ABCD])\b", t, re.IGNORECASE)
        if m:
            answer = m.group(1).upper()
            break

    # Fallbacks (same priority chain as facet_eval.py)
    if answer is None:
        text_up = text.upper()
        for pattern in [
            r"(?:MY\s+)?(?:FINAL\s+)?ANSWER(?:\s+IS)?[\s:]+([ABCD])\b",
            r"(?:THEREFORE|SO|THUS|HENCE)[,\s]+(?:THE\s+ANSWER\s+IS\s+)?([ABCD])\b",
            r"(?:I\s+)?(?:CHOOSE|SELECT|PICK|GO\s+WITH)\s+([ABCD])\b",
            r"OPTION\s*\(?([ABCD])\)?",
        ]:
            m = re.search(pattern, text_up)
            if m:
                answer = m.group(1)
                break

    if answer is None:
        matches = re.findall(r"\(([ABCD])\)", text.upper())
        if matches:
            answer = matches[-1]

    if answer is None:
        tail = text.upper()[-100:]
        m = re.search(r"\b([ABCD])\b\.?\s*$", tail)
        if m:
            answer = m.group(1)

    if answer is None:
        m = re.search(r"\\BOXED\{([ABCD])\}", text.upper())
        if m:
            answer = m.group(1)

    if answer is None:
        m = re.search(r"\*\*([ABCD])\*\*", text.upper())
        if m:
            answer = m.group(1)

    return confidence, answer


# ─────────────────────────────────────────────────────────────
# CALIBRATION METRICS
# ─────────────────────────────────────────────────────────────

def compute_ece(confidences, corrects, n_bins=10):
    """
    Expected Calibration Error using equal-width bins [0,10), [10,20), ..., [90,100].
    ECE = Σ (|bin| / N) * |accuracy(bin) − confidence(bin)|
    Lower is better. Perfect calibration = 0.
    """
    bins = [[] for _ in range(n_bins)]
    for conf, corr in zip(confidences, corrects):
        bin_idx = min(int(conf / 100 * n_bins), n_bins - 1)
        bins[bin_idx].append((conf / 100, corr))

    ece = 0.0
    n = len(confidences)
    bin_details = []
    for i, b in enumerate(bins):
        if not b:
            bin_details.append(None)
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc  = sum(cr for _, cr in b) / len(b)
        ece += (len(b) / n) * abs(avg_acc - avg_conf)
        bin_details.append({
            "bin_lower": i * 10,
            "bin_upper": (i + 1) * 10,
            "count": len(b),
            "avg_confidence": round(avg_conf * 100, 1),
            "avg_accuracy":   round(avg_acc  * 100, 1),
            "gap":            round((avg_acc - avg_conf) * 100, 1),
        })
    return round(ece * 100, 2), bin_details   # return as percentage points


def compute_auroc(confidences, corrects):
    """
    Area Under ROC Curve via trapezoidal rule.
    Measures whether model ranks correct answers with higher confidence.
    0.5 = random, 1.0 = perfect discrimination.
    """
    # Sort by confidence descending
    pairs = sorted(zip(confidences, corrects), key=lambda x: -x[0])
    n_pos = sum(corrects)
    n_neg = len(corrects) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None   # undefined

    tp, fp = 0, 0
    auroc = 0.0
    prev_fp = 0
    prev_tp = 0
    for conf, corr in pairs:
        if corr:
            tp += 1
        else:
            fp += 1
        # Trapezoid area
        auroc += (fp - prev_fp) * (tp + prev_tp) / 2
        prev_fp, prev_tp = fp, tp

    auroc /= (n_pos * n_neg)
    return round(auroc, 4)


def compute_brier(confidences, corrects):
    """
    Brier Score = mean((confidence/100 − correct)²). Lower = better.
    Proper scoring rule that penalises both miscalibration and discrimination.
    """
    return round(
        sum((c / 100 - cr) ** 2 for c, cr in zip(confidences, corrects)) / len(confidences),
        4
    )


# ─────────────────────────────────────────────────────────────
# MODEL CALLERS  (mirrors facet_eval.py exactly)
# ─────────────────────────────────────────────────────────────

def call_openai(prompt, api_key, model="gpt-4o-mini"):
    try:
        import openai
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50, temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def call_groq(prompt, api_key, model="llama-3.1-8b-instant"):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 50, "temperature": 0}
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
                   "max_tokens": 50, "temperature": 0}
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

def _call_gemini_base(prompt, api_key, model, max_output_tokens=512):
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {
            "parts": [{"text": (
                "You must respond in this exact format with no preamble:\n"
                "CONFIDENCE: <integer 0-100>\n"
                "ANSWER: <single letter A, B, C, or D>"
            )}]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_output_tokens}
    }
    for attempt in range(4):
        try:
            r = requests.post(f"{url}?key={api_key}",
                             headers={"Content-Type": "application/json"},
                             json=payload, timeout=60)
            if r.status_code in [503, 429]:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
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
        # System prompt forces structured output BEFORE any reasoning.
        # Qwen3 writes CONFIDENCE/ANSWER first, then optionally reasons — 
        # this ensures the structured lines appear before the token limit.
        system = (
            "You must respond in this exact format with no preamble:\n"
            "CONFIDENCE: <integer 0-100>\n"
            "ANSWER: <single letter A, B, C, or D>\n"
            "You may then reason briefly if needed, but the two lines above must come first."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 8192,
            "temperature": 0,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        # Return raw — extractor handles <think> blocks
        return r.json()["choices"][0]["message"]["content"]
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
# SPOTCHECK
# ─────────────────────────────────────────────────────────────

def run_spotcheck(tasks, seed=42, n=8):
    rng = random.Random(seed)
    sample = rng.sample(tasks, min(n, len(tasks)))
    print(f"\n{'='*65}\nFACET Calibration Spotcheck — {n} tasks\n{'='*65}")
    for i, task in enumerate(sample, 1):
        print(f"\n[{i:02d}] {task['category']} | {task['difficulty']}")
        print(f"Q: {task['question'][:120]}...")
        print(f"Answer: ({task['answer']})")
        print(f"\nPrompt preview (last 4 lines):")
        prompt = build_calibration_prompt(task)
        print('\n'.join(prompt.split('\n')[-4:]))
    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────

def run_eval(tasks, model_name, api_key, seed=42):
    random.seed(seed)
    random.shuffle(tasks)
    caller = MODEL_CALLERS[model_name]
    total  = len(tasks)

    print(f"\n{'='*65}")
    print(f"FACET Confidence Calibration Evaluation")
    print(f"Model  : {model_name}")
    print(f"Tasks  : {total}")
    print(f"{'='*65}\n")

    results = []
    errors  = 0
    conf_parse_failures = 0

    for i, task in enumerate(tasks, 1):
        prompt = build_calibration_prompt(task)
        label  = f"{task['category'][:16]}/{task['difficulty'][:4]}"
        print(f"[{i:03d}/{total}] {label:<22}", end="", flush=True)

        # Retry on rate limits
        response = None
        for attempt in range(3):
            response = caller(prompt, api_key)
            if response and "429" in str(response):
                wait = 60 * (attempt + 1)
                print(f" ⏳ {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            break

        if response and (str(response).startswith("ERROR:") or "429" in str(response)):
            confidence, answer, correct = None, None, False
            errors += 1
            print(f" ⚠ {str(response)[:50]}")
        else:
            confidence, answer = extract_confidence_and_answer(response)
            correct = (answer == task["answer"]) if answer else False

            if confidence is None:
                conf_parse_failures += 1
                conf_str = "conf=?"
            else:
                conf_str = f"conf={confidence:3d}%"

            status = "✓" if correct else ("?" if not answer else "✗")
            print(f" {status} {conf_str}  (got {answer or '?'}, exp {task['answer']})")

        results.append({
            "id":               task["id"],
            "category":         task["category"],
            "difficulty":       task["difficulty"],
            "correct_answer":   task["answer"],
            "response":         (response or "")[:500],
            "extracted_answer": answer,
            "confidence":       confidence,
            "correct":          correct,
        })
        time.sleep(0.3)

    # ── Save ───────────────────────────────────────────────
    output = {
        "model":                model_name,
        "run_date":             datetime.now().isoformat(),
        "seed":                 seed,
        "total_tasks":          total,
        "api_errors":           errors,
        "conf_parse_failures":  conf_parse_failures,
        "results":              results,
    }
    filename = f"facet_calib_results_{model_name}_seed{seed}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {filename}")
    print(f"Conf parse failures: {conf_parse_failures}/{total}")

    print_summary(results, model_name)


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(results, model_name):
    # Filter to tasks where we have both confidence and answer
    valid = [r for r in results if r["confidence"] is not None and r["extracted_answer"] is not None]
    if not valid:
        print("No valid results to summarise.")
        return

    confs   = [r["confidence"] for r in valid]
    corrects = [int(r["correct"]) for r in valid]

    overall_acc  = sum(corrects) / len(corrects) * 100
    mean_conf    = sum(confs) / len(confs)
    overconf_bias = mean_conf - overall_acc   # positive = overconfident

    ece, bin_details = compute_ece(confs, corrects)
    auroc = compute_auroc(confs, corrects)
    brier = compute_brier(confs, corrects)

    print(f"\n{'='*70}")
    print(f"FACET Calibration Results — {model_name}")
    print(f"{'='*70}")
    print(f"  Tasks with valid conf+answer : {len(valid)}/{len(results)}")
    print(f"  Accuracy                     : {overall_acc:.1f}%")
    print(f"  Mean stated confidence       : {mean_conf:.1f}%")
    bias_sign = "+" if overconf_bias >= 0 else ""
    print(f"  Overconfidence bias          : {bias_sign}{overconf_bias:.1f}pp  "
          f"({'overconfident' if overconf_bias > 0 else 'underconfident'})")
    print(f"  ECE  (↓ better, 0=perfect)  : {ece:.2f}pp")
    auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
    print(f"  AUROC (↑ better, 0.5=random): {auroc_str}")
    print(f"  Brier score (↓ better)      : {brier:.4f}")

    # ── Per-category breakdown ─────────────────────────────
    print(f"\n  {'Category':<28} {'Acc':>5} {'Conf':>5} {'Bias':>6} {'ECE':>6} {'AUROC':>6}")
    print(f"  {'─'*60}")
    cat_stats = defaultdict(lambda: {"confs": [], "corrects": []})
    for r in valid:
        cat_stats[r["category"]]["confs"].append(r["confidence"])
        cat_stats[r["category"]]["corrects"].append(int(r["correct"]))

    for cat in CATEGORIES:
        cs = cat_stats[cat]
        if not cs["confs"]:
            continue
        c_acc   = sum(cs["corrects"]) / len(cs["corrects"]) * 100
        c_conf  = sum(cs["confs"]) / len(cs["confs"])
        c_bias  = c_conf - c_acc
        c_ece, _ = compute_ece(cs["confs"], cs["corrects"])
        c_auroc = compute_auroc(cs["confs"], cs["corrects"])
        label   = cat.replace("_", " ").title()[:27]
        bias_s  = f"{'+' if c_bias>=0 else ''}{c_bias:.1f}"
        auroc_s = f"{c_auroc:.2f}" if c_auroc is not None else " N/A"
        print(f"  {label:<28} {c_acc:4.1f}% {c_conf:4.1f}% {bias_s:>6}pp {c_ece:5.1f}pp {auroc_s:>6}")

    # ── Calibration bins ──────────────────────────────────
    print(f"\n  Calibration bins (confidence range → actual accuracy):")
    print(f"  {'Range':<12} {'Count':>6} {'Avg Conf':>9} {'Avg Acc':>8} {'Gap':>7}")
    print(f"  {'─'*48}")
    for b in bin_details:
        if b is None:
            continue
        gap_str = f"{'+' if b['gap']>=0 else ''}{b['gap']:.1f}pp"
        print(f"  {b['bin_lower']:3d}–{b['bin_upper']:3d}%   "
              f"{b['count']:6d}   {b['avg_confidence']:7.1f}%   "
              f"{b['avg_accuracy']:6.1f}%   {gap_str:>8}")

    print(f"\n  Interpretation:")
    if overconf_bias > 15:
        print(f"  ⚠ Severely overconfident — model claims {overconf_bias:.0f}pp more confidence than warranted")
    elif overconf_bias > 5:
        print(f"  ⚠ Moderately overconfident — common in LLMs (GPT-4 benchmark: ~10pp bias)")
    elif overconf_bias < -5:
        print(f"  ✓ Underconfident — model hedges more than necessary")
    else:
        print(f"  ✓ Well-calibrated bias — within ±5pp of accuracy")

    if auroc is not None:
        if auroc > 0.75:
            print(f"  ✓ Good discrimination (AUROC {auroc:.2f}) — confidence tracks correctness well")
        elif auroc > 0.60:
            print(f"  ~ Moderate discrimination (AUROC {auroc:.2f}) — better than random, room to improve")
        else:
            print(f"  ✗ Poor discrimination (AUROC {auroc:.2f}) — confidence barely predicts correctness")

    print(f"{'='*70}\n")


def load_and_print_results(filepath):
    with open(filepath) as f:
        data = json.load(f)
    print_summary(data["results"], data["model"])


# ─────────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────────

def run_debug(tasks, model_name, api_key, n=5, seed=42):
    """Call the model on n tasks and print raw responses."""
    random.seed(seed)
    sample = random.sample(tasks, min(n, len(tasks)))
    caller = MODEL_CALLERS[model_name]
    print(f"\n{'='*65}")
    print(f"DEBUG — raw responses from {model_name} (n={n})")
    print(f"{'='*65}")
    for i, task in enumerate(sample, 1):
        prompt = build_calibration_prompt(task)
        response = caller(prompt, api_key)
        conf, ans = extract_confidence_and_answer(response)
        print(f"\n[{i}] {task['category']} | {task['difficulty']} | correct={task['answer']}")
        print(f"─── RAW RESPONSE ({len(response or '')} chars) ───")
        print(repr((response or 'None')[:800]))
        print(f"─── EXTRACTED: conf={conf}  answer={ans} ───")
    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACET Confidence Calibration")
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--spotcheck", action="store_true")
    parser.add_argument("--results",   action="store_true")
    parser.add_argument("--debug",     action="store_true", help="Print raw responses for 5 tasks")
    parser.add_argument("--model",     choices=list(MODEL_CALLERS.keys()), default="openai")
    parser.add_argument("--key",       type=str, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--file",      type=str, default=None)
    args = parser.parse_args()

    if args.results:
        filepath = args.file or f"facet_calib_results_{args.model}_seed{args.seed}.json"
        load_and_print_results(filepath)
    else:
        with open(DATASET_FILE) as f:
            tasks = json.load(f)["tasks"]

        if args.spotcheck:
            run_spotcheck(tasks, seed=args.seed)

        elif args.debug:
            api_key = args.key or {
                "openai":      os.environ.get("OPENAI_API_KEY"),
                "groq":        os.environ.get("GROQ_API_KEY"),
                "deepseek":    os.environ.get("DEEPSEEK_API_KEY"),
                "deepseek_r1": os.environ.get("DEEPSEEK_API_KEY"),
                "gemini":      os.environ.get("GEMINI_API_KEY"),
                "qwen":        os.environ.get("GROQ_API_KEY"),
            }.get(args.model)
            if not api_key:
                print("Error: No API key.")
                exit(1)
            run_debug(tasks, args.model, api_key, seed=args.seed)

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
                print(f"Error: No API key. Use --key YOUR_KEY or set environment variable.")
                exit(1)

            run_eval(tasks, args.model, api_key, seed=args.seed)

        else:
            print("FACET Confidence Calibration\n")
            print("Usage:")
            print("  python facet_calibration.py --eval --model groq --key YOUR_KEY")
            print("  python facet_calibration.py --debug --model qwen --key YOUR_KEY")
            print("  python facet_calibration.py --spotcheck")
            print("  python facet_calibration.py --results --file facet_calib_results_groq_seed42.json")
            print("\nModels: openai | groq | deepseek | deepseek_r1 | gemini | qwen")