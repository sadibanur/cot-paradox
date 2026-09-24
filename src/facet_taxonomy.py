"""
FACET — Error Taxonomy (Hybrid: Automated + Manual Validation)
==============================================================
Step 1 — AUTO:   Classifies all failures across all results files using Claude API
Step 2 — REVIEW: Interactive CLI for manual validation of ~50 failures per model
Step 3 — FIGURE: Generates Figure 6 — stacked bar chart of failure types per model

Error taxonomy (6 types):
  WRONG_REASONING    — model shows reasoning, logic is verifiably flawed
  RIGHT_REASON_WRONG — reasoning is correct but final answer contradicts it
  DISTRACTOR_CAPTURE — model latches onto irrelevant number/entity in question
  OVERCONFIDENT_FLIP — model states correct reasoning then reverses at the last step
  GUESSING           — short/no reasoning, no justification shown
  FORMAT_FAILURE     — answer not extractable / refusal / off-format response

Usage:
  python facet_taxonomy.py --auto   --key YOUR_ANTHROPIC_KEY
  python facet_taxonomy.py --review --model groq
  python facet_taxonomy.py --figure
  python facet_taxonomy.py --stats
"""

import json
import glob
import os
import re
import sys
import time
import random
import argparse
import ssl
from collections import defaultdict, Counter
from datetime import datetime

# Fix for macOS SSL certificate verification failure
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

TAXONOMY = {
    "WRONG_REASONING":    "Model reasons explicitly but logic is flawed",
    "RIGHT_REASON_WRONG": "Reasoning is correct but conclusion contradicts it",
    "DISTRACTOR_CAPTURE": "Model latches onto irrelevant number/entity in question",
    "OVERCONFIDENT_FLIP": "States correct path then reverses answer at last step",
    "GUESSING":           "Short/no reasoning, answer appears unjustified",
    "FORMAT_FAILURE":     "Response not parseable / refusal / off-format",
    "EXTRACTION_ERROR":   "Model answered correctly but extractor failed to parse it",
}

TAXONOMY_CODES = list(TAXONOMY.keys())

MODEL_LABELS = {
    "groq":        "LLaMA 3.1 8B",
    "openai":      "GPT-4o-mini",
    "deepseek":    "DeepSeek-V3",
    "deepseek_r1": "DeepSeek-R1",
    "qwen":        "Qwen3-32B",
    "gemini":      "Gemini 2.5 Flash",
}

RESULTS_GLOB   = "facet_results_*.json"
DATASET_FILE   = "facet_v2_dataset.json"
TAXONOMY_FILE  = "facet_taxonomy_auto.json"
VALIDATED_FILE = "facet_taxonomy_validated.json"
VALIDATE_N     = 50   # failures to manually validate per model


# ─────────────────────────────────────────────────────────────
# STEP 1 — AUTO CLASSIFICATION
# ─────────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """CATEGORY: {category}
DIFFICULTY: {difficulty}
QUESTION: {question}
CORRECT ANSWER: ({correct}) {correct_text}
MODEL ANSWER: ({model_ans}) {model_ans_text}
MODEL RESPONSE:
{response}

Category code:"""


def classify_failure(task_info, api_key):
    """Call Claude claude-sonnet-4-20250514 to classify a single failure."""
    import urllib.request
    import urllib.error

    prompt = CLASSIFIER_PROMPT.format(
        category=task_info["category"],
        difficulty=task_info["difficulty"],
        question=task_info["question"][:600],
        correct=task_info["correct_answer"],
        correct_text=task_info["correct_text"],
        model_ans=task_info["model_answer"] or "?",
        model_ans_text=task_info["model_answer_text"],
        response=task_info["response"][:800],
    )

    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 20,
        "system": [
            {
                "type": "text",
                "text": (
                    "You are an expert annotator for an LLM reasoning benchmark. "
                    "Classify each failed response into exactly one of these categories:\n\n"
                    "WRONG_REASONING    — The model shows its reasoning explicitly, but the logic contains a clear error.\n"
                    "RIGHT_REASON_WRONG — The model's stated reasoning is correct but the final answer contradicts it.\n"
                    "DISTRACTOR_CAPTURE — The model ignores the core question and latches onto an irrelevant number, name, or detail.\n"
                    "OVERCONFIDENT_FLIP — The model correctly identifies the answer mid-reasoning, then changes its mind without justification.\n"
                    "GUESSING           — The response is very short, shows no reasoning, and the answer appears unjustified.\n"
                    "FORMAT_FAILURE     — The response is a refusal, is off-topic, or the answer could not be extracted.\n"
                    "EXTRACTION_ERROR   — The model clearly states the correct answer letter but it was marked wrong — the extractor failed to parse it.\n\n"
                    "IMPORTANT: If the model's final stated answer matches the correct answer, classify as EXTRACTION_ERROR regardless of anything else.\n"
                    "Respond with ONLY the category code. No explanation."
                ),
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta":    "prompt-caching-2024-07-31",
        },
        method="POST"
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
                data = json.loads(resp.read())
                text = data["content"][0]["text"].strip().upper()
                # Extract the code — match any of the known codes
                for code in TAXONOMY_CODES:
                    if code in text:
                        return code
                return "WRONG_REASONING"  # fallback
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code == 429 or e.code == 529:
                wait = 60 * (attempt + 1)
                print(f"\n    ⏳ {'rate limit' if e.code==429 else 'overloaded'}, waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            if e.code == 401:
                print(f"\n    ✗ AUTH ERROR — check your Anthropic API key: {body}")
                return "API_KEY_ERROR"
            print(f"\n    ✗ HTTP {e.code}: {body}")
            return "FORMAT_FAILURE"
        except Exception as ex:
            print(f"\n    ✗ Exception: {type(ex).__name__}: {ex}")
            if attempt < 2:
                time.sleep(10)
                continue
            return "FORMAT_FAILURE"
    # All retries exhausted — return placeholder so we can retry later
    return "RETRY"


def load_failures_from_results():
    """
    Load all failed tasks from facet_results_*.json files.
    Joins question + options from facet_v2_dataset.json so the
    classifier has full context (not just the response text).
    Returns dict: {model_name: [task_info, ...]}
    """
    # Load dataset for question + options — source of truth
    dataset_by_id = {}
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE) as f:
            ds = json.load(f)
        for t in ds.get("tasks", []):
            dataset_by_id[t["id"]] = t
        print(f"  Loaded {len(dataset_by_id)} tasks from {DATASET_FILE}")
    else:
        print(f"  WARNING: {DATASET_FILE} not found — question/options will be empty!")

    all_failures = defaultdict(list)

    for filepath in sorted(glob.glob(RESULTS_GLOB)):
        with open(filepath) as f:
            data = json.load(f)

        model = data.get("model", "unknown")
        model_key = model
        for k in MODEL_LABELS:
            if k in model.lower() or model.lower() in k:
                model_key = k
                break

        for r in data.get("results", []):
            task_id = r.get("id", "")
            # Join from dataset
            ds_task  = dataset_by_id.get(task_id, {})
            question = ds_task.get("question") or r.get("question", "")
            options  = ds_task.get("options")  or r.get("options", {})
            correct  = ds_task.get("answer")   or r.get("correct_answer") or r.get("answer", "")

            for condition in ["zero_shot", "cot"]:
                if condition not in r:
                    continue
                cond = r[condition]
                extracted = cond.get("extracted_answer")
                if not correct:
                    continue

                is_wrong = (extracted != correct)
                is_null  = (extracted is None)
                if not (is_wrong or is_null):
                    continue

                response = cond.get("response", "") or ""

                all_failures[model_key].append({
                    "id":                task_id,
                    "category":          r.get("category", ""),
                    "difficulty":        r.get("difficulty", ""),
                    "condition":         condition,
                    "question":          question[:800],
                    "options":           options,
                    "correct_answer":    correct,
                    "correct_text":      options.get(correct, "") if isinstance(options, dict) else "",
                    "model_answer":      extracted,
                    "model_answer_text": options.get(extracted, "") if (isinstance(options, dict) and extracted) else "",
                    "response":          response[:1200],
                    "auto_label":        None,
                    "human_label":       None,
                    "human_note":        "",
                })

    return all_failures


def validate_api_key(api_key):
    """Quick test call to verify the key works before running 13k classifications."""
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Hi"}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01",
                 "anthropic-beta": "prompt-caching-2024-07-31"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15, context=SSL_CONTEXT) as resp:
            json.loads(resp.read())
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"\n✗ API key validation failed (HTTP {e.code}): {body}")
        return False
    except Exception as ex:
        print(f"\n✗ API key validation failed: {ex}")
        return False


def run_auto_classification(api_key):
    """Classify all failures automatically and save to TAXONOMY_FILE."""
    print(f"\n{'='*65}")
    print("FACET Error Taxonomy — Automated Classification")
    print(f"{'='*65}\n")

    print("Validating API key...", end="", flush=True)
    if not validate_api_key(api_key):
        print("\nFix your API key and try again.")
        return
    print(" ✓\n")

    all_failures = load_failures_from_results()

    if not all_failures:
        print(f"No results files found matching: {RESULTS_GLOB}")
        return

    # ── Resume: load any previously saved progress ─────────────────
    existing = {}
    if os.path.exists(TAXONOMY_FILE):
        with open(TAXONOMY_FILE) as f:
            prev = json.load(f)
        for mk, tasks in prev.get("models", {}).items():
            existing[mk] = {}
            for t in tasks:
                if t.get("auto_label") and t["auto_label"] != "RETRY":
                    existing[mk][t["id"] + t["condition"]] = t["auto_label"]
        total_done = sum(len(v) for v in existing.values())
        print(f"Resuming from previous run — {total_done} already classified, skipping those.\n")

    total = sum(len(v) for v in all_failures.values())
    todo  = sum(1 for mk, tasks in all_failures.items()
                for t in tasks
                if t["id"] + t["condition"] not in existing.get(mk, {}))
    print(f"Found {total} failures across {len(all_failures)} models")
    print(f"To classify: {todo}  |  Already done: {total - todo}\n")

    call_num   = 0
    save_every = 50   # save progress every N classifications

    for model_key in sorted(all_failures.keys()):
        failures  = all_failures[model_key]
        done_keys = existing.get(model_key, {})
        label     = MODEL_LABELS.get(model_key, model_key)

        # Restore previously saved labels
        for t in failures:
            key = t["id"] + t["condition"]
            if key in done_keys:
                t["auto_label"] = done_keys[key]

        pending = [t for t in failures if not t.get("auto_label") or t["auto_label"] == "RETRY"]
        if not pending:
            print(f"\n── {label} — all {len(failures)} already done, skipping ──")
            continue

        print(f"\n── {label} ({len(pending)} remaining / {len(failures)} total) ──────────")

        for t in pending:
            call_num += 1
            cat_short = t["category"][:18]
            print(f"  [{call_num:04d}] {cat_short:<20} {t['difficulty']:<6} "
                  f"{t['condition']:<10}", end="", flush=True)

            label_code = classify_failure(t, api_key)
            t["auto_label"] = label_code
            print(f" → {label_code}")

            if label_code == "API_KEY_ERROR":
                print("\nStopping — invalid API key.")
                return

            # Save progress every N calls
            if call_num % save_every == 0:
                _save_progress(all_failures)
                print(f"  💾 Progress saved ({call_num} classified)")

            time.sleep(0.3)

    _save_progress(all_failures)
    print(f"\n✓ Complete. Saved to {TAXONOMY_FILE}")
    print_auto_stats(all_failures)


def _save_progress(all_failures):
    total = sum(len(v) for v in all_failures.values())
    output = {
        "run_date":       datetime.now().isoformat(),
        "total_failures": total,
        "models":         {k: v for k, v in all_failures.items()},
    }
    with open(TAXONOMY_FILE, "w") as f:
        json.dump(output, f, indent=2)


def print_auto_stats(all_failures):
    print(f"\n{'='*65}")
    print("Auto-classification summary")
    print(f"{'='*65}")
    print(f"{'Model':<20} " + "  ".join(f"{c[:8]:>8}" for c in TAXONOMY_CODES))
    print("─" * 75)
    for model_key in sorted(all_failures.keys()):
        failures = all_failures[model_key]
        counts = Counter(t["auto_label"] for t in failures)
        total  = len(failures)
        label  = MODEL_LABELS.get(model_key, model_key)[:19]
        row    = "  ".join(f"{counts.get(c,0):7d}" for c in TAXONOMY_CODES)
        print(f"{label:<20} {row}  (n={total})")


# ─────────────────────────────────────────────────────────────
# STEP 2 — MANUAL VALIDATION (Interactive CLI)
# ─────────────────────────────────────────────────────────────

def load_results_lookup():
    """
    Build a lookup dict keyed by task_id+condition with full display content.
    - question, options: from facet_v2_dataset.json (source of truth)
    - response, extracted_answer: from facet_results_*.json
    """
    # Load dataset for question + options
    dataset_by_id = {}
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE) as f:
            ds = json.load(f)
        for t in ds.get("tasks", []):
            dataset_by_id[t["id"]] = t
    else:
        print(f"  Warning: {DATASET_FILE} not found — questions won't display")

    lookup = defaultdict(dict)
    for filepath in sorted(glob.glob(RESULTS_GLOB)):
        with open(filepath) as f:
            data = json.load(f)
        model = data.get("model", "")
        model_key = model
        for k in MODEL_LABELS:
            if k in model.lower() or model.lower() in k:
                model_key = k
                break

        for r in data.get("results", []):
            task_id = r.get("id", "")
            ds_task = dataset_by_id.get(task_id, {})
            options = ds_task.get("options") or r.get("options", {})
            correct = ds_task.get("answer") or r.get("correct_answer") or r.get("answer", "")
            question = ds_task.get("question") or r.get("question", "")

            for condition in ["zero_shot", "cot"]:
                if condition not in r:
                    continue
                cond      = r[condition]
                key       = task_id + condition
                extracted = cond.get("extracted_answer")
                lookup[model_key][key] = {
                    "question":          question,
                    "options":           options,
                    "correct_answer":    correct,
                    "correct_text":      options.get(correct, "") if isinstance(options, dict) else "",
                    "model_answer":      extracted,
                    "model_answer_text": options.get(extracted, "") if (isinstance(options, dict) and extracted) else "",
                    "response":          (cond.get("response") or "")[:1200],
                }
    return lookup


def run_manual_review(model_key):
    """Interactive terminal review of auto-classified failures for one model."""
    if not os.path.exists(TAXONOMY_FILE):
        print(f"Error: {TAXONOMY_FILE} not found. Run --auto first.")
        return

    with open(TAXONOMY_FILE) as f:
        data = json.load(f)

    # Load full content from original results files
    print("Loading task content from results files...", end="", flush=True)
    results_lookup = load_results_lookup()
    print(" ✓\n")

    # Load existing validated file if it exists
    validated = {}
    if os.path.exists(VALIDATED_FILE):
        with open(VALIDATED_FILE) as f:
            validated = json.load(f)

    all_failures = data["models"]
    if model_key not in all_failures:
        available = list(all_failures.keys())
        print(f"Model '{model_key}' not found. Available: {available}")
        return

    failures = all_failures[model_key]
    label    = MODEL_LABELS.get(model_key, model_key)

    # Sample VALIDATE_N failures, stratified by auto_label
    already_done = set(validated.get(model_key, {}).keys())
    remaining    = [t for t in failures if t["id"] + t["condition"] not in already_done
                    and t["auto_label"] is not None]

    # Stratified sample: proportional to auto_label distribution
    by_label = defaultdict(list)
    for t in remaining:
        by_label[t["auto_label"]].append(t)

    sample = []
    total_remaining = len(remaining)
    for lbl, items in by_label.items():
        n = max(1, round(VALIDATE_N * len(items) / total_remaining))
        sample.extend(random.sample(items, min(n, len(items))))
    random.shuffle(sample)
    sample = sample[:VALIDATE_N]

    if not sample:
        print(f"All {VALIDATE_N} validations already done for {label}.")
        return

    print(f"\n{'='*70}")
    print(f"FACET Manual Validation — {label}")
    print(f"Reviewing {len(sample)} failures (of {len(failures)} total)")
    print(f"{'='*70}")
    print("\nKeyboard shortcuts:")
    for i, code in enumerate(TAXONOMY_CODES, 1):
        print(f"  {i} = {code:<25} {TAXONOMY[code]}")
    print("  s = skip (keep auto label)    q = quit and save\n")

    if model_key not in validated:
        validated[model_key] = {}

    confirmed = 0
    changed   = 0

    for idx, task in enumerate(sample, 1):
        task_key = task["id"] + task["condition"]

        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(sample)}] {task['category']} | {task['difficulty']} | {task['condition']}")

        # Enrich from original results files
        task_key = task["id"] + task["condition"]
        full = results_lookup.get(model_key, {}).get(task_key, {})
        question   = full.get("question")    or task.get("question", "")
        options    = full.get("options")     or task.get("options", {})
        correct    = full.get("correct_answer") or task.get("correct_answer", "")
        model_ans  = full.get("model_answer")   or task.get("model_answer")
        response   = full.get("response")    or task.get("response", "")

        print(f"Q: {question[:400]}")
        print(f"\nOptions:")
        for k, v in options.items():
            marker = " ◀ CORRECT" if k == correct   else \
                     " ✗ MODEL"   if k == model_ans  else ""
            print(f"  ({k}) {v}{marker}")
        print(f"\nModel response (first 500 chars):")
        resp_display = response[:500] if response else "(no response saved)"
        print(f"  {resp_display}")
        print(f"\nAuto label: {task['auto_label']}  —  {TAXONOMY.get(task['auto_label'], '')}")

        while True:
            try:
                raw = input(f"\nYour label [1-{len(TAXONOMY_CODES)} / s / q]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                raw = "q"

            if raw == "q":
                # Save and exit
                with open(VALIDATED_FILE, "w") as f:
                    json.dump(validated, f, indent=2)
                print(f"\n✓ Saved. Confirmed={confirmed} Changed={changed}")
                return

            if raw == "s":
                # Keep auto label
                validated[model_key][task_key] = {
                    "auto_label":   task["auto_label"],
                    "human_label":  task["auto_label"],
                    "human_note":   "skip",
                    "category":     task["category"],
                    "difficulty":   task["difficulty"],
                    "condition":    task["condition"],
                }
                confirmed += 1
                break

            if raw.isdigit() and 1 <= int(raw) <= len(TAXONOMY_CODES):
                chosen = TAXONOMY_CODES[int(raw) - 1]
                note   = ""
                if chosen != task["auto_label"]:
                    note = input("  Optional note on why you changed it: ").strip()
                    changed += 1
                else:
                    confirmed += 1

                validated[model_key][task_key] = {
                    "auto_label":  task["auto_label"],
                    "human_label": chosen,
                    "human_note":  note,
                    "category":    task["category"],
                    "difficulty":  task["difficulty"],
                    "condition":   task["condition"],
                }
                break
            else:
                print("  Invalid input. Enter 1-6, s, or q.")

    # Save at end
    with open(VALIDATED_FILE, "w") as f:
        json.dump(validated, f, indent=2)

    agree_rate = confirmed / (confirmed + changed) * 100 if (confirmed + changed) > 0 else 0
    print(f"\n{'='*70}")
    print(f"Session complete for {label}")
    print(f"  Confirmed auto label : {confirmed}")
    print(f"  Changed by human     : {changed}")
    print(f"  Agreement rate       : {agree_rate:.1f}%")
    print(f"  Saved to             : {VALIDATED_FILE}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────
# STEP 3 — FIGURE 6
# ─────────────────────────────────────────────────────────────

def run_figure():
    """Generate Figure 6: stacked bar chart of failure types per model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Load auto results (use human labels where available)
    if not os.path.exists(TAXONOMY_FILE):
        print(f"Error: {TAXONOMY_FILE} not found. Run --auto first.")
        return

    with open(TAXONOMY_FILE) as f:
        auto_data = json.load(f)

    human_data = {}
    if os.path.exists(VALIDATED_FILE):
        with open(VALIDATED_FILE) as f:
            human_data = json.load(f)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 300,
    })

    MODELS_ORDER = ['groq', 'openai', 'deepseek', 'deepseek_r1', 'qwen', 'gemini']
    MODEL_LABELS_SHORT = ['LLaMA\n3.1 8B', 'GPT-4o\nmini', 'DeepSeek\nV3',
                          'DeepSeek\nR1', 'Qwen3\n32B', 'Gemini\n2.5F']

    TYPE_COLORS = {
        'WRONG_REASONING':    '#e74c3c',
        'RIGHT_REASON_WRONG': '#e67e22',
        'DISTRACTOR_CAPTURE': '#f39c12',
        'OVERCONFIDENT_FLIP': '#9b59b6',
        'GUESSING':           '#3498db',
        'FORMAT_FAILURE':     '#95a5a6',
        'EXTRACTION_ERROR':   '#2ecc71',
    }
    TYPE_LABELS = {
        'WRONG_REASONING':    'Wrong reasoning',
        'RIGHT_REASON_WRONG': 'Right reason → wrong answer',
        'DISTRACTOR_CAPTURE': 'Distractor capture',
        'OVERCONFIDENT_FLIP': 'Overconfident flip',
        'GUESSING':           'Guessing',
        'FORMAT_FAILURE':     'Format failure',
        'EXTRACTION_ERROR':   'Extraction error (benchmark artifact)',
    }

    # Build counts per model — prefer human labels
    model_counts = {}
    for model_key in MODELS_ORDER:
        failures = auto_data["models"].get(model_key, [])
        if not failures:
            model_counts[model_key] = Counter()
            continue

        counts = Counter()
        human_labels = human_data.get(model_key, {})

        for t in failures:
            task_key = t["id"] + t["condition"]
            if task_key in human_labels:
                label = human_labels[task_key]["human_label"]
            else:
                label = t.get("auto_label") or "FORMAT_FAILURE"
            counts[label] += 1

        model_counts[model_key] = counts

    # Normalise to percentages
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5.5),
                                          gridspec_kw={'wspace': 0.35})

    x = np.arange(len(MODELS_ORDER))
    bar_w = 0.55

    for ax, normalise in [(ax_abs, False), (ax_pct, True)]:
        bottoms = np.zeros(len(MODELS_ORDER))

        for code in TAXONOMY_CODES:
            vals = []
            for mk in MODELS_ORDER:
                counts = model_counts[mk]
                total  = sum(counts.values()) or 1
                raw    = counts.get(code, 0)
                vals.append((raw / total * 100) if normalise else raw)

            vals = np.array(vals)
            bars = ax.bar(x, vals, bar_w, bottom=bottoms,
                          color=TYPE_COLORS[code], label=TYPE_LABELS[code])

            # Label bars that are wide enough
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > (3 if normalise else 5):
                    ax.text(xi, b + v/2, f'{v:.0f}{"%" if normalise else ""}',
                            ha='center', va='center', fontsize=6.8,
                            color='white', fontweight='600')
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS_SHORT, fontsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)

        if normalise:
            ax.set_ylabel('Failure type distribution (%)')
            ax.set_title('Normalised failure type distribution\n(% of all failures per model)',
                         fontsize=10.5)
            ax.set_ylim(0, 108)
        else:
            ax.set_ylabel('Number of failures')
            ax.set_title('Raw failure counts by type\n(absolute number per model)',
                         fontsize=10.5)

    # Shared legend
    handles = [plt.Rectangle((0,0),1,1, color=TYPE_COLORS[c], label=TYPE_LABELS[c])
               for c in TAXONOMY_CODES]
    fig.legend(handles, [TYPE_LABELS[c] for c in TAXONOMY_CODES],
               loc='lower center', ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Error taxonomy: failure mode distribution across 6 models\n'
                 '(auto-classified by Claude, human-validated sample)',
                 fontsize=11.5, y=1.03)

    plt.savefig('fig6_error_taxonomy.pdf', bbox_inches='tight')
    plt.savefig('fig6_error_taxonomy.png', bbox_inches='tight')
    plt.close()
    print("Figure 6 saved: fig6_error_taxonomy.pdf / fig6_error_taxonomy.png")


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────

def run_stats():
    """Print agreement stats between auto and human labels."""
    if not os.path.exists(VALIDATED_FILE):
        print(f"No validated file found: {VALIDATED_FILE}")
        return

    with open(VALIDATED_FILE) as f:
        validated = json.load(f)

    print(f"\n{'='*65}")
    print("FACET Taxonomy — Human Validation Stats")
    print(f"{'='*65}")

    total_agree = 0
    total_all   = 0
    confusion   = Counter()

    for model_key, reviews in validated.items():
        label = MODEL_LABELS.get(model_key, model_key)
        n     = len(reviews)
        skip  = sum(1 for v in reviews.values() if v.get("human_note") == "skip")
        # Confirmed = auto==human (excluding skips) + skips (implicitly confirmed)
        confirmed = sum(1 for v in reviews.values()
                        if v["auto_label"] == v["human_label"]
                        and v.get("human_note") != "skip")
        agree = confirmed + skip
        rate  = agree / max(n, 1) * 100

        print(f"\n  {label}")
        print(f"    Reviewed : {n}  |  Skipped (confirmed): {skip}  |  Agreement: {rate:.1f}%")

        changed = [(v["auto_label"], v["human_label"])
                   for v in reviews.values()
                   if v["auto_label"] != v["human_label"] and v.get("human_note") != "skip"]
        if changed:
            print(f"    Changes  : {len(changed)}")
            for auto, human in changed[:5]:
                print(f"      {auto} → {human}")
            if len(changed) > 5:
                print(f"      ... and {len(changed)-5} more")

        for v in reviews.values():
            total_all += 1
            if v["auto_label"] == v["human_label"] or v.get("human_note") == "skip":
                total_agree += 1
            else:
                confusion[(v["auto_label"], v["human_label"])] += 1

    overall = total_agree / total_all * 100 if total_all > 0 else 0
    print(f"\n  Overall agreement: {total_agree}/{total_all} = {overall:.1f}%")
    print(f"  (This is your inter-rater reliability for the paper)")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACET Error Taxonomy")
    parser.add_argument("--auto",   action="store_true",
                        help="Run automated classification on all results files")
    parser.add_argument("--review", action="store_true",
                        help="Interactive manual validation for one model")
    parser.add_argument("--figure", action="store_true",
                        help="Generate Figure 6")
    parser.add_argument("--stats",  action="store_true",
                        help="Print human-auto agreement statistics")
    parser.add_argument("--model",  type=str, default=None,
                        help="Model key for --review (groq/openai/deepseek/deepseek_r1/qwen/gemini)")
    parser.add_argument("--key",    type=str, default=None,
                        help="Anthropic API key for --auto")
    args = parser.parse_args()

    if args.auto:
        api_key = args.key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: No Anthropic API key. Use --key YOUR_KEY or set ANTHROPIC_API_KEY.")
            sys.exit(1)
        run_auto_classification(api_key)

    elif args.review:
        if not args.model:
            print("Error: --review requires --model (groq/openai/deepseek/deepseek_r1/qwen/gemini)")
            sys.exit(1)
        run_manual_review(args.model)

    elif args.figure:
        run_figure()

    elif args.stats:
        run_stats()

    else:
        print("FACET Error Taxonomy\n")
        print("Workflow:")
        print("  1. python facet_taxonomy.py --auto --key YOUR_ANTHROPIC_KEY")
        print("     → Classifies all failures, saves facet_taxonomy_auto.json")
        print()
        print("  2. python facet_taxonomy.py --review --model groq")
        print("     python facet_taxonomy.py --review --model openai")
        print("     ... (repeat for each model, ~50 reviews each)")
        print("     → Saves facet_taxonomy_validated.json")
        print()
        print("  3. python facet_taxonomy.py --stats")
        print("     → Prints agreement rate (inter-rater reliability for paper)")
        print()
        print("  4. python facet_taxonomy.py --figure")
        print("     → Generates fig6_error_taxonomy.pdf / .png")
        print()
        print("Taxonomy codes:")
        for code, desc in TAXONOMY.items():
            print(f"  {code:<25} {desc}")