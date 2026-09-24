"""
FACET — Results Averaging Script
==================================
Loads multiple result JSON files, averages accuracy across seeds,
and produces a clean final results table.

Usage:
    # Average 3 seeds for one model:
    python facet_average.py --files facet_results_groq_seed42.json facet_results_groq_seed99.json facet_results_groq_seed7.json

    # Average and compare two models (each averaged across 3 seeds):
    python facet_average.py --model1 facet_results_groq_seed42.json facet_results_groq_seed99.json facet_results_groq_seed7.json --model2 facet_results_openai_seed42.json facet_results_openai_seed99.json facet_results_openai_seed7.json

    # Print comparison table of all averaged models:
    python facet_average.py --compare facet_results_groq_seed*.json -- facet_results_openai_seed*.json
"""

import json
import argparse
import os
import glob
from collections import defaultdict

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


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


def safe_condition(r, cond):
    """Safely extract a condition dict, ensuring correct key exists.
    Returns empty dict if condition is not present — signals to compute_stats
    that this condition was not evaluated in this file."""
    d = r.get(cond, {})
    if not isinstance(d, dict) or not d:
        return {}  # Empty — condition not run
    if "correct" not in d:
        extracted = d.get("extracted_answer")
        correct_ans = r.get("correct_answer", "")
        d["correct"] = (extracted == correct_ans) if extracted else False
    return d


def merge_zs_and_cot(original_files, cot_only_files):
    """
    Merge zero-shot results from original files with CoT results from cot_only files.
    Returns a combined results list ready for compute_stats.
    """
    # Load and combine original ZS results (one entry per seed per task)
    # We need to group by seed to maintain 3-seed averaging
    all_original = []
    for f in original_files:
        data = load_results(f)
        all_original.append(data["results"])

    # Load and combine cot_only results
    all_cot = []
    for f in cot_only_files:
        data = load_results(f)
        all_cot.append(data["results"])

    # Build merged list: one entry per (seed, task)
    merged = []
    for seed_idx, zs_results in enumerate(all_original):
        # Get corresponding cot results for this seed if available
        cot_results = all_cot[seed_idx] if seed_idx < len(all_cot) else []
        cot_by_id = {r["id"]: r for r in cot_results}

        for r in zs_results:
            merged_r = {
                "id": r["id"],
                "category": r["category"],
                "difficulty": r["difficulty"],
                "num_steps": r.get("num_steps", 0),
                "correct_answer": r.get("correct_answer", ""),
                "zero_shot": safe_condition(r, "zero_shot"),
            }
            # Use CoT from cot_only file if available
            if r["id"] in cot_by_id:
                merged_r["cot"] = safe_condition(cot_by_id[r["id"]], "cot")
            else:
                merged_r["cot"] = safe_condition(r, "cot")
            merged.append(merged_r)

    return merged


def compute_stats(results_list):
    """
    Given a list of result dicts (one per seed run),
    compute averaged accuracy per category/difficulty/condition.
    Only includes a condition if it has valid data (not empty dict).
    Returns nested dict: stats[category][difficulty][condition] = avg_accuracy
    """
    all_stats = defaultdict(lambda: defaultdict(lambda: {
        "zero_shot": [], "cot": []
    }))

    for run in results_list:
        for r in run["results"]:
            cat  = r["category"]
            diff = r["difficulty"]
            for cond in ["zero_shot", "cot"]:
                if cond not in r:
                    continue
                cond_data = r[cond]
                # Skip empty dicts — condition was not run in this file
                if not isinstance(cond_data, dict) or not cond_data:
                    continue
                if "correct" in cond_data:
                    all_stats[cat][diff][cond].append(cond_data["correct"])
                elif "extracted_answer" in cond_data:
                    extracted = cond_data.get("extracted_answer")
                    correct_ans = r.get("correct_answer", "")
                    correct = (extracted == correct_ans) if extracted else False
                    all_stats[cat][diff][cond].append(correct)

    # Average
    avg_stats = defaultdict(lambda: defaultdict(dict))
    for cat in all_stats:
        for diff in all_stats[cat]:
            for cond in ["zero_shot", "cot"]:
                vals = all_stats[cat][diff][cond]
                avg_stats[cat][diff][cond] = sum(vals) / len(vals) if vals else 0.0

    return avg_stats


def pct(val):
    return f"{val * 100:5.1f}%"


def print_results_table(stats, model_name, n_seeds):
    print(f"\n{'='*68}")
    print(f"FACET Averaged Results — {model_name} ({n_seeds} seeds)")
    print(f"{'='*68}")
    print(f"{'Category':<28} {'Diff':<8} {'Zero-shot':>10} {'CoT':>10} {'CoT gain':>10}")
    print(f"{'─'*68}")

    overall_zs, overall_cot = [], []
    for cat in CATEGORIES:
        for diff in ["easy", "medium", "hard"]:
            zs  = stats[cat][diff].get("zero_shot", 0)
            cot = stats[cat][diff].get("cot", 0)
            overall_zs.append(zs)
            overall_cot.append(cot)
            gain = cot - zs
            sign = "+" if gain >= 0 else ""
            label = cat.replace("_", " ").title()[:27]
            print(f"{label:<28} {diff:<8} {pct(zs):>10} {pct(cot):>10} {sign}{gain*100:5.1f}%")
        print(f"{'─'*68}")

    overall_zs_avg = sum(overall_zs) / len(overall_zs)
    overall_cot_avg = sum(overall_cot) / len(overall_cot)
    gain = overall_cot_avg - overall_zs_avg
    sign = "+" if gain >= 0 else ""
    print(f"{'OVERALL':<28} {'':8} {pct(overall_zs_avg):>10} {pct(overall_cot_avg):>10} {sign}{gain*100:5.1f}%")
    print(f"{'='*68}")

    # Difficulty ordering check
    print(f"\nDifficulty ordering (zero-shot, should be easy > medium > hard):")
    print(f"{'─'*55}")
    issues = 0
    for cat in CATEGORIES:
        e = stats[cat]["easy"].get("zero_shot", 0) * 100
        m = stats[cat]["medium"].get("zero_shot", 0) * 100
        h = stats[cat]["hard"].get("zero_shot", 0) * 100
        ok = (e >= m >= h)
        status = "✓ OK" if ok else "✗ ISSUE"
        if not ok:
            issues += 1
        label = cat.replace("_", " ").title()[:28]
        print(f"  {label:<30} easy:{e:.0f}% mid:{m:.0f}% hard:{h:.0f}%  {status}")
    print(f"\n  {issues} ordering issue(s) found out of {len(CATEGORIES)} categories")

    # CoT gain summary
    print(f"\nCoT gain per category (averaged across seeds):")
    print(f"{'─'*55}")
    for cat in CATEGORIES:
        zs_all  = [stats[cat][d].get("zero_shot", 0) for d in ["easy","medium","hard"]]
        cot_all = [stats[cat][d].get("cot", 0) for d in ["easy","medium","hard"]]
        zs_avg  = sum(zs_all) / 3
        cot_avg = sum(cot_all) / 3
        gain = (cot_avg - zs_avg) * 100
        sign = "+" if gain >= 0 else ""
        label = cat.replace("_", " ").title()[:28]
        print(f"  {label:<30} ZS:{zs_avg*100:.1f}%  CoT:{cot_avg*100:.1f}%  Gain:{sign}{gain:.1f}%")

    print(f"\n{'='*68}")


def print_comparison_table(stats_dict):
    """Print side-by-side zero-shot comparison of multiple models."""
    models = list(stats_dict.keys())

    print(f"\n{'='*80}")
    print(f"FACET Cross-Model Comparison (Zero-shot)")
    print(f"{'='*80}")

    # Header
    header = f"{'Category':<28} {'Diff':<8}"
    for m in models:
        header += f" {m[:12]:>12}"
    print(header)
    print(f"{'─'*80}")

    overall = {m: [] for m in models}
    for cat in CATEGORIES:
        for diff in ["easy", "medium", "hard"]:
            row = f"{cat.replace('_',' ').title()[:27]:<28} {diff:<8}"
            for m in models:
                val = stats_dict[m][cat][diff].get("zero_shot", 0) * 100
                overall[m].append(val)
                row += f" {val:>11.1f}%"
            print(row)
        print(f"{'─'*80}")

    row = f"{'OVERALL':<28} {'':8}"
    for m in models:
        avg = sum(overall[m]) / len(overall[m])
        row += f" {avg:>11.1f}%"
    print(row)
    print(f"{'='*80}")

    # CoT gain comparison
    print(f"\nCoT Gain Comparison:")
    print(f"{'─'*80}")
    header = f"{'Category':<28} {'':8}"
    for m in models:
        header += f" {m[:12]:>12}"
    print(header)
    print(f"{'─'*80}")

    for cat in CATEGORIES:
        zs_vals  = {m: sum(stats_dict[m][cat][d].get("zero_shot",0) for d in ["easy","medium","hard"])/3 for m in models}
        cot_vals = {m: sum(stats_dict[m][cat][d].get("cot",0) for d in ["easy","medium","hard"])/3 for m in models}
        row = f"{cat.replace('_',' ').title()[:27]:<28} {'':8}"
        for m in models:
            gain = (cot_vals[m] - zs_vals[m]) * 100
            sign = "+" if gain >= 0 else ""
            row += f" {sign}{gain:>10.1f}%"
        print(row)
    print(f"{'='*80}")


def save_averaged_json(stats, model_name, n_seeds, output_file):
    """Save averaged results as JSON for further analysis."""
    output = {
        "model": model_name,
        "n_seeds": n_seeds,
        "averaged_results": {}
    }
    for cat in CATEGORIES:
        output["averaged_results"][cat] = {}
        for diff in ["easy", "medium", "hard"]:
            output["averaged_results"][cat][diff] = {
                "zero_shot": round(stats[cat][diff].get("zero_shot", 0) * 100, 2),
                "cot": round(stats[cat][diff].get("cot", 0) * 100, 2),
                "cot_gain": round((stats[cat][diff].get("cot", 0) - stats[cat][diff].get("zero_shot", 0)) * 100, 2)
            }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAveraged results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACET Results Averaging Script")
    parser.add_argument("--files", nargs="+",
                        help="Result JSON files to average (single model)")
    parser.add_argument("--cot_only", nargs="+",
                        help="CoT-only result files to merge with --files (single model mode)")
    parser.add_argument("--model1", nargs="+",
                        help="Result files for model 1 (comparison mode)")
    parser.add_argument("--cot_only1", nargs="+",
                        help="CoT-only files for model 1")
    parser.add_argument("--model2", nargs="+",
                        help="Result files for model 2 (comparison mode)")
    parser.add_argument("--cot_only2", nargs="+",
                        help="CoT-only files for model 2")
    parser.add_argument("--model3", nargs="+",
                        help="Result files for model 3 (optional)")
    parser.add_argument("--cot_only3", nargs="+",
                        help="CoT-only files for model 3")
    parser.add_argument("--model4", nargs="+",
                        help="Result files for model 4 (optional)")
    parser.add_argument("--cot_only4", nargs="+",
                        help="CoT-only files for model 4")
    parser.add_argument("--model5", nargs="+",
                        help="Result files for model 5 (optional)")
    parser.add_argument("--cot_only5", nargs="+",
                        help="CoT-only files for model 5")
    parser.add_argument("--model6", nargs="+",
                        help="Result files for model 6 (optional)")
    parser.add_argument("--cot_only6", nargs="+",
                        help="CoT-only files for model 6")
    parser.add_argument("--save", type=str, default=None,
                        help="Save averaged results to this JSON file")
    args = parser.parse_args()

    if args.files:
        # Single model averaging
        files = []
        for pattern in args.files:
            files.extend(glob.glob(pattern))
        files = sorted(set(files))

        if not files:
            print(f"No files found matching: {args.files}")
            exit(1)

        model_name = load_results(files[0])["model"]
        print(f"Loading {len(files)} result files for {model_name}:")
        for f in files:
            print(f"  {f}")

        if args.cot_only:
            # Merge mode: ZS from files, CoT from cot_only files
            cot_files = []
            for pattern in args.cot_only:
                cot_files.extend(glob.glob(pattern))
            cot_files = sorted(set(cot_files))
            print(f"Merging with {len(cot_files)} CoT-only files:")
            for f in cot_files:
                print(f"  {f}")
            merged = merge_zs_and_cot(files, cot_files)
            stats = compute_stats([{"results": merged}])
            print_results_table(stats, model_name, len(files))
        else:
            results_list = []
            for f in files:
                data = load_results(f)
                results_list.append(data)
            stats = compute_stats(results_list)
            print_results_table(stats, model_name, len(files))

        if args.save:
            save_averaged_json(stats, model_name, len(files), args.save)

    elif args.model1:
        # Multi-model comparison
        model_groups = {}
        for i, attr in enumerate(["model1", "model2", "model3", "model4", "model5", "model6"], 1):
            file_patterns = getattr(args, attr, None)
            if not file_patterns:
                continue
            files = []
            for pattern in file_patterns:
                files.extend(glob.glob(pattern))
            files = sorted(set(files))
            if not files:
                continue

            model_name = load_results(files[0])["model"]

            # Check if cot_only files provided for this model
            cot_attr = f"cot_only{i}"
            cot_patterns = getattr(args, cot_attr, None)
            if cot_patterns:
                cot_files = []
                for pattern in cot_patterns:
                    cot_files.extend(glob.glob(pattern))
                cot_files = sorted(set(cot_files))
                if cot_files:
                    print(f"\nMerging ZS + CoT-only for {model_name}...")
                    merged = merge_zs_and_cot(files, cot_files)
                    stats = compute_stats([{"results": merged}])
                    print_results_table(stats, model_name, len(files))
                    model_groups[model_name] = stats
                    continue

            results_list = []
            for f in files:
                data = load_results(f)
                results_list.append(data)
            stats = compute_stats(results_list)
            model_groups[model_name] = stats
            print_results_table(stats, model_name, len(files))

        if len(model_groups) > 1:
            print_comparison_table(model_groups)

        if args.save:
            combined = {"models": {}}
            for m, s in model_groups.items():
                combined["models"][m] = {}
                for cat in CATEGORIES:
                    combined["models"][m][cat] = {}
                    for diff in ["easy","medium","hard"]:
                        combined["models"][m][cat][diff] = {
                            "zero_shot": round(s[cat][diff].get("zero_shot",0)*100, 2),
                            "cot": round(s[cat][diff].get("cot",0)*100, 2),
                        }
            with open(args.save, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"\nCombined averaged results saved to: {args.save}")

    else:
        print("FACET Results Averaging Script")
        print()
        print("Average 3 seeds for one model:")
        print("  python facet_average.py --files facet_results_groq_seed42.json facet_results_groq_seed99.json facet_results_groq_seed7.json")
        print()
        print("Compare two models (each averaged across 3 seeds):")
        print("  python facet_average.py \\")
        print("    --model1 facet_results_groq_seed42.json facet_results_groq_seed99.json facet_results_groq_seed7.json \\")
        print("    --model2 facet_results_openai_seed42.json facet_results_openai_seed99.json facet_results_openai_seed7.json")
        print()
        print("Save averaged results to JSON:")
        print("  python facet_average.py --files facet_results_groq_seed*.json --save facet_avg_groq.json")