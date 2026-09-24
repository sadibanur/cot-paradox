"""
Inspect unextracted responses to understand why extraction is failing.
Run: python check_responses.py facet_results_deepseek_r1_seed42_cot_only.json
"""
import json, sys

filename = sys.argv[1] if len(sys.argv) > 1 else "facet_results_deepseek_r1_seed42_cot_only.json"

with open(filename) as f:
    data = json.load(f)

print(f"Model: {data['model']}")
print(f"Total tasks: {data['total_tasks']}")
print()

# Find tasks where CoT extraction failed
failed = []
for r in data["results"]:
    if "cot" in r and r["cot"]["extracted_answer"] is None:
        failed.append(r)

print(f"CoT extraction failures: {len(failed)}")
print()
print("=" * 60)
print("Sample of 5 failed responses:")
print("=" * 60)

for r in failed[:5]:
    print(f"\nCategory: {r['category']} | Difficulty: {r['difficulty']}")
    print(f"Correct answer: {r['correct_answer']}")
    response = r["cot"].get("response", "")
    print(f"Response length: {len(response)} chars")
    print(f"Response (last 200 chars):")
    print(f"  ...{response[-200:]}")
    print(f"Extracted: {r['cot']['extracted_answer']}")
    print("-" * 40)