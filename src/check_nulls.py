import json, glob

for f in sorted(glob.glob('facet_results_*.json')):
    with open(f) as fh:
        d = json.load(fh)
    zs_null = sum(1 for r in d['results']
                  if 'zero_shot' in r and r['zero_shot']['extracted_answer'] is None)
    cot_null = sum(1 for r in d['results']
                   if 'cot' in r and r['cot']['extracted_answer'] is None)
    total = len(d['results'])
    print(f"{f}: ZS_null={zs_null}/{total}, CoT_null={cot_null}/{total}")