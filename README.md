# Beyond Accuracy: A Multi-Dimensional Diagnostic Benchmark for LLM Reasoning Failures

Code and data for *"Beyond Accuracy: A Multi-Dimensional Diagnostic Benchmark for LLM
Reasoning Failures."* The benchmark evaluates 6 frontier models on 1,200 procedurally
generated tasks across 8 cognitively motivated reasoning categories, along four
dimensions: accuracy (zero-shot vs. CoT), robustness to surface perturbation,
confidence calibration, and error taxonomy.


## Repo layout

```
cot-paradox/
├── data/
│   └── facet_v2_dataset.json              # the 1,200-task dataset
├── src/
│   ├── facet_generator.py                 # builds the dataset
│   ├── facet_eval.py                      # zero-shot / CoT accuracy
│   ├── facet_calibration.py               # verbalized 0–100% confidence
│   ├── facet_perturbation.py              # surface / distractor / reframe robustness
│   ├── facet_average.py                   # cross-seed averaging + model comparison
│   ├── facet_taxonomy.py                  # auto-classify failures, human review, stats, Fig 6
│   ├── facet_figures.py                   # Fig 1 (CoT heatmap), Fig 2 (difficulty curves), Fig 3 (correlation)
│   ├── facet_calibration_fig.py           # Fig 5 (calibration)
│   ├── facet_fragility_fig.py             # Fig 4 (fragility heatmap)
│   ├── check_nulls.py                     # counts extraction failures per results file
│   └── check_responses.py                 # inspect raw failed responses for one file
└── paper/
    ├── *.tex
    └── paper.pdf
```

## Where result files get saved

There's no `results/` folder in this repo — none of the scripts expect one. Every
script writes its own JSON or figure straight into whatever directory you run it
from, using a hardcoded filename with no path prefix (e.g. `facet_eval.py` writes
`facet_results_{model}_seed{N}.json`, `facet_figures.py` writes `fig1_heatmap.pdf`,
etc.). So results accumulate wherever you happen to be `cd`'d into when you run
`python src/facet_eval.py ...` — the "Reproducing the pipeline" commands below all
assume you're running from a single working directory of your choice (repo root, an
empty scratch folder, whatever), and everything just lands there together. If you
want them separated from `src/` and `data/`, just `mkdir` a folder and run the
commands from inside it.

## Model keys

Scripts refer to models by short internal keys, not their public names:

| key           | model             |
|---------------|-------------------|
| `groq`        | LLaMA 3.1 8B      |
| `openai`      | GPT-4o-mini       |
| `deepseek`    | DeepSeek-V3       |
| `deepseek_r1` | DeepSeek-R1       |
| `qwen`        | Qwen3-32B         |
| `gemini`      | Gemini 2.5 Flash  |

## Setup

```bash
pip install numpy matplotlib seaborn requests openai
```

API keys, as environment variables:

```bash
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
export DEEPSEEK_API_KEY=...     # used for both deepseek and deepseek_r1
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...    # for facet_taxonomy.py --auto only
```

**Qwen key inconsistency:** `facet_eval.py` reads `HF_API_KEY` for `qwen`, but
`facet_calibration.py` and `facet_perturbation.py` both read `GROQ_API_KEY` for `qwen`
instead. Set whichever the script you're running actually expects (or set both to the
same value) — this looks like a leftover from switching Qwen's provider partway
through the project.

## Reproducing the pipeline end to end

**1. Generate the dataset**

```bash
python src/facet_generator.py --seed 42 --output data/facet_v2_dataset.json
```

Add `--validate` first to sanity-check all 8 generators before a full run.

**2. Run accuracy evaluation (zero-shot + CoT), 3 seeds per model**

```bash
for seed in 42 7 99; do
  python src/facet_eval.py --eval --model groq --seed $seed --key $GROQ_API_KEY
done
```

Repeat per model. Each run writes `facet_results_{model}_seed{N}.json` in your
current directory. If a model's CoT extraction had systematic failures (e.g.
bold-markdown answers not being parsed — see Discussion in the paper), rerun just
that condition with `--cot_only` and merge in the next step.

**3. Average across seeds**

```bash
python src/facet_average.py \
  --model1 facet_results_groq_seed*.json \
  --model2 facet_results_openai_seed*.json \
  --model3 facet_results_deepseek_seed*.json \
  --model4 facet_results_deepseek_r1_seed*.json \
  --model5 facet_results_qwen_seed*.json \
  --model6 facet_results_gemini_seed*.json \
  --save facet_final_results.json
```

If a model needed a `_cot_only` merge, pass `--cot_only1` (etc.) alongside its
`--model1` files instead.

**4. Run calibration and perturbation evaluation** (single seed each, per the paper)

```bash
python src/facet_calibration.py --eval --model groq --key $GROQ_API_KEY
python src/facet_perturbation.py --eval --model groq --key $GROQ_API_KEY
```

Repeat per model. Each writes `facet_calib_results_{model}_seed42.json` /
`facet_perturb_results_{model}_seed42.json`.

**5. Error taxonomy**

```bash
python src/facet_taxonomy.py --auto --key $ANTHROPIC_API_KEY   # classifies all failures
python src/facet_taxonomy.py --review --model groq             # ~50 manual reviews per model
python src/facet_taxonomy.py --stats                           # inter-rater agreement for the paper
python src/facet_taxonomy.py --figure                          # Fig 6
```

Note DeepSeek-R1 is excluded from taxonomy analysis (near-ceiling accuracy leaves too
few failures).

**6. Generate the remaining figures**

```bash
python src/facet_figures.py            # reads facet_final_results.json → Fig 1-3
python src/facet_fragility_fig.py       # Fig 4 (see caveat below)
python src/facet_calibration_fig.py     # Fig 5 (see caveat below)
```

## Utilities

- `check_nulls.py` — run after any eval batch to see extraction-failure counts per
  file before averaging.
- `check_responses.py <file>` — inspect the actual failed responses to debug why
  extraction is missing an answer format.
