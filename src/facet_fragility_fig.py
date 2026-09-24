"""
FACET — Figure 4: Robustness Fragility Heatmap
===============================================
Generates a publication-quality figure showing fragility rates
(% of originally-correct answers broken by perturbation) across:
  - 6 models (rows)
  - 8 categories (columns)
  - 3 perturbation types (subplots / panel layout)

Output: fig4_fragility.pdf, fig4_fragility.png

Data is hardcoded from facet_perturbation results
(paste new values here when re-running models).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Hardcoded fragility data from perturbation eval results ───────────────
# Shape: [model][perturbation_type][category] = fragility_rate (0–100)
# Order matches MODELS / CATS lists below

#          Reversal  Compos.  Syllog.  WorkMem  Inhibit  Counting  Anchoring  ToM
FRAGILITY = {
    'groq': {
        'surface':    [ 0.0,    5.5,    0.0,    13.8,   22.4,    0.0,    42.5,    0.0],
        'distractor': [ 3.6,    4.8,    6.7,     5.5,   12.9,   33.3,     6.9,   11.6],
        'reframe':    [ 8.4,    8.3,    8.3,    10.1,   14.1,   27.8,    36.8,   12.8],
    },
    'openai': {
        'surface':    [ 0.0,    7.0,    0.9,    14.0,   16.8,    5.8,    11.6,    4.1],
        'distractor': [ 0.8,    2.3,    5.7,    16.7,    6.3,   14.5,     6.3,    2.7],
        'reframe':    [ 2.5,    2.3,    2.8,     6.1,    4.2,    5.8,     5.3,    2.1],
    },
    'deepseek': {
        'surface':    [ 0.0,    2.0,    0.0,     4.0,   49.3,    3.9,    47.9,    5.7],
        'distractor': [ 2.4,    0.7,    6.3,     4.0,    4.8,    7.0,     4.1,    8.6],
        'reframe':    [ 6.5,    2.0,    4.8,     4.0,   15.1,   10.9,     7.5,    6.4],
    },
    'deepseek_r1': {
        'surface':    [ 0.7,    3.3,    0.8,     9.4,   68.7,    0.7,    77.6,   12.7],
        'distractor': [ 0.0,    0.7,    0.0,     0.0,    4.7,    0.7,     4.2,    4.2],
        'reframe':    [ 0.7,    0.0,    0.0,     0.0,    4.0,    2.7,     2.8,    2.8],
    },
    'qwen': {
        'surface':    [11.4,   12.9,   23.4,    33.7,   49.5,    9.8,    69.0,   10.5],
        'distractor': [14.3,   12.9,   34.4,    36.0,   18.3,   10.7,    11.0,   12.8],
        'reframe':    [17.1,    7.3,   39.1,    24.7,   26.6,   13.4,    12.0,    7.5],
    },
    'gemini': {
        'surface':    [ 0.0,    9.2,    1.5,     7.3,   33.6,    4.1,    75.0,    7.4],
        'distractor': [ 1.3,    9.2,   10.4,     0.0,    8.6,    7.6,     5.6,    0.7],
        'reframe':    [ 0.7,    0.8,    1.5,     0.0,    3.4,    7.6,     4.2,    0.0],
    },
}

MODELS      = ['groq', 'openai', 'deepseek', 'deepseek_r1', 'qwen', 'gemini']
MODEL_LABELS = ['LLaMA 3.1 8B', 'GPT-4o-mini', 'DeepSeek-V3',
                'DeepSeek-R1', 'Qwen3-32B', 'Gemini 2.5 Flash']

CATS = ['reversal_curse', 'compositional_reasoning', 'syllogistic_reasoning',
        'working_memory', 'inhibitory_control', 'counting',
        'anchoring_bias', 'theory_of_mind']
CAT_LABELS_SHORT = ['Reversal', 'Compositional', 'Syllogistic', 'Working Mem.',
                    'Inhibitory', 'Counting', 'Anchoring', 'Theory of Mind']

PTYPES       = ['surface', 'distractor', 'reframe']
PTYPE_TITLES = ['Surface\n(entity/number swap)', 'Distractor\n(irrelevant sentence)', 'Reframe\n(scenario wrap)']

# Paper style — matches facet_figures.py
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':    'tight',
    'savefig.pad_inches': 0.1,
})

# ── Colormap: white (0%) → amber → red (100%) ─────────────────────────────
FRAG_CMAP = LinearSegmentedColormap.from_list(
    'fragility', ['#f7f7f7', '#fdd49e', '#fc8d59', '#d7301f', '#7f0000'], N=256
)

# ── Build figure: 1 row of 3 subplots ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5),
                         gridspec_kw={'wspace': 0.08})

vmin, vmax = 0, 80   # cap at 80% so colour range is meaningful

for ax_idx, (ptype, ptitle) in enumerate(zip(PTYPES, PTYPE_TITLES)):
    ax = axes[ax_idx]

    # Build matrix: rows = models, cols = categories
    mat = np.array([FRAGILITY[m][ptype] for m in MODELS])

    im = ax.imshow(mat, cmap=FRAG_CMAP, vmin=vmin, vmax=vmax, aspect='auto')

    # Annotate every cell
    for i in range(len(MODELS)):
        for j in range(len(CATS)):
            val = mat[i, j]
            # White text on dark cells, dark on light
            text_color = 'white' if val > 45 else '#222222'
            ax.text(j, i, f'{val:.0f}%',
                    ha='center', va='center',
                    fontsize=9.5, color=text_color, fontweight='500')

    # Axes labels
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CAT_LABELS_SHORT, rotation=32, ha='right', fontsize=10)
    ax.set_yticks(range(len(MODELS)))

    if ax_idx == 0:
        ax.set_yticklabels(MODEL_LABELS, fontsize=11)
    else:
        ax.set_yticklabels([])

    ax.set_title(ptitle, fontsize=12, fontweight='600', pad=8)

    # Minor grid lines
    ax.set_xticks(np.arange(-0.5, len(CATS),   1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.4)
    ax.tick_params(which='minor', bottom=False, left=False)

# ── Shared colourbar ──────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.18, 0.013, 0.64])
sm = plt.cm.ScalarMappable(cmap=FRAG_CMAP,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Fragility rate (%)', fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([0, 20, 40, 60, 80])
cbar.set_ticklabels(['0%', '20%', '40%', '60%', '≥80%'])

# ── Overall title ─────────────────────────────────────────────────────────
fig.suptitle(
    'Robustness fragility: % of originally-correct answers broken by semantics-preserving perturbation',
    fontsize=13, y=1.03
)

# ── Row average annotations (right margin) ────────────────────────────────
# Compute average fragility per model across all cats & ptypes
for i, m in enumerate(MODELS):
    all_vals = [v for pt in PTYPES for v in FRAGILITY[m][pt]]
    avg = np.mean(all_vals)
    axes[2].text(len(CATS) - 0.45, i, f'  avg\n  {avg:.1f}%',
                 ha='left', va='center', fontsize=9, color='#555555',
                 transform=axes[2].transData)

plt.savefig('fig4_fragility.pdf', bbox_inches='tight')
plt.savefig('fig4_fragility.png', bbox_inches='tight')
plt.close()
print('Figure 4 saved: fig4_fragility.pdf / fig4_fragility.png')

# ── Print cross-model summary to terminal ────────────────────────────────
print('\n── Cross-model fragility summary (avg across categories) ──')
print(f'{"Model":<20} {"Surface":>8} {"Distractor":>11} {"Reframe":>8} {"Overall":>8}')
print('─' * 60)
for m, ml in zip(MODELS, MODEL_LABELS):
    s  = np.mean(FRAGILITY[m]['surface'])
    d  = np.mean(FRAGILITY[m]['distractor'])
    r  = np.mean(FRAGILITY[m]['reframe'])
    ov = np.mean([s, d, r])
    print(f'{ml:<20} {s:7.1f}%  {d:9.1f}%  {r:7.1f}%  {ov:7.1f}%')

print('\n── Most fragile (category, perturbation) pairs ──')
hits = []
for m in MODELS:
    for pi, pt in enumerate(PTYPES):
        for ci, cat in enumerate(CATS):
            hits.append((FRAGILITY[m][pt][ci], m, pt, cat))
hits.sort(reverse=True)
for val, m, pt, cat in hits[:10]:
    ml = MODEL_LABELS[MODELS.index(m)]
    cl = CAT_LABELS_SHORT[CATS.index(cat)]
    print(f'  {val:5.1f}%  {ml:<20} {pt:<12} {cl}')