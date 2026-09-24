"""
FACET — Analysis Figures
Generates three publication-quality figures:
  1. CoT Selectivity Heatmap (6 models × 8 categories)
  2. Difficulty Scaling Curves (zero-shot accuracy vs difficulty)
  3. Cross-Category Failure Correlation Matrix
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────
with open('facet_final_results.json') as f:
    raw = json.load(f)['models']

MODELS = ['groq', 'openai', 'deepseek', 'deepseek_r1', 'qwen', 'gemini']
MODEL_LABELS = ['LLaMA 3.1 8B', 'GPT-4o-mini', 'DeepSeek-V3',
                'DeepSeek-R1', 'Qwen3-32B', 'Gemini 2.5 Flash']

CATS = ['reversal_curse', 'compositional_reasoning', 'syllogistic_reasoning',
        'working_memory', 'inhibitory_control', 'counting',
        'anchoring_bias', 'theory_of_mind']
CAT_LABELS = ['Reversal\nCurse', 'Compositional\nReasoning', 'Syllogistic\nReasoning',
              'Working\nMemory', 'Inhibitory\nControl', 'Counting',
              'Anchoring\nBias', 'Theory of\nMind']
CAT_LABELS_SHORT = ['Reversal', 'Compositional', 'Syllogistic', 'Working Mem.',
                    'Inhibitory', 'Counting', 'Anchoring', 'Theory of Mind']

DIFFS = ['easy', 'medium', 'hard']

# Paper style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# ── Helper: average across difficulty levels ───────────────────────────────
def avg_zs(model, cat):
    vals = [raw[model][cat][d]['zero_shot'] for d in DIFFS]
    return np.mean(vals)

def avg_cot(model, cat):
    vals = [raw[model][cat][d]['cot'] for d in DIFFS]
    return np.mean(vals)

def cot_gain(model, cat):
    return avg_cot(model, cat) - avg_zs(model, cat)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CoT Selectivity Heatmap
# ══════════════════════════════════════════════════════════════════════════
def make_heatmap():
    gains = np.array([[cot_gain(m, c) for c in CATS] for m in MODELS])

    fig, ax = plt.subplots(figsize=(12, 5))

    # Diverging colormap: red (hurts) → white (neutral) → green (helps)
    cmap = LinearSegmentedColormap.from_list(
        'rg', ['#c0392b', '#f4a58a', '#fef9f9', '#9ecfb3', '#2d7a4f'], N=256)

    im = ax.imshow(gains, cmap=cmap, vmin=-50, vmax=50, aspect='auto')

    # Cell annotations
    for i, m in enumerate(MODELS):
        for j, c in enumerate(CATS):
            g = gains[i, j]
            color = 'white' if abs(g) > 28 else '#333333'
            sign = '+' if g >= 0 else ''
            ax.text(j, i, f'{sign}{g:.1f}%', ha='center', va='center',
                    fontsize=10, color=color, fontweight='500')

    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CAT_LABELS_SHORT, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODEL_LABELS, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('CoT gain (percentage points)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title('CoT Selectivity: gain/loss from chain-of-thought prompting per model and failure type',
                 fontsize=12, pad=10)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(CATS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    plt.savefig('fig1_heatmap.pdf')
    plt.savefig('fig1_heatmap.png')
    plt.close()
    print('Figure 1 saved.')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Difficulty Scaling Curves
# ══════════════════════════════════════════════════════════════════════════
def make_difficulty_curves():
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#1abc9c', '#9b59b6', '#3498db']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    linestyles = ['-', '--', '-', '--', '-', '--']
    x = [0, 1, 2]
    x_labels = ['Easy', 'Medium', 'Hard']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes = axes.flatten()

    for j, (cat, cat_label) in enumerate(zip(CATS, CAT_LABELS_SHORT)):
        ax = axes[j]
        for i, (m, ml) in enumerate(zip(MODELS, MODEL_LABELS)):
            ys = [raw[m][cat][d]['zero_shot'] for d in DIFFS]
            ax.plot(x, ys, color=colors[i], marker=markers[i],
                    linestyle=linestyles[i], linewidth=1.8,
                    markersize=6, label=ml)
        ax.set_title(cat_label, fontsize=12, fontweight='500')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_tick_params(labelsize=10)
        ax.axhline(25, color='#cccccc', linewidth=0.8, linestyle=':')
        ax.grid(axis='y', alpha=0.3, linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared y-label
    fig.text(0.01, 0.55, 'Zero-shot accuracy (%)', va='center',
             rotation='vertical', fontsize=12)

    # Legend below
    handles = [plt.Line2D([0], [0], color=colors[i], marker=markers[i],
                          linestyle=linestyles[i], linewidth=1.8, markersize=6)
               for i in range(len(MODELS))]
    fig.legend(handles, MODEL_LABELS, loc='lower center',
               ncol=6, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Difficulty scaling: zero-shot accuracy across easy / medium / hard tasks',
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0.03, 0.05, 1, 1])
    plt.savefig('fig2_difficulty_curves.pdf', bbox_inches='tight')
    plt.savefig('fig2_difficulty_curves.png', bbox_inches='tight')
    plt.close()
    print('Figure 2 saved.')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cross-Category Failure Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════
def make_correlation_matrix():
    # Build vector of avg zero-shot per (model, category)
    # Shape: (n_models=6) × (n_cats=8)
    # Pearson correlation across the 6 model data points between each pair of categories
    zs_matrix = np.array([[avg_zs(m, c) for c in CATS] for m in MODELS])
    # zs_matrix shape: (6, 8) — rows=models, cols=categories
    # We want correlation between categories → transpose to (8, 6) then corrcoef
    corr = np.corrcoef(zs_matrix.T)  # (8, 8)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    cmap = LinearSegmentedColormap.from_list(
        'br', ['#8b1a1a', '#f4a58a', '#f9f9f9', '#a8cfe3', '#1a5c8a'], N=256)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # show lower triangle + diagonal

    # Full matrix but style upper triangle differently
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Annotate cells
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            if i == j:
                ax.text(j, i, '1.00', ha='center', va='center',
                        fontsize=8, color='white', fontweight='500')
            elif i > j:  # lower triangle
                r = corr[i, j]
                color = 'white' if abs(r) > 0.7 else '#333333'
                ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                        fontsize=8, color=color)
            else:  # upper triangle — lighter
                r = corr[i, j]
                color = '#aaaaaa'
                ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                        fontsize=7.5, color=color)

    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CAT_LABELS_SHORT, rotation=40, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(CATS)))
    ax.set_yticklabels(CAT_LABELS_SHORT, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Pearson r', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('Cross-category failure correlation\n(Pearson r of zero-shot accuracy across 6 models)',
                 fontsize=11, pad=10)

    # Grid
    ax.set_xticks(np.arange(-0.5, len(CATS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CATS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    plt.savefig('fig3_correlation.pdf')
    plt.savefig('fig3_correlation.png')
    plt.close()
    print('Figure 3 saved.')


# ══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════
make_heatmap()
make_difficulty_curves()
make_correlation_matrix()
print('\nAll three figures generated successfully.')