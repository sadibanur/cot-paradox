"""
FACET — Figure 5: Confidence Calibration
Two panels:
  Left:  Reliability diagram (6 models, confidence bins vs accuracy)
  Right: Summary bar chart (ECE, overconfidence bias, AUROC) per model
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

# ── Data ──────────────────────────────────────────────────────────────────
MODELS      = ['groq', 'openai', 'deepseek', 'deepseek_r1', 'qwen', 'gemini']
MODEL_LABELS = ['LLaMA 3.1 8B', 'GPT-4o-mini', 'DeepSeek-V3',
                'DeepSeek-R1', 'Qwen3-32B', 'Gemini 2.5 Flash']
COLORS      = ['#e74c3c', '#e67e22', '#2ecc71', '#1abc9c', '#9b59b6', '#3498db']
MARKERS     = ['o', 's', '^', 'D', 'v', 'P']

# Overall metrics
ACCURACY  = [64.5,  71.2,  85.8,  99.8,  95.5,  96.3]
MEAN_CONF = [89.9,  93.8,  98.9,  99.7,  94.8,  99.9]
ECE       = [31.56, 22.79, 13.16,  0.13,  2.22,  3.58]
AUROC     = [0.476, 0.555, 0.511,  0.697, 0.679, 0.594]
BIAS      = [25.4,  22.5,  13.2,  -0.1,  -0.6,   3.5]

# Calibration bins: (bin_lower, avg_conf, avg_acc, count) per model
# Only populated bins included — matches terminal output exactly
BINS = {
    'groq':        [(0,   0.0,  78.7, 47),  (80, 80.0, 64.4, 371), (90, 100.0, 63.7, 782)],
    'openai':      [(0,   0.0,   0.0,  1),  (50, 50.0,  0.0,   4), (80, 80.0, 92.3,  13), (90, 94.1, 71.3,1182)],
    'deepseek':    [(0,   0.0,   0.0, 10),  (10, 10.0,  0.0,   1), (90, 99.8, 86.6,1184)],
    'deepseek_r1': [(90, 99.7,  99.8,1134)],
    'qwen':        [(70, 73.4,  41.4, 29),  (80, 83.0, 80.0,   5), (90, 95.4, 96.9,1156)],
    'gemini':      [(80, 80.0, 100.0,  1),  (90, 99.9, 96.3,1199)],
}

# ── Figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[2, 1.1, 1.1], wspace=0.38)

ax_rel   = fig.add_subplot(gs[0])   # reliability diagram
ax_ece   = fig.add_subplot(gs[1])   # ECE + bias bar chart
ax_auroc = fig.add_subplot(gs[2])   # AUROC bar chart

# ── Panel 1: Reliability diagram ──────────────────────────────────────────
ax_rel.plot([0, 100], [0, 100], 'k--', linewidth=1.2, alpha=0.4, label='Perfect calibration')
ax_rel.fill_between([0, 100], [0, 100], [100, 100], alpha=0.04, color='red')
ax_rel.fill_between([0, 100], [0, 0],   [0, 100],   alpha=0.04, color='red')
ax_rel.text(72, 88, 'Overconfident', fontsize=9, color='#cc3333', alpha=0.7, rotation=34)
ax_rel.text(18, 52, 'Underconfident', fontsize=9, color='#cc3333', alpha=0.7, rotation=34)

for i, (m, ml) in enumerate(zip(MODELS, MODEL_LABELS)):
    bins = BINS[m]
    if not bins:
        continue
    xs = [b[1] for b in bins]   # avg_conf
    ys = [b[2] for b in bins]   # avg_acc
    sz = [max(b[3]/8, 20) for b in bins]   # dot size ~ count
    ax_rel.scatter(xs, ys, s=sz, color=COLORS[i], marker=MARKERS[i],
                   zorder=5, alpha=0.9, label=ml)
    if len(xs) > 1:
        ax_rel.plot(xs, ys, color=COLORS[i], linewidth=1.2, alpha=0.5)

ax_rel.set_xlim(-3, 103)
ax_rel.set_ylim(-3, 103)
ax_rel.set_xlabel('Mean stated confidence (%)')
ax_rel.set_ylabel('Actual accuracy (%)')
ax_rel.set_title('Reliability diagram\n(dot size ∝ number of tasks in bin)', fontsize=12)
ax_rel.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
ax_rel.set_xticks([0, 20, 40, 60, 80, 100])
ax_rel.set_yticks([0, 20, 40, 60, 80, 100])
ax_rel.grid(alpha=0.25, linewidth=0.6)
ax_rel.spines['top'].set_visible(False)
ax_rel.spines['right'].set_visible(False)

# ── Panel 2: ECE bars + bias overlay ──────────────────────────────────────
x = np.arange(len(MODELS))
bar_w = 0.38

bars = ax_ece.bar(x - bar_w/2, ECE,  bar_w, color=COLORS, alpha=0.85, label='ECE (↓)')
bars2= ax_ece.bar(x + bar_w/2, [abs(b) for b in BIAS], bar_w,
                   color=COLORS, alpha=0.40, hatch='//', label='|Overconf. bias| (↓)')

# Annotate ECE values
for bar, val in zip(bars, ECE):
    ax_ece.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='500')

ax_ece.set_xticks(x)
ax_ece.set_xticklabels(['LLaMA\n8B', 'GPT-4o\nmini', 'DS-V3', 'DS-R1', 'Qwen3\n32B', 'Gemini\n2.5F'],
                        fontsize=9.5)
ax_ece.set_ylabel('Percentage points')
ax_ece.set_title('Calibration error\n& overconfidence bias', fontsize=12)
ax_ece.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_ece.set_ylim(0, 68)
ax_ece.spines['top'].set_visible(False)
ax_ece.spines['right'].set_visible(False)
ax_ece.grid(axis='y', alpha=0.25, linewidth=0.6)

# ── Panel 3: AUROC bars ───────────────────────────────────────────────────
bars3 = ax_auroc.bar(x, AUROC, 0.6, color=COLORS, alpha=0.85)
ax_auroc.axhline(0.5, color='#888888', linewidth=1.2, linestyle='--', label='Random (0.5)')

for bar, val in zip(bars3, AUROC):
    ax_auroc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='500')

ax_auroc.set_xticks(x)
ax_auroc.set_xticklabels(['LLaMA\n8B', 'GPT-4o\nmini', 'DS-V3', 'DS-R1', 'Qwen3\n32B', 'Gemini\n2.5F'],
                          fontsize=9.5)
ax_auroc.set_ylabel('AUROC')
ax_auroc.set_title('Discrimination ability\n(confidence → correctness)', fontsize=12)
ax_auroc.set_ylim(0.35, 0.80)
ax_auroc.legend(fontsize=9, framealpha=0.9)
ax_auroc.spines['top'].set_visible(False)
ax_auroc.spines['right'].set_visible(False)
ax_auroc.grid(axis='y', alpha=0.25, linewidth=0.6)

fig.suptitle('Confidence calibration across 6 models — verbalized 0–100% elicitation protocol',
             fontsize=13, y=1.03)

plt.savefig('fig5_calibration.pdf', bbox_inches='tight')
plt.savefig('fig5_calibration.png', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")