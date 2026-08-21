"""
Regenerate Supplementary Figure S2 with the post-transition / pre-transition framing
(change-point at 2025-W48 = 2025-11-27; pre n=135, post n=24), matching the manuscript
and the S2 caption. Uses the full 159-week n19 dataset.
"""
import os
import runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import FIGURES

# Pull the variant W-matrices and per-variant Moran's I series from the existing
# robustness machinery, run over the 159-week n19 data.
ns = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'robustness_variants.py'))
variants = ns['variants']
results = ns['results']
dates = pd.to_datetime(ns['dates'])
T = ns['T']

# Shared style (subset of figs_setup.py — avoids its heavy data loading)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 9,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})
C_PRE = '#4472C4'
C_2026 = '#C00000'

# Change-point (2025-W48). Pre = [0:cp] (n=135), Post = [cp:] (n=24).
CP_DATE = pd.Timestamp('2025-11-27')
cp = int(dates.searchsorted(CP_DATE))
pre_mask = np.arange(T) < cp
post_mask = ~pre_mask
n_pre = int(pre_mask.sum())
n_post = int(post_mask.sum())
assert (n_pre, n_post) == (135, 24), (n_pre, n_post)

variant_labels = list(variants.keys())

ratios, pre_means, post_means = [], [], []
for name in variant_labels:
    arr = results[name]
    mp = np.nanmean(arr[pre_mask])
    m_post = np.nanmean(arr[post_mask])
    pre_means.append(mp)
    post_means.append(m_post)
    ratios.append(m_post / mp if mp != 0 else np.nan)
ratios = np.array(ratios)

fig = plt.figure(figsize=(7.0, 6.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=1.00)

# ----- Panel A: post/pre ratio across variants -----
ax = fig.add_subplot(gs[0])
x_pos = np.arange(len(variant_labels))
colors = ['#4472C4'] + ['#7FA0D0'] * (len(variant_labels) - 1)
ax.bar(x_pos, ratios, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x_pos, ratios, color='none', edgecolor='black', linewidth=0.4)
ax.axhline(1.0, color='black', linewidth=0.6, alpha=0.5, linestyle=':')
for i, r in enumerate(ratios):
    if not np.isnan(r):
        ax.text(i, r + 0.05, f'{r:.2f}\u00d7', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(variant_labels, rotation=30, ha='right', fontsize=7.5)
ax.set_ylabel("Ratio: post-transition mean I /\npre-transition mean I", fontsize=9)
ax.set_ylim(0, max(ratios) * 1.25)
ax.grid(True, axis='y', alpha=0.3, linewidth=0.3)
ax.text(-0.07, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('All 7 adjacency-matrix variants confirm elevated post-transition spatial autocorrelation',
             fontsize=8.5, pad=4)

# ----- Panel B: I(t) trajectories for the 7 variants -----
ax = fig.add_subplot(gs[1])
import matplotlib.cm as cm
cmap = cm.get_cmap('Blues', 9)
for k, name in enumerate(variant_labels):
    arr = results[name]
    if k == 0:
        ax.plot(dates, arr, '-', color='#4472C4', linewidth=1.4, alpha=0.95,
                label='baseline (k=0)', zorder=10)
    else:
        ax.plot(dates, arr, '-', color=cmap(0.3 + 0.1 * k), linewidth=0.7, alpha=0.55,
                label=name, zorder=2)
# Shade the post-transition window (change-point onward)
ax.axvspan(CP_DATE, dates[-1], color=C_2026, alpha=0.08)
ax.axvline(CP_DATE, color=C_2026, linewidth=0.8, alpha=0.5, linestyle='--')
ax.set_ylabel("Global Moran's I (per variant)", fontsize=9)
# legend outside the data area: a horizontal strip between the panel title
# and the axes (it used to sit on top of the seven trajectories)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=6.5, ncol=4,
          frameon=False, columnspacing=1.2, handlelength=1.6)
ax.grid(True, alpha=0.3, linewidth=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=7)
ax.text(-0.07, 1.05, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title('I(t) trajectories for all 7 variants \u2014 overall pattern preserved (dashed line: transition)',
             fontsize=8.5, pad=30)

plt.savefig(FIGURES / 'FigS2_robustness.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Emit the numeric table for the caption / Table S2 reconciliation
print("=== FigS2 regenerated (post/pre split at 2025-11-27; pre n=%d, post n=%d) ===" % (n_pre, n_post))
print(f"{'Variant':32s} {'pre':>8s} {'post':>8s} {'ratio':>7s}")
for i, name in enumerate(variant_labels):
    print(f"{name:32s} {pre_means[i]:8.3f} {post_means[i]:8.3f} {ratios[i]:6.2f}x")
print("ratio range: %.2fx - %.2fx" % (np.nanmin(ratios), np.nanmax(ratios)))
