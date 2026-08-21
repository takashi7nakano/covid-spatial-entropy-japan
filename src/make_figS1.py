"""
Supplementary Figure S1: the 92-edge adjacency specification.

A: the network drawn on Japan (prefecture capitals), sea/strait crossings highlighted
B: the W matrix with prefectures sorted north -> south

Recovered from handoff_2026-06-23 (6_Regenerated_Figures_n19/figS1.py) and adapted to
the repository layout on 2026-08-10. The two corner annotations, which used to sit on
the matrix and on the x-axis label, are now placed in the empty off-diagonal triangles
with leader lines.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from config import ADJACENCY, FIGURES
from prefecture_coords import PREFS, COORDS, idx, N

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 9,
    'axes.linewidth': 0.7,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})
C_2026 = '#C00000'

# ---------------------------------------------------------------- W
adj = pd.read_csv(ADJACENCY)
W = np.zeros((N, N))
for a, b in adj.itertuples(index=False):
    i, j = idx[a], idx[b]
    W[i, j] = W[j, i] = 1.0
edges = [(i, j) for i in range(N) for j in range(i + 1, N) if W[i, j] > 0]
print('Drawing %d edges' % len(edges))

SEA_EDGES = [('Hokkaido', 'Aomori'), ('Yamaguchi', 'Fukuoka'), ('Okayama', 'Kagawa'),
             ('Hiroshima', 'Ehime'), ('Hyogo', 'Tokushima'), ('Kagoshima', 'Okinawa')]

fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.10)

# ---------------------------------------------------------------- Panel A
ax = fig.add_subplot(gs[0])
land, sea = [], []
for i, j in edges:
    a, b = PREFS[i], PREFS[j]
    seg = [(COORDS[a][1], COORDS[a][0]), (COORDS[b][1], COORDS[b][0])]
    (sea if ((a, b) in SEA_EDGES or (b, a) in SEA_EDGES) else land).append(seg)
ax.add_collection(LineCollection(land, colors='#7090C0', linewidths=0.8, alpha=0.7))
ax.add_collection(LineCollection(sea, colors=C_2026, linewidths=1.6, alpha=0.9,
                                 linestyles='--'))
deg = W.sum(axis=0)
for i, name in enumerate(PREFS):
    lat, lon = COORDS[name]
    ax.scatter(lon, lat, s=30 + 30 * deg[i], color='#FFE699', edgecolors='black',
               linewidth=0.5, zorder=5)
ax.set_xlim(126, 146.5)
ax.set_ylim(25, 46)
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title('A  Adjacency network (92 edges)', fontsize=10, fontweight='bold', loc='left')
ax.legend(handles=[
    Line2D([0], [0], color='#7090C0', lw=1.2, label='Land border (86 edges)'),
    Line2D([0], [0], color=C_2026, lw=1.8, linestyle='--',
           label='Sea/strait crossing (6 edges)')],
    loc='upper left', fontsize=8, framealpha=0.92)

# ---------------------------------------------------------------- Panel B
ax = fig.add_subplot(gs[1])
lats = np.array([COORDS[p][0] for p in PREFS])
order = np.argsort(-lats)
ax.imshow(W[order][:, order], cmap='Blues', aspect='equal', interpolation='nearest')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('B  W matrix (sorted N→S)', fontsize=10, fontweight='bold', loc='left')

# The two corner notes used to be written on top of the matrix (Hokkaido) and across
# the x-axis label (Okinawa). Both now sit in verified-empty off-diagonal blocks
# (cols 8-21 x rows 1-4, and cols 15-30 x rows 43-46) with a leader to the cell.
ARROW = dict(arrowstyle='-', color='#7A0000', lw=0.6, shrinkA=1, shrinkB=1)
ax.annotate('Hokkaido\n(only Aomori link)', xy=(0, 0), xytext=(8.5, 3.0),
            fontsize=6, color='#7A0000', ha='left', va='center', arrowprops=ARROW)
ax.annotate('Okinawa\n(only Kagoshima link)', xy=(46, 46), xytext=(29.5, 44.0),
            fontsize=6, color='#7A0000', ha='right', va='center', arrowprops=ARROW)
ax.set_xlabel('Prefecture index (N→S)', fontsize=8)
ax.set_ylabel('Prefecture index (N→S)', fontsize=8)

plt.savefig(FIGURES / 'FigS1_adjacency.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close()
print('Saved FigS1_adjacency.png')
