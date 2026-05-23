"""
Figure 1 (revised): the post-Class-5 wave template.
Panel A: S(t) full series + national per-sentinel mean (1/47)Sum_i x_i(t) overlaid
         on a logarithmic right axis (fulfils the figure's original design intent of
         showing nationwide case magnitude alongside the pseudo-entropy).
Panel B: composite S(t) trace aligned to the case-count peak of the 5 pre-transition
         waves (unchanged logic; reproduces min -7 wk / max +2 wk).
Data: n19 (159 weeks, ->2026-W19), matching the frozen submission manuscript.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR,_RES_DIR,_FIG_DIR=str(_DATA_DIR),str(_RES_DIR),str(_FIG_DIR)
# Portable paths: resolve inputs relative to the handoff root (parent of this folder).
_HERE = os.path.dirname(os.path.abspath(__file__)); _HANDOFF = os.path.dirname(_HERE)
from epi_week_axis import set_epi_week_xaxis, date_to_epi

# ---- style (match figs_setup.py) ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 9, 'axes.linewidth': 0.7,
    'axes.spines.top': False, 'axes.spines.right': True,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'legend.frameon': True,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
C_PRE = '#4472C4'; C_2026 = '#C00000'; C_NEUTRAL = '#666666'
CASE_COLOR = '#9C6B3F'

# ---- data ----
NPZ = os.path.join(_RES_DIR, 'phase1_changepoint_results.npz')
COMBINED = os.path.join(_DATA_DIR, 'combined_n19.csv')
d = np.load(NPZ, allow_pickle=True)
dates = pd.to_datetime(d['dates']); S = d['S']; T = len(S)
comb = pd.read_csv(COMBINED, parse_dates=['date']).set_index('date').reindex(dates)
percap = comb['cases'].values / 47.0
assert not np.isnan(percap).any()

cp_date = pd.Timestamp('2025-11-24')   # 2025-W48
cp_idx = int(np.argmin(np.abs((dates - cp_date).total_seconds())))

# pre-transition wave peaks (per-sentinel), for labels + composite
v = percap.copy(); raw = []
for i in range(2, T-2):
    if v[i] == max(v[i-2:i+3]) and v[i] > 5 and dates[i] < cp_date:
        raw.append(i)
peaks = []
for p in raw:
    if peaks and p - peaks[-1] <= 6:
        if v[p] > v[peaks[-1]]: peaks[-1] = p
        continue
    peaks.append(p)
recent_idx = int(np.argmin(np.abs((dates - pd.Timestamp('2026-02-05')).total_seconds())))

# ---- figure ----
fig = plt.figure(figsize=(7.2, 5.1))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.42)

# ===== Panel A =====
axA = fig.add_subplot(gs[0])
axR = axA.twinx(); axR.set_zorder(axA.get_zorder() - 1); axA.patch.set_visible(False)
R_FLOOR, R_TOP = 0.3, 200.0
axR.set_yscale('log')
band = axR.fill_between(dates, percap, R_FLOOR, color=CASE_COLOR, alpha=0.14, lw=0, zorder=1)
axR.plot(dates, percap, color=CASE_COLOR, lw=0.8, alpha=0.5, zorder=1)
axR.set_ylim(R_FLOOR, R_TOP)
axR.yaxis.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
axR.yaxis.set_major_formatter(mticker.FixedFormatter(['0.5', '1', '2', '5', '10', '20']))
axR.yaxis.set_minor_locator(mticker.NullLocator())
axR.set_ylabel(r'Mean reports per sentinel (log)', fontsize=8.5, color=CASE_COLOR)
axR.tick_params(axis='y', colors=CASE_COLOR, labelsize=7.5)
axR.spines['right'].set_color(CASE_COLOR); axR.spines['top'].set_visible(False)

# entropy (foreground)
axA.plot(dates, S, '-', color=C_PRE, linewidth=1.3, label='S(t)', zorder=6)
axA.axhline(np.log(47), color=C_NEUTRAL, linestyle=':', alpha=0.6, linewidth=0.8,
            label='ln 47 ≈ 3.850', zorder=4)
axA.axvspan(dates[cp_idx], dates[-1], color=C_2026, alpha=0.08, zorder=2,
            label='Post-transition (n = 24)')
axA.axvline(dates[cp_idx], color=C_2026, linestyle='--', linewidth=1.6, zorder=5,
            label='Change-point: 2025-W48')

# wave labels on the case humps
for k, p in enumerate(peaks, start=1):
    axR.annotate(f'Wave {k}', (dates[p], percap[p]), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=7.5, color=CASE_COLOR, fontweight='bold', zorder=8)
axR.annotate('Recent wave\n(2026)', (dates[recent_idx], percap[recent_idx]),
             xytext=(-1, 30), textcoords='offset points', ha='center', va='bottom',
             fontsize=7.5, color=CASE_COLOR, fontweight='bold', zorder=8,
             arrowprops=dict(arrowstyle='->', color=CASE_COLOR, lw=0.8, alpha=0.8))

axA.set_ylabel('Pseudo-entropy S(t)', fontsize=9)
axA.set_ylim(3.50, 3.88)
axA.set_title('A. Pseudo-entropy time series with national per-sentinel mean', fontsize=10, loc='left')
set_epi_week_xaxis(axA, dates[0], dates[-1], n_ticks=8)
axA.grid(True, alpha=0.3, linewidth=0.3)
h, l = axA.get_legend_handles_labels()
h.append(band); l.append(r'$(1/47)\sum_i x_i(t)$ (right axis)')
leg = axA.legend(h, l, loc='lower left', fontsize=7, ncol=2, framealpha=0.75); leg.set_zorder(20)

# ===== Panel B (composite, unchanged logic) =====
axB = fig.add_subplot(gs[1])
axB.spines['right'].set_visible(False)
W = 10; traces = [S[p-W:p+W+1] for p in peaks if p >= W and p < T-W]
comp = np.array(traces); m = comp.mean(0); sem = comp.std(0)/np.sqrt(len(comp))
axis = np.arange(-W, W+1)
axB.fill_between(axis, m-sem, m+sem, color=C_PRE, alpha=0.20, label=f'±SEM ({len(traces)} waves)')
axB.plot(axis, m, '-', color=C_PRE, linewidth=1.6, label=f'Mean across {len(traces)} waves')
axB.axvline(0, color=C_NEUTRAL, linewidth=0.8, alpha=0.5)
axB.axhline(np.log(47), color=C_NEUTRAL, linestyle=':', alpha=0.4, linewidth=0.7)
ti, pi = m.argmin(), m.argmax()
axB.annotate(f'min at\n{axis[ti]:+d} wk', xy=(axis[ti], m[ti]),
             xytext=(axis[ti]-0.5, m[ti]-0.04), ha='right', fontsize=7,
             arrowprops=dict(arrowstyle='->', color='#444444', lw=0.5))
axB.annotate(f'max at\n{axis[pi]:+d} wk', xy=(axis[pi], m[pi]),
             xytext=(axis[pi]+0.5, m[pi]+0.012), ha='left', fontsize=7,
             arrowprops=dict(arrowstyle='->', color='#444444', lw=0.5))
axB.set_xlabel('Weeks relative to wave peak (case-count maximum)', fontsize=9)
axB.set_ylabel('Pseudo-entropy S', fontsize=9)
axB.set_title('B. Composite trace, pre-transition waves (n = %d)' % len(traces), fontsize=10, loc='left')
axB.legend(loc='lower right', fontsize=7.5, framealpha=0.8)
axB.grid(True, alpha=0.3, linewidth=0.3); axB.set_xlim(-W, W)

plt.savefig(os.path.join(_FIG_DIR, 'Fig1_wave_template.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(_FIG_DIR, 'Fig1_wave_template.pdf'), bbox_inches='tight', facecolor='white')
print('saved Fig1 overlay; peaks:', [str(dates[p].date()) for p in peaks],
      '| composite min %+dwk max %+dwk' % (axis[ti], axis[pi]))
