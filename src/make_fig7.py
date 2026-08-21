"""
Figure 7 (publication quality): entropy-ceiling-failure restricted change-point detection
Panel A: Full S(t) series with threshold + restricted region
Panel B: Zoomed restricted region with PELT change-point and pre/post means
Panel C: Threshold sensitivity (PELT-detected CP for thresholds 3.78..3.82)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR, _RES_DIR, _FIG_DIR = str(_DATA_DIR), str(_RES_DIR), str(_FIG_DIR)

import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from epi_week_axis import date_to_epi as _de, set_epi_week_xaxis
from matplotlib.dates import DateFormatter, MonthLocator

# Load
data = np.load(os.path.join(_RES_DIR, 'phase1_changepoint_results.npz'), allow_pickle=True)
dates = pd.to_datetime(data['dates'])
S = data['S']
T = len(S)

THRESHOLD = 3.80
above = np.where(S >= THRESHOLD)[0]
t_last = above[-1]

# PELT on restricted region
S_restricted = S[t_last+1:]
T_r = len(S_restricted)
pen_main = 2 * np.var(S_restricted) * np.log(T_r)
cps_local = rpt.Pelt(model='l2', min_size=5).fit(S_restricted).predict(pen=pen_main)
cps_local = [c for c in cps_local if c < T_r]
cp_local = cps_local[0]
cp_global = cp_local + t_last + 1

# Compute means before/after CP within restricted region
S_pre_local = S_restricted[:cp_local]
S_post_local = S_restricted[cp_local:]

# Compute pre-transition ceiling stats (Wave 1..Wave 5 S_max)
wave_smax = [3.836, 3.827, 3.827, 3.820, 3.826]

# Threshold sensitivity
threshold_results = []
for th in [3.78, 3.79, 3.80, 3.81, 3.82]:
    above_th = np.where(S >= th)[0]
    if len(above_th) == 0:
        continue
    t_last_th = above_th[-1]
    if T - t_last_th - 1 < 10:
        continue
    S_r = S[t_last_th+1:]
    pen = 2 * np.var(S_r) * np.log(len(S_r))
    cps = rpt.Pelt(model='l2', min_size=5).fit(S_r).predict(pen=pen)
    cps = [c for c in cps if c < len(S_r)]
    if cps:
        threshold_results.append({'threshold': th,
                                  't_last': t_last_th,
                                  'cp_global': cps[0] + t_last_th + 1,
                                  'cp_date': dates[cps[0] + t_last_th + 1]})

# ---------- Figure ----------
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 0.8], hspace=0.45)

# Panel A: Full series with threshold and restricted region
axA = fig.add_subplot(gs[0])
axA.plot(dates, S, color='#1f77b4', linewidth=1.0, label='S(t)')
# Mark threshold
axA.axhline(THRESHOLD, color='#d62728', linestyle='--', linewidth=1.2,
            alpha=0.75, label=f'Entropy-ceiling threshold = {THRESHOLD}')
# Mark theoretical max
axA.axhline(np.log(47), color='k', linestyle=':', alpha=0.4, label='ln 47 ≈ 3.850')
# Mark wave S_max points using EXACT dates from Table 3 to avoid mismatches
# (some waves share S_max values, so find by date)
wave_targets = [
    ('Wave 1', pd.Timestamp('2023-09-21'), 3.836, (5, 8),   'left'),
    ('Wave 2', pd.Timestamp('2024-02-22'), 3.827, (5, 8),   'left'),
    ('Wave 3', pd.Timestamp('2024-08-08'), 3.827, (5, 8),   'left'),
    ('Wave 4', pd.Timestamp('2025-01-30'), 3.820, (-5, 8),  'right'),
    ('Wave 5', pd.Timestamp('2025-10-09'), 3.826, (-25, -2), 'right'),
]
for label, target_date, expected_s, offset, halign in wave_targets:
    # Find the week closest to target_date
    idx_smax = int(np.argmin(np.abs((dates - target_date).total_seconds())))
    axA.plot(dates[idx_smax], S[idx_smax], 'o', color='#2ca02c', markersize=6,
            zorder=5)
    axA.annotate(label, (dates[idx_smax], S[idx_smax]),
                xytext=offset, textcoords='offset points', fontsize=8,
                color='#2ca02c', fontweight='bold', ha=halign)
# Mark t_last
def date_to_epi(ts):
    iso = pd.Timestamp(ts).isocalendar()
    return f'{iso.year}-W{iso.week:02d}'

t_last_epi = date_to_epi(dates[t_last])  # 2025-W42
cp_epi = date_to_epi(dates[cp_global])    # 2025-W48

axA.axvline(dates[t_last], color='#2ca02c', linewidth=2, alpha=0.6,
            label=f't_last (last S ≥ {THRESHOLD}) = {t_last_epi}')
# Shade restricted region
axA.axvspan(dates[t_last], dates[-1], color='#ff9896', alpha=0.18,
            label='Restricted CP search region')
# Mark detected CP
axA.axvline(dates[cp_global], color='#d62728', linewidth=2.5,
            label=f'PELT CP = {cp_epi}')

axA.set_ylabel('Shannon entropy S(t)', fontsize=11)
axA.set_title('A. Full series with entropy-ceiling-failure restriction', fontsize=11, loc='left')
axA.legend(loc='lower left', fontsize=7, ncol=2)
axA.grid(True, alpha=0.3)
axA.set_ylim(3.52, 3.87)
set_epi_week_xaxis(axA, dates[0], dates[-1], n_ticks=8)

# Panel B: Zoomed restricted region with PELT CP and pre/post means
axB = fig.add_subplot(gs[1])
restricted_dates = dates[t_last+1:]
axB.plot(restricted_dates, S_restricted, color='#1f77b4', linewidth=1.5, marker='o', markersize=4)
# Mark CP
axB.axvline(dates[cp_global], color='#d62728', linewidth=2.5, linestyle='--',
            label=f'PELT CP: {cp_epi}')
# Pre-CP mean
if len(S_pre_local) > 0:
    pre_mean = S_pre_local.mean()
    axB.hlines(pre_mean, restricted_dates[0], dates[cp_global],
               colors='#2ca02c', linewidth=2.5, alpha=0.75)
    axB.annotate(f'mean = {pre_mean:.3f}', xy=(dates[cp_global], pre_mean),
                xytext=(-6, 7), textcoords='offset points', ha='right', va='bottom',
                fontsize=9, color='#2ca02c', fontweight='bold')
# Post-CP mean
post_mean = S_post_local.mean()
axB.hlines(post_mean, dates[cp_global], dates[-1],
           colors='#ff7f0e', linewidth=2.5, alpha=0.75)
_lbl = next(i for i in range(cp_global, len(dates)) if date_to_epi(dates[i]) == '2026-W07')
axB.annotate(f'mean = {post_mean:.3f}', xy=(dates[_lbl], post_mean),
            xytext=(-12, 5), textcoords='offset points', ha='center', va='bottom',
            fontsize=9, color='#ff7f0e', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0))
# CUSUM and Bai-Perron secondary markers (epi week labels)
cusum_idx = t_last + 1 + 7  # 2025-12-11
bp_idx = t_last + 1 + 11   # 2026-01-08
cusum_epi = date_to_epi(dates[cusum_idx])  # 2025-W50
bp_epi = date_to_epi(dates[bp_idx])         # 2026-W02
axB.axvline(dates[cusum_idx], color='#9467bd', linestyle=':', linewidth=1.2, alpha=0.7,
           label=f'CUSUM peak: {cusum_epi}')
axB.axvline(dates[bp_idx], color='#8c564b', linestyle=':', linewidth=1.2, alpha=0.7,
           label=f'Bai-Perron sup-F peak: {bp_epi}')

axB.set_ylabel('S(t) in restricted region', fontsize=11)
axB.set_title('B. Restricted region (29 weeks): PELT change-point and pre/post means',
              fontsize=11, loc='left')
axB.legend(loc='upper right', fontsize=8)
axB.grid(True, alpha=0.3)
set_epi_week_xaxis(axB, restricted_dates[0], dates[-1], n_ticks=8)

# Panel C: Threshold sensitivity (epi week x-axis)
axC = fig.add_subplot(gs[2])
ths = [r['threshold'] for r in threshold_results]
cp_dates_th = [r['cp_date'] for r in threshold_results]
# Convert CP dates to epi week numbers for x-positioning
def epi_week_num(ts):
    iso = pd.Timestamp(ts).isocalendar()
    # If 2025-Wnn, return nn; if 2026-Wnn, return 52 + nn
    if iso.year == 2025:
        return iso.week
    elif iso.year == 2026:
        return 52 + iso.week
    return iso.week

cp_xvals = [epi_week_num(d) for d in cp_dates_th]
cp_labels = [date_to_epi(d) for d in cp_dates_th]
axC.plot(cp_xvals, ths, 'o-', color='#1f77b4', markersize=8, linewidth=1.5)
# Annotate each point with epi week label and date
for cp_x, th, cp_l, cp_d in zip(cp_xvals, ths, cp_labels, cp_dates_th):
    axC.annotate(f'  {cp_l}',
                 (cp_x, th), fontsize=8, va='center')
axC.set_xlabel('PELT-detected change-point epi week (with BIC penalty)', fontsize=10)
axC.set_ylabel('Entropy-ceiling\nthreshold', fontsize=10)
axC.set_title('C. Threshold sensitivity: PELT change-points across threshold values 3.78–3.82',
              fontsize=11, loc='left')
axC.grid(True, alpha=0.3)
# X range: from 2025-W45 (early Nov 2025) to 2025-W52 (end Dec 2025)
axC.set_xlim(45, 52)
axC.set_ylim(3.77, 3.83)
# Custom x-tick labels at integer epi week values
axC.set_xticks([45, 46, 47, 48, 49, 50, 51, 52])
axC.set_xticklabels(['2025-W45', '2025-W46', '2025-W47', '2025-W48',
                     '2025-W49', '2025-W50', '2025-W51', '2025-W52'],
                    rotation=20, ha='right', fontsize=8)

# Figure title is supplied by the manuscript caption (BMC style): no in-image title.
plt.savefig(os.path.join(_FIG_DIR, 'Fig7_changepoint.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(_FIG_DIR, 'Fig7_changepoint.pdf'), bbox_inches='tight')
print('Figure 7 saved: Fig7_changepoint.png, .pdf')
print(f'\nKey details for caption:')
print(f't_last = {dates[t_last].date()}, S(t_last) = {S[t_last]:.4f}')
print(f'PELT CP = {dates[cp_global].date()}')
print(f'Pre-CP mean (in restricted region): {pre_mean:.4f} (n = {len(S_pre_local)})')
print(f'Post-CP mean: {post_mean:.4f} (n = {len(S_post_local)})')
