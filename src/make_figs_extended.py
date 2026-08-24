#!/usr/bin/env python3
"""Figures 1, 6 and 8 of the submitted manuscript, from the full 173-week record.

The remaining figures (2, 3, 4, 5, 7) are produced by `regenerate_figs.py` and
`make_fig7.py` from the frozen 159-week estimation window and are unchanged.
This script is the only one that reads the 14 out-of-sample weeks
(2026-W20 .. 2026-W30); no estimation is performed here.

Outputs (repository `figures/submission/`, manuscript figure numbers):
  Fig1_wave_template.png       panel A extended to 2026-W30 (B unchanged)
  Fig6_phase_plane.png         panel B post-transition extended 24 -> 35 weeks
  Fig8_prospective_test.png    out-of-sample test of the pre-specified criterion

Run `export_submission_figures.py` afterwards to copy the remaining five figures
into the same directory under their manuscript numbers (see README,
"Figure numbering").

Moran's I uses the same (N/S0)*z'Wz/z'z estimator as `common.py::morans_I` and
`regenerate_figs.py::moran_t`.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import DATA, FIGURES

# This script writes straight into figures/submission/ under the figure numbers
# used in the submitted manuscript. The five figures that come from the frozen
# 159-week window keep their preprint numbers in figures/ and are copied across
# by export_submission_figures.py. Keeping the two numbering schemes in separate
# directories is what stops "Fig6" from meaning two different things.
SUBMISSION = FIGURES / 'submission'
SUBMISSION.mkdir(exist_ok=True)
OUTS = [str(SUBMISSION)]

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
C_PRE = '#4472C4'; C_POST = '#C00000'; C_SUM26 = '#E8820C'
C_NEUTRAL = '#666666'; CASE_COLOR = '#9C6B3F'
C_NORTH = '#2E5FA3'; C_KYUSHU = '#D9822B'

PREFS = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima','Ibaraki','Tochigi','Gunma',
         'Saitama','Chiba','Tokyo','Kanagawa','Niigata','Toyama','Ishikawa','Fukui','Yamanashi','Nagano',
         'Gifu','Shizuoka','Aichi','Mie','Shiga','Kyoto','Osaka','Hyogo','Nara','Wakayama','Tottori',
         'Shimane','Okayama','Hiroshima','Yamaguchi','Tokushima','Kagawa','Ehime','Kochi','Fukuoka',
         'Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa']
IDX = {p: i for i, p in enumerate(PREFS)}
KY7 = ['Fukuoka','Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima']
KYOKI = KY7 + ['Okinawa']
NORTH7 = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima']
CP = pd.Timestamp('2025-11-24')          # 2025-W48
CUTOFF = pd.Timestamp('2026-05-07')      # last week of the preprint data (2026-W19)
CEIL = 3.80


def build():
    """Load the 173-week record and return (dates, per-sentinel rate matrix).

    `pref_full_n33.csv` stores the weekly prefecture shares p_i(t) and
    `combined_n33.csv` the national per-sentinel total cases(t) = sum_i x_i(t),
    so the rate matrix is recovered as x_i(t) = p_i(t) * cases(t). Because every
    prefecture value is published to two decimals, the stored total is exact and
    the reconstruction is faithful to ~1e-9.
    """
    p = pd.read_csv(os.path.join(str(DATA), 'pref_full_n33.csv'))
    c = pd.read_csv(os.path.join(str(DATA), 'combined_n33.csv'))
    for df in (p, c):
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    cols = [x for x in p.columns if x not in ('date', 'S')]
    assert cols == PREFS, 'unexpected prefecture column order'
    m = p.merge(c[['date', 'cases', 'window']], on='date')
    M = m[cols].values * m['cases'].values[:, None]
    assert (m['window'] == 'out-of-sample').sum() == 14, 'expected 14 out-of-sample weeks'
    return pd.to_datetime(m['date'].values), M


dates, M = build()
T = len(dates)
amp = M.mean(axis=1)
P = M / M.sum(axis=1, keepdims=True)
S = -(P * np.log(P + 1e-30)).sum(axis=1)

E = pd.read_csv(os.path.join(DATA, 'adjacency_edges.csv'))
ca, cb = E.columns[0], E.columns[1]
W = np.zeros((47, 47))
for a, b in zip(E[ca], E[cb]):
    W[IDX[a], IDX[b]] = W[IDX[b], IDX[a]] = 1.0
S0 = W.sum()


def moran_t(pt):
    z = pt - pt.mean()
    den = (z ** 2).sum()
    return np.nan if den == 0 else (47 / S0) * (W * np.outer(z, z)).sum() / den


I_t = np.array([moran_t(P[t]) for t in range(T)])


def kl_signed(pt, grp):
    return sum(pt[IDX[p]] * np.log(47 * pt[IDX[p]] + 1e-30) for p in grp)


KL_k = np.array([kl_signed(P[t], KYOKI) for t in range(T)])
KL_n = np.array([kl_signed(P[t], NORTH7) for t in range(T)])
x_pp = KL_n - KL_k

pre = np.asarray(dates < CP)
post = ~pre
sum26 = np.asarray(dates > CUTOFF)                     # 2026-W20..W30
post_early = post & ~sum26                            # change-point .. 2026-W19

# Wave windows, delimited by troughs of the national amplitude
tr = [i for i in range(2, T - 2) if amp[i] == amp[max(0, i - 4):i + 5].min()]
ded = []
for i in tr:
    if not ded or i - ded[-1] > 8: ded.append(i)
    elif amp[i] < amp[ded[-1]]: ded[-1] = i
WAVES = [('Wave 1', '2023-08-31'), ('Wave 2', '2024-02-01'), ('Wave 3', '2024-07-25'),
         ('Wave 4', '2025-01-09'), ('Wave 5', '2025-08-21'),
         ('Winter\n2025-26', '2026-02-05'), ('Summer\n2026', '2026-07-16')]
dstr = [d.strftime('%Y-%m-%d') for d in dates]


def window(pkd):
    k = dstr.index(pkd)
    return (max([t for t in ded if t < k], default=max(0, k - 12)),
            min([t for t in ded if t > k], default=T - 1))


wave_rows = []
for lab, pkd in WAVES:
    k = dstr.index(pkd)
    lo, hi = window(pkd)
    j = lo + int(np.argmax(S[lo:hi + 1]))
    wave_rows.append(dict(label=lab.replace('\n', ' '), peak=k, amp=amp[k],
                          smax=S[j], smax_idx=j, is_pre=(dates[k] < CP)))
WV = pd.DataFrame(wave_rows)


def save(fig, name):
    for o in OUTS:
        fig.savefig(os.path.join(o, name), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(o, name.replace('.png', '.pdf')), bbox_inches='tight')
    plt.close(fig)
    print('  saved %s' % name)


def epi(ts):
    iso = pd.Timestamp(ts).isocalendar()
    return f'{iso.year}-W{iso.week:02d}'


print('data %s .. %s (%d weeks)' % (dstr[0], dstr[-1], T))
print('pre %d / post %d (of which 2026-W20..W30: %d)' % (pre.sum(), post.sum(), sum26.sum()))

# =====================================================================
# Figure 1 - wave template (panel A extended to 173 weeks; panel B unchanged)
# =====================================================================
fig = plt.figure(figsize=(11.5, 8.0))
axA = fig.add_subplot(2, 1, 1)

axA.axvspan(dates[post][0], dates[-1], color='#C00000', alpha=0.055, zorder=0)
axA.axhline(np.log(47), ls=':', lw=0.9, color='#888888', zorder=1)
axA.axhline(CEIL, ls='--', lw=1.0, color=C_POST, alpha=0.85, zorder=1)
axA.plot(dates, S, color=C_PRE, lw=1.5, zorder=4, label='$S(t)$')
axA.axvline(CP, color=C_POST, ls='--', lw=1.8, zorder=3)
axA.axvline(CUTOFF, color='#444444', ls='-.', lw=1.1, alpha=0.8, zorder=3)

ax2 = axA.twinx()
ax2.fill_between(dates, amp, color=CASE_COLOR, alpha=0.20, zorder=2)
ax2.set_yscale('log')
ax2.set_ylabel('Mean reports per sentinel (log)', color=CASE_COLOR)
ax2.tick_params(axis='y', colors=CASE_COLOR)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))

# the last wave peaks close to the right edge; nudge its label left so that it
# clears the right-hand axis label
LABEL_DX = {'Summer 2026': -16}
for r in WV.itertuples():
    axA.annotate(r.label, xy=(dates[r.peak], 3.505), ha='center', va='bottom',
                 xytext=(LABEL_DX.get(r.label, 0), 0), textcoords='offset points',
                 fontsize=8, color='#7B4B1E' if r.is_pre else C_POST,
                 fontweight='bold', zorder=6)
axA.annotate('ceiling  $S \\geq 3.80$', xy=(dates[T // 2], CEIL), xytext=(0, 3),
             textcoords='offset points', fontsize=8, color=C_POST, va='bottom', ha='center')
axA.annotate('$\\ln 47 = 3.850$', xy=(dates[T // 2], np.log(47)), xytext=(0, 2),
             textcoords='offset points', fontsize=8, color='#777777', va='bottom', ha='center')
axA.annotate('change-point\n2025-W48', xy=(CP, 3.875), ha='right', va='top',
             fontsize=8, color=C_POST, xytext=(-4, 0), textcoords='offset points')
axA.annotate('preprint data cutoff\n2026-W19', xy=(CUTOFF, 3.875), ha='left', va='top',
             fontsize=8, color='#333333', xytext=(4, 0), textcoords='offset points')

axA.set_ylim(3.10, 3.895)   # low enough to include the summer-2026 trough at 3.14
axA.set_ylabel('Pseudo-entropy $S(t)$')
axA.set_title('A. Pseudo-entropy and national per-sentinel mean, 2023-W17 to 2026-W30', loc='left')
ticks = pd.date_range(dates[0], dates[-1], periods=9)
axA.set_xticks(ticks)
axA.set_xticklabels([epi(t) for t in ticks], rotation=25, ha='right')
axA.set_xlim(dates[0], dates[-1])
axA.grid(True, alpha=0.25)
h = [plt.Line2D([], [], color=C_PRE, lw=1.5, label='$S(t)$'),
     plt.Line2D([], [], color=C_POST, ls='--', lw=1.0, label='entropy ceiling 3.80'),
     plt.Line2D([], [], color='#888888', ls=':', lw=0.9, label='$\\ln 47 = 3.850$'),
     plt.Line2D([], [], color=C_POST, ls='--', lw=1.8, label='change-point 2025-W48'),
     plt.Line2D([], [], color='#444444', ls='-.', lw=1.1, label='preprint data cutoff'),
     plt.Rectangle((0, 0), 1, 1, fc=CASE_COLOR, alpha=0.20,
                   label='$(1/47)\\sum_i x_i(t)$ (right axis)')]
axA.legend(handles=h, fontsize=7.5, loc='lower left', ncol=2, framealpha=0.95)

# --- B: composite of the five pre-transition waves (unchanged) ---
axB = fig.add_subplot(2, 1, 2)
REL = np.arange(-10, 11)
comp = []
for r in WV[WV.is_pre].itertuples():
    seg = [S[r.peak + k] if 0 <= r.peak + k < T else np.nan for k in REL]
    comp.append(seg)
comp = np.array(comp, float)
mu = np.nanmean(comp, axis=0)
sem = np.nanstd(comp, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(comp), axis=0))
axB.axhline(np.log(47), ls=':', lw=0.9, color='#888888')
axB.axhline(CEIL, ls='--', lw=1.0, color=C_POST, alpha=0.85)
axB.fill_between(REL, mu - sem, mu + sem, color=C_PRE, alpha=0.22, label='±SEM (5 waves)')
axB.plot(REL, mu, color=C_PRE, lw=2.0, label='Mean across 5 pre-transition waves')
s26 = WV[~WV.is_pre]
for r, c, ls in zip(s26.itertuples(), [C_SUM26, C_POST], ['--', '-']):
    seg = [S[r.peak + k] if 0 <= r.peak + k < T else np.nan for k in REL]
    axB.plot(REL, seg, color=c, lw=1.6, ls=ls, label=r.label)
axB.axvline(0, color='k', lw=0.6, alpha=0.6)
imin, imax = int(np.nanargmin(mu)), int(np.nanargmax(mu))
axB.annotate('min at %+d wk' % REL[imin], xy=(REL[imin], mu[imin]), xytext=(-6, -14),
             textcoords='offset points', fontsize=8,
             arrowprops=dict(arrowstyle='->', lw=0.6))
axB.annotate('max at %+d wk' % REL[imax], xy=(REL[imax], mu[imax]), xytext=(4, 8),
             textcoords='offset points', fontsize=8,
             arrowprops=dict(arrowstyle='->', lw=0.6))
axB.set_xlabel('Weeks relative to wave peak (national per-sentinel maximum)')
axB.set_ylabel('Pseudo-entropy $S$')
axB.set_title('B. Composite trace aligned on the wave peak', loc='left')
axB.legend(fontsize=8, loc='lower right')
axB.grid(True, alpha=0.25)

fig.tight_layout()
save(fig, 'Fig1_wave_template.png')

# =====================================================================
# Figure 6 (Fig5 in the preprint) - phase plane (panel B extended to 173 weeks)
# =====================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.2))
epi_w = np.array([pd.Timestamp(d).isocalendar().week for d in dates])

sc = axA.scatter(x_pp[pre], I_t[pre], c=epi_w[pre], cmap=plt.cm.twilight,
                 s=22, alpha=0.85, vmin=1, vmax=53)
axA.axhline(0, color='k', lw=0.5, alpha=0.5)
axA.axvline(0, color='k', lw=0.5, alpha=0.5)
axA.set_xlabel('$KL_{north} - KL_{kyushu}$')
axA.set_ylabel("Global Moran's $I$")
axA.set_title('A. Pre-transition trajectory (135 wks), colour = epi week', loc='left')
axA.grid(True, alpha=0.3)
cb = plt.colorbar(sc, ax=axA, shrink=0.75)
cb.set_label('Epi week', fontsize=8)

axB.scatter(x_pp[pre], I_t[pre], c='#bbbbbb', s=15, alpha=0.6,
            label='Pre-transition (n = 135)')
axB.scatter(x_pp[post_early], I_t[post_early], c=C_POST, s=38, alpha=0.9,
            edgecolors='k', linewidths=0.5, label='Post-transition to 2026-W19 (n = 24)')
o = np.argsort(dates[sum26])
xs, ys = x_pp[sum26][o], I_t[sum26][o]
axB.plot(xs, ys, color=C_SUM26, lw=1.2, alpha=0.85, zorder=4)
axB.scatter(xs, ys, c=C_SUM26, s=46, marker='D', edgecolors='k', linewidths=0.5,
            zorder=5, label='Summer 2026 wave (n = 11)')
for k in (0, len(xs) - 1):
    axB.annotate(epi(dates[sum26][o][k]), xy=(xs[k], ys[k]), xytext=(6, -9),
                 textcoords='offset points', fontsize=7.5, color='#8A4B00')
axB.annotate('', xy=(xs[-3], ys[-3]), xytext=(xs[-4], ys[-4]),
             arrowprops=dict(arrowstyle='-|>', color=C_SUM26, lw=1.4))
axB.axvline(x_pp[pre].min(), color='#999999', ls=':', lw=1.0)
axB.annotate('pre-transition minimum\n%.3f' % x_pp[pre].min(),
             xy=(x_pp[pre].min(), axB.get_ylim()[0]), xytext=(4, 12),
             textcoords='offset points', fontsize=7.5, color='#555555')
axB.axhline(0, color='k', lw=0.5, alpha=0.5)
axB.axvline(0, color='k', lw=0.5, alpha=0.5)
axB.annotate('Mode B-dominant', xy=(0.97, 0.96), xycoords='axes fraction',
             ha='right', va='top', fontsize=9, color=C_NORTH, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
axB.annotate('Mode A-dominant', xy=(0.03, 0.04), xycoords='axes fraction',
             ha='left', va='bottom', fontsize=9, color=C_KYUSHU, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
axB.set_xlabel('$KL_{north} - KL_{kyushu}$')
axB.set_ylabel("Global Moran's $I$")
axB.set_title('B. Post-transition weeks overlaid on the pre-transition record', loc='left')
axB.legend(fontsize=7.5, loc='upper left')
axB.grid(True, alpha=0.3)

fig.tight_layout()
save(fig, 'Fig6_phase_plane.png')

# =====================================================================
# Figure 8 - out-of-sample confirmatory test (new)
# =====================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

pr, po = WV[WV.is_pre], WV[~WV.is_pre]
axA.axhspan(CEIL, 3.86, color='#4472C4', alpha=0.07)
axA.axhline(CEIL, color=C_POST, ls='--', lw=1.4)
axA.axhline(np.log(47), color='#888888', ls=':', lw=0.9)
axA.scatter(pr.amp, pr.smax, s=95, c=C_PRE, edgecolors='k', linewidths=0.6,
            zorder=5, label='Pre-transition waves (5/5 attained)')
axA.scatter(po.amp, po.smax, s=110, marker='s', facecolors='none',
            edgecolors=C_POST, linewidths=1.8, zorder=5,
            label='Post-transition waves (0/2 attained)')
LBL = {'Wave 1': (0, 11, 'center'), 'Wave 2': (0, -17, 'center'),
       'Wave 3': (0, 11, 'center'), 'Wave 4': (0, 11, 'center'),
       'Wave 5': (0, -17, 'center'),
       'Winter 2025-26': (13, 4, 'left'), 'Summer 2026': (13, -4, 'left')}
for r in WV.itertuples():
    dx, dy, ha = LBL[r.label]
    axA.annotate(r.label, xy=(r.amp, r.smax), xytext=(dx, dy),
                 textcoords='offset points', ha=ha, va='center', fontsize=7.5,
                 color='#333333' if r.is_pre else C_POST)
axA.annotate('', xy=(pr.amp.min(), 3.868), xytext=(pr.amp.max(), 3.868),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
axA.annotate('2.7-fold range of peak amplitude', xy=(np.sqrt(pr.amp.min() * pr.amp.max()), 3.872),
             ha='center', va='bottom', fontsize=8, color='#555555')
axA.annotate('entropy ceiling  $S \\geq 3.80$', xy=(22, CEIL), xytext=(0, -16),
             textcoords='offset points', fontsize=8, color=C_POST, ha='center')
axA.set_xscale('log')
axA.set_xlim(2.0, 34)
axA.set_xticks([2, 3, 5, 8, 10, 15, 20, 30])
axA.set_ylim(3.63, 3.895)
axA.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
axA.set_xlabel('Peak amplitude (reports per sentinel facility, log scale)')
axA.set_ylabel('Within-wave maximum of $S$')
axA.set_title('A. Attainment of the pre-specified criterion, by wave', loc='left')
axA.legend(fontsize=8, loc='lower right')
axA.grid(True, alpha=0.25)

lo26, hi26 = window('2026-07-16')
sl = slice(lo26, hi26 + 1)
sh_ky = M[sl][:, [IDX[p] for p in KY7]].sum(axis=1) / M[sl].sum(axis=1)
sh_ok = M[sl][:, IDX['Okinawa']] / M[sl].sum(axis=1)
dd = dates[sl]
axB.plot(dd, sh_ky, color=C_KYUSHU, lw=2.0, marker='o', ms=4,
         label='Kyushu mainland (7 prefectures)')
axB.plot(dd, sh_ok, color=C_NORTH, lw=2.0, marker='s', ms=4, label='Okinawa')
axB.axhline(0.0453, color=C_NORTH, ls=':', lw=1.0)
axB.annotate('Okinawa share at 2026-W19\n(0.045; rising when the preprint was posted)',
             xy=(dd[0], 0.0453), xytext=(8, 16), textcoords='offset points',
             fontsize=7.5, color=C_NORTH,
             arrowprops=dict(arrowstyle='->', color=C_NORTH, lw=0.7))
kmax = int(np.argmax(sh_ky))
axB.annotate('%.3f' % sh_ky[kmax], xy=(dd[kmax], sh_ky[kmax]), xytext=(0, 8),
             textcoords='offset points', ha='center', fontsize=8,
             color=C_KYUSHU, fontweight='bold')
pk = dstr.index('2026-07-16')
axB.axvline(dates[pk], color='#555555', ls='--', lw=1.0)
axB.annotate('national peak\n2026-W29', xy=(dates[pk], 0.32), ha='right', va='top',
             fontsize=7.5, color='#555555', xytext=(-4, 0), textcoords='offset points')
axB.set_ylim(0, 0.60)
axB.set_ylabel('Share of national per-sentinel total')
axB.set_title('B. The Kyushu cascade returned without its Okinawa seed', loc='left')
tk = dd[::2]
axB.set_xticks(tk)
axB.set_xticklabels([epi(t) for t in tk], rotation=25, ha='right')
axB.legend(fontsize=8, loc='upper left')
axB.grid(True, alpha=0.25)

fig.tight_layout()
save(fig, 'Fig8_prospective_test.png')

print('\nValues reported in the figures')
print('  within-wave max S: ' + ', '.join('%s %.4f%s' % (r.label, r.smax, ' *' if r.smax >= CEIL else '')
                                   for r in WV.itertuples()))
print('  pre-transition amplitude %.2f-%.2f (%.1f-fold)' % (pr.amp.min(), pr.amp.max(), pr.amp.max() / pr.amp.min()))
print('  summer 2026 x = KLn-KLk: %.3f .. %.3f / pre-transition minimum %.3f'
      % (x_pp[sum26].min(), x_pp[sum26].max(), x_pp[pre].min()))
print('  summer 2026 weeks in the Mode A quadrant (x<0): %d/%d' % ((x_pp[sum26] < 0).sum(), sum26.sum()))
print('  max mainland-Kyushu share %.3f / max Okinawa share %.4f'
      % (sh_ky.max(), max(sh_ok[i] for i in range(len(dd)) if '2026-06-11' <= dstr[sl][i] <= '2026-07-16')
         if False else sh_ok.max()))
print('  Moran I: pre %.3f / post (first 24 wks) %.3f' % (I_t[pre].mean(), I_t[post_early].mean()))
