"""
Regenerate Figures 1-6 for v3 manuscript:
- n=19 data (159 weeks, through 2026-05-07)
- Pre-transition: weeks 0..134 (2023-04-24 .. 2025-11-20)
- Post-transition: weeks 135..158 (2025-11-27 .. 2026-05-07)
- Change-point at 2025-11-27 (data-driven)
- Five pre-transition waves (Wave 1..Wave 5); no "Wave 6"
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR, _RES_DIR, _FIG_DIR = str(_DATA_DIR), str(_RES_DIR), str(_FIG_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
from scipy import stats
from epi_week_axis import date_to_epi, set_epi_week_xaxis

# ============================================================
# Load
DATA = os.path.join(_DATA_DIR, 'pref_full_n19.csv')
ADJ = os.path.join(_DATA_DIR, 'adjacency_edges.csv')
df = pd.read_csv(DATA, parse_dates=['date'])
adj_df = pd.read_csv(ADJ)
PREFS = [c for c in df.columns if c not in ['date', 'S']]
dates = pd.to_datetime(df['date'])
P = df[PREFS].values
T = len(df); N = len(PREFS)

CP_DATE = pd.Timestamp('2025-11-27')
cp_idx = int(np.argmax(dates >= CP_DATE))
pre = np.arange(cp_idx); post = np.arange(cp_idx, T)

# Adjacency
idx = {p:i for i,p in enumerate(PREFS)}
W = np.zeros((N, N))
for _, row in adj_df.iterrows():
    a, b = idx[row['Prefecture A']], idx[row['Prefecture B']]
    W[a,b] = 1; W[b,a] = 1
S0 = W.sum()

# Compute S(t), I(t), I(τ; t) needed by figures
S = -(P * np.log(P + 1e-30)).sum(axis=1)
LN47 = np.log(47)

def moran_t(P_t):
    z = P_t - P_t.mean()
    num = (W * np.outer(z, z)).sum()
    den = (z**2).sum()
    if den == 0: return np.nan
    return (N / S0) * num / den

I_t = np.array([moran_t(P[t]) for t in range(T)])

def bivariate_moran(x, y):
    xm = x.mean(); ym = y.mean()
    x_z = x - xm; y_z = y - ym
    den = np.sqrt((x_z**2).sum() * (y_z**2).sum())
    if den == 0: return np.nan
    return (N / S0) * (W * np.outer(x_z, y_z)).sum() / den

LAGS = [0, 1, 2, 4, 8]
I_lag = {tau: np.full(T, np.nan) for tau in LAGS}
for tau in LAGS:
    for t in range(tau, T):
        I_lag[tau][t] = bivariate_moran(P[t], P[t-tau])

# Regional KL
KYUSHU_OKI = ['Fukuoka','Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa']
NORTH = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima']
def regional_KL_signed(P_t, group):
    return sum(P_t[idx[p]] * np.log(47 * P_t[idx[p]] + 1e-30) for p in group)
KL_kyushu = np.array([regional_KL_signed(P[t], KYUSHU_OKI) for t in range(T)])
KL_north = np.array([regional_KL_signed(P[t], NORTH) for t in range(T)])

# Wave peaks (pre-transition only — 5 waves)
WAVE_PEAKS_DATES = [
    pd.Timestamp('2023-08-31'),
    pd.Timestamp('2024-02-01'),
    pd.Timestamp('2024-07-25'),
    pd.Timestamp('2025-01-09'),
    pd.Timestamp('2025-08-21'),
]
WAVE_LABELS = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']

# ============================================================
# Common figure style
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,   # 表示用。書き出しは 300 dpi + PDF（2026-08-21 に 180→300 へ引き上げ）
})

C_PRE = '#1f77b4'      # blue
C_POST = '#d62728'     # red
C_CP = '#d62728'       # red CP line
C_CEIL = '#888888'     # grey ceiling
C_KYUSHU = '#ff7f0e'   # orange (Mode A)
C_NORTH = '#2c79c2'    # blue (Mode B)

OUT = _FIG_DIR

# ============================================================
# FIGURE 1: Post-Class-5 wave template
# A: S(t) with 5 waves marked, CP line, post-transition shading
# B: Composite trace of S(t) aligned to 5 wave peaks
# ============================================================
fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 6.5))

# Panel A
axA.plot(dates, S, color=C_PRE, linewidth=1.0, label='S(t)')
axA.axhline(LN47, color=C_CEIL, linestyle=':', alpha=0.6, label='ln 47 ≈ 3.850')
# Mark 5 wave peaks
for d, lbl in zip(WAVE_PEAKS_DATES, WAVE_LABELS):
    axA.axvline(d, color='#aaaaaa', linestyle=':', linewidth=0.6, alpha=0.5)
    axA.annotate(lbl, xy=(d, 3.84), ha='center', fontsize=8,
                 color='#666666', fontweight='bold')
# Post-transition shading
axA.axvspan(dates[cp_idx], dates.iloc[-1], color=C_POST, alpha=0.10,
            label='Post-transition (n=24)')
# Change-point line
axA.axvline(dates[cp_idx], color=C_CP, linewidth=1.5, linestyle='--',
            label='Change-point: 2025-W48')
axA.set_ylabel('Pseudo-entropy S(t)')
axA.set_title('A. Pseudo-entropy time series', loc='left')
axA.legend(loc='lower left', fontsize=7)
axA.grid(True, alpha=0.3)
axA.set_ylim(3.50, 3.87)
set_epi_week_xaxis(axA, dates.iloc[0], dates.iloc[-1], n_ticks=8)

# Panel B: Composite S(t) aligned to wave peaks (pre-transition only)
window = 10  # ± weeks
composite = []
for peak_date in WAVE_PEAKS_DATES:
    peak_idx = int(np.argmin(np.abs((dates - peak_date).dt.total_seconds())))
    lo, hi = peak_idx - window, peak_idx + window + 1
    if lo < 0 or hi > T:
        continue
    composite.append(S[lo:hi])
composite = np.array(composite)
mean_curve = composite.mean(axis=0)
sem_curve = composite.std(axis=0) / np.sqrt(len(composite))
rel_weeks = np.arange(-window, window + 1)

axB.fill_between(rel_weeks, mean_curve - sem_curve, mean_curve + sem_curve,
                 color=C_PRE, alpha=0.25, label='±SEM (5 waves)')
axB.plot(rel_weeks, mean_curve, color=C_PRE, linewidth=2, label='Mean across 5 waves')
# Mark min / max relative position
min_idx = int(np.argmin(mean_curve))
max_idx = int(np.argmax(mean_curve))
axB.annotate(f'min at\n{rel_weeks[min_idx]:+d} wk',
             xy=(rel_weeks[min_idx], mean_curve[min_idx]),
             xytext=(-25, -25), textcoords='offset points',
             fontsize=8, ha='center')
axB.annotate(f'max at\n{rel_weeks[max_idx]:+d} wk',
             xy=(rel_weeks[max_idx], mean_curve[max_idx]),
             xytext=(15, 15), textcoords='offset points',
             fontsize=8, ha='center')
axB.axvline(0, color='k', linewidth=0.5, alpha=0.5)
axB.set_xlabel('Weeks relative to wave peak (case-count maximum)')
axB.set_ylabel('Pseudo-entropy S')
axB.set_title('B. Composite trace, pre-transition waves (n = 5)', loc='left')
axB.grid(True, alpha=0.3)
axB.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig1_wave_template.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/Fig1_wave_template.pdf', bbox_inches='tight')
plt.close()
print('Fig1 done.')

# ============================================================
# FIGURE 2: Two anti-phase modes — joint scatter S vs I, period-coded
# A: scatter
# B-D: simplified geographic vignettes (text only — abbreviated since full map regen is heavy)
# ============================================================
fig = plt.figure(figsize=(9, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.4, wspace=0.55)

axA = fig.add_subplot(gs[0, :])
axA.scatter(S[pre], I_t[pre], c=C_PRE, s=18, alpha=0.55, label=f'Pre-transition (n={len(pre)})')
axA.scatter(S[post], I_t[post], c=C_POST, s=22, alpha=0.85, label=f'Post-transition (n={len(post)})')

# Highlight 3 representative weeks
# (date format for picking the week, epi week for the label)
rep_dates = [
    ('Mode A', pd.Timestamp('2025-07-06'), '2025-W27', C_KYUSHU),
    ('Mode B', pd.Timestamp('2024-11-24'), '2024-W47', C_NORTH),
    ('Typical post-transition', pd.Timestamp('2026-03-08'), '2026-W10', '#a82020'),
]
# label placement: Mode B is moved into clear space below-left of its marker,
# where it does not sit on the post-transition points
LABEL_POS = {
    'Mode A':                  dict(xytext=(12, 8), textcoords='offset points'),
    'Mode B':                  dict(xytext=(3.560, 0.307), textcoords='data',
                                    ha='left', va='bottom'),
    'Typical post-transition': dict(xytext=(12, 8), textcoords='offset points'),
}
for label, dt, ew, col in rep_dates:
    ii = int(np.argmin(np.abs((dates - dt).dt.total_seconds())))
    axA.scatter(S[ii], I_t[ii], s=140, facecolors='none', edgecolors=col, linewidths=2)
    axA.annotate(f'{label}\n({ew})',
                xy=(S[ii], I_t[ii]),
                fontsize=8, color=col, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
                **LABEL_POS[label])
axA.set_xlabel('Pseudo-entropy S(t)')
axA.set_ylabel("Global Moran's I(t)")
axA.set_title('A. Joint scatter: pseudo-entropy vs. global Moran\'s I (159 weeks)', loc='left')
axA.grid(True, alpha=0.3)
axA.legend(loc='upper left')

# Panels B-D: simplified mini-summary panels (substitute for full maps)
for j, (label, dt, ew, col) in enumerate(rep_dates):
    ax = fig.add_subplot(gs[1, j])
    ii = int(np.argmin(np.abs((dates - dt).dt.total_seconds())))
    # Bar chart of top-10 prefectures by share
    shares = P[ii]
    top10 = np.argsort(shares)[-10:][::-1]
    names = [PREFS[k] for k in top10]
    vals = shares[top10]
    bars = ax.barh(range(len(names)), vals, color=col, alpha=0.75)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Share pᵢ(t)', fontsize=8)
    ax.set_title(f'{chr(ord("B")+j)}. {label}\n{ew}\nS={S[ii]:.3f}, I={I_t[ii]:+.3f}',
                loc='left', fontsize=8.5)
    ax.grid(True, alpha=0.3, axis='x')

plt.savefig(f'{OUT}/Fig2_two_modes.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/Fig2_two_modes.pdf', bbox_inches='tight')
plt.close()
print('Fig2 done.')

# ============================================================
# FIGURE 3: Seasonal anti-phase
# A: KL_north (blue, Mode B) and KL_kyushu (orange, Mode A) time series + AR(1) harmonic fits
# B: Epi-week-folded view
# C: Post-transition deviation
# ============================================================
from numpy.polynomial import polynomial as P_poly

# Compute epi week (ISO 8601 like, 1-53)
epi_weeks = np.array([d.isocalendar()[1] for d in dates])
omega = 2 * np.pi / 52.1775

# Fit K=2 OLS harmonic on pre-transition data
def fit_K2_OLS(y_pre, ew_pre):
    X = np.column_stack([
        np.ones(len(ew_pre)),
        np.cos(omega * ew_pre), np.sin(omega * ew_pre),
        np.cos(2 * omega * ew_pre), np.sin(2 * omega * ew_pre),
    ])
    beta, *_ = np.linalg.lstsq(X, y_pre, rcond=None)
    return beta

def predict_K2(beta, ew):
    X = np.column_stack([
        np.ones(len(ew)),
        np.cos(omega * ew), np.sin(omega * ew),
        np.cos(2 * omega * ew), np.sin(2 * omega * ew),
    ])
    return X @ beta

beta_kyushu = fit_K2_OLS(KL_kyushu[pre], epi_weeks[pre])
beta_north = fit_K2_OLS(KL_north[pre], epi_weeks[pre])
fit_kyushu = predict_K2(beta_kyushu, epi_weeks)
fit_north = predict_K2(beta_north, epi_weeks)

fig, axes = plt.subplots(3, 1, figsize=(9, 8))

# Panel A
ax = axes[0]
ax.plot(dates, KL_north, color=C_NORTH, linewidth=1.0, alpha=0.5, label='KL_north (Mode B)')
ax.plot(dates, KL_kyushu, color=C_KYUSHU, linewidth=1.0, alpha=0.5, label='KL_kyushu (Mode A)')
ax.plot(dates, fit_north, color=C_NORTH, linewidth=1.5, linestyle='--', label='K=2 OLS fit (B)')
ax.plot(dates, fit_kyushu, color=C_KYUSHU, linewidth=1.5, linestyle='--', label='K=2 OLS fit (A)')
ax.axvspan(dates[cp_idx], dates.iloc[-1], color=C_POST, alpha=0.08, label='Post-transition')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
ax.set_ylabel('Regional KL contribution')
ax.set_title('A. Regional KL contributions: time series', loc='left')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
ax.grid(True, alpha=0.3)
set_epi_week_xaxis(ax, dates.iloc[0], dates.iloc[-1], n_ticks=8)

# Panel B: Epi-week-folded
ax = axes[1]
ax.scatter(epi_weeks[pre], KL_north[pre], c=C_NORTH, s=12, alpha=0.5, label='KL_north (pre, observed)')
ax.scatter(epi_weeks[pre], KL_kyushu[pre], c=C_KYUSHU, s=12, alpha=0.5, label='KL_kyushu (pre, observed)')
ew_grid = np.linspace(1, 53, 200)
ax.plot(ew_grid, predict_K2(beta_north, ew_grid), color=C_NORTH, linewidth=2, label='K=2 OLS (B)')
ax.plot(ew_grid, predict_K2(beta_kyushu, ew_grid), color=C_KYUSHU, linewidth=2, label='K=2 OLS (A)')
# Mark peaks
ax.axvline(26, color=C_KYUSHU, linestyle=':', alpha=0.5)
ax.axvline(51, color=C_NORTH, linestyle=':', alpha=0.5)
ax.annotate('Mode A peak\nepi wk 26', xy=(26, 0.08), xytext=(0, -14.4), textcoords='offset points',
            fontsize=8, color=C_KYUSHU, fontweight='bold', ha='center')
# placed hard against the epi-week-51 marker; a light white patch keeps the x = 50
# gridline from running through the text
ax.annotate('Mode B peak\nepi wk 51', xy=(50.6, 0.295), fontsize=8, color=C_NORTH,
            fontweight='bold', ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0))
ax.set_xlabel('Epi week')
ax.set_ylabel('Regional KL contribution')
ax.set_title('B. Epi-week-folded view with K = 2 harmonic fits', loc='left')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 53)

# Panel C: Post-transition deviation
ax = axes[2]
dev_north = KL_north - fit_north
dev_kyushu = KL_kyushu - fit_kyushu
ax.plot(dates, dev_north, color=C_NORTH, linewidth=0.8, alpha=0.4)
ax.plot(dates, dev_kyushu, color=C_KYUSHU, linewidth=0.8, alpha=0.4)
ax.scatter(dates[post], dev_north[post], c=C_NORTH, s=18, alpha=0.85,
          label='KL_north (post): deviation')
ax.scatter(dates[post], dev_kyushu[post], c=C_KYUSHU, s=18, alpha=0.85,
          label='KL_kyushu (post): deviation')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
ax.axvline(dates[cp_idx], color=C_CP, linewidth=1.2, linestyle='--', alpha=0.7,
           label='CP: 2025-W48')
ax.set_xlabel('Epi week')
ax.set_ylabel('Observed − fitted')
ax.set_title('C. Post-transition deviation from pre-transition seasonal model', loc='left')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
ax.grid(True, alpha=0.3)
set_epi_week_xaxis(ax, dates.iloc[0], dates.iloc[-1], n_ticks=8)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig3_seasonal_antiphase.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/Fig3_seasonal_antiphase.pdf', bbox_inches='tight')
plt.close()
print('Fig3 done.')

# ============================================================
# FIGURE 4: Post-transition departure
# A: S(t) distribution pre vs post
# B: North vs Kyushu+Oki share over time
# C: 5-quantity comparison bar plot
# D: Moran's I time series with harmonic + 95% CI
# E: Residual distribution
# ============================================================
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1], hspace=0.85, wspace=0.32)

# A: S(t) distributions
axA = fig.add_subplot(gs[0, 0])
axA.hist(S[pre], bins=30, color=C_PRE, alpha=0.6, label=f'Pre (n={len(pre)})', density=True)
axA.hist(S[post], bins=15, color=C_POST, alpha=0.7, label=f'Post (n={len(post)})', density=True)
axA.axvline(S[pre].max(), color=C_PRE, linestyle='--', linewidth=1, label=f'Pre max = {S[pre].max():.3f}')
axA.axvline(S[post].max(), color=C_POST, linestyle='--', linewidth=1, label=f'Post max = {S[post].max():.3f}')
axA.axvline(LN47, color='k', linestyle=':', alpha=0.5)
axA.set_xlabel('Pseudo-entropy S(t)')
axA.set_ylabel('Density')
axA.set_title('A. Pseudo-entropy distribution by period', loc='left', pad=30)
axA.legend(fontsize=7, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
axA.grid(True, alpha=0.3)

# B: Aggregate share over time
axB = fig.add_subplot(gs[0, 1])
share_north = np.array([sum(P[t, idx[p]] for p in NORTH) for t in range(T)])
share_kyushu = np.array([sum(P[t, idx[p]] for p in KYUSHU_OKI) for t in range(T)])
# Population shares (approx)
pop_north = 7/47  # approximate
pop_kyushu = 8/47
axB.plot(dates, share_north, color=C_NORTH, linewidth=1, label='North Japan')
axB.plot(dates, share_kyushu, color=C_KYUSHU, linewidth=1, label='Kyushu+Okinawa')
axB.axhline(pop_north, color=C_NORTH, linestyle=':', alpha=0.5, label=f'N pop share ≈ {pop_north:.2f}')
axB.axhline(pop_kyushu, color=C_KYUSHU, linestyle=':', alpha=0.5, label=f'K+O pop share ≈ {pop_kyushu:.2f}')
axB.axvspan(dates[cp_idx], dates.iloc[-1], color=C_POST, alpha=0.08)
axB.axvline(dates[cp_idx], color=C_CP, linewidth=1, linestyle='--', alpha=0.7,
           label='CP: 2025-W48')
axB.set_ylabel('Regional share')
axB.set_title('B. Regional case share over time', loc='left', pad=48)
axB.legend(fontsize=7, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
axB.grid(True, alpha=0.3)
set_epi_week_xaxis(axB, dates.iloc[0], dates.iloc[-1], n_ticks=8)

# C: 5-quantity comparison
axC = fig.add_subplot(gs[1, 0])
quantities = ['Pre-trans', 'Post-trans']
metrics = ['S(t)', "I(t)", 'KL_north', 'KL_kyushu', 'I(τ=8)/I(0)']

# Compute persistence ratio
ratio = I_lag[8] / I_lag[0]
ratio[I_lag[0] <= 0] = np.nan
r_pre = ratio[pre]; r_pre = r_pre[np.isfinite(r_pre)]
r_post = ratio[post]; r_post = r_post[np.isfinite(r_post)]

means_pre = [S[pre].mean(), I_t[pre].mean(), KL_north[pre].mean(), KL_kyushu[pre].mean(), r_pre.mean()]
means_post = [S[post].mean(), I_t[post].mean(), KL_north[post].mean(), KL_kyushu[post].mean(), r_post.mean()]

x_pos = np.arange(len(metrics))
w = 0.35
b1 = axC.bar(x_pos - w/2, means_pre, w, color=C_PRE, alpha=0.8, label='Pre-transition')
b2 = axC.bar(x_pos + w/2, means_post, w, color=C_POST, alpha=0.8, label='Post-transition')
axC.set_xticks(x_pos)
axC.set_xticklabels(metrics, fontsize=7, rotation=25, ha='right')
axC.set_ylabel('Mean value')
axC.set_title('C. Between-period mean comparison', loc='left')
axC.legend(fontsize=7)
axC.axhline(0, color='k', linewidth=0.5)
axC.grid(True, alpha=0.3, axis='y')

# D: Moran's I trajectory
axD = fig.add_subplot(gs[1, 1])
# Fit K=2 OLS harmonic on I_t for pre-transition
beta_I = fit_K2_OLS(I_t[pre], epi_weeks[pre])
fit_I = predict_K2(beta_I, epi_weeks)
resid_I_pre = I_t[pre] - fit_I[pre]
resid_std = resid_I_pre.std()
axD.plot(dates, I_t, color='#666666', linewidth=1, alpha=0.6, label='Observed I(t)')
axD.plot(dates, fit_I, color=C_PRE, linewidth=1.5, linestyle='--', label='K=2 OLS fit (pre)')
axD.fill_between(dates, fit_I - 1.96*resid_std, fit_I + 1.96*resid_std,
                color=C_PRE, alpha=0.18, label='±1.96·σ residuals (pre)')
axD.scatter(dates[post], I_t[post], c=C_POST, s=20, alpha=0.85, label='Post-transition obs')
axD.axvline(dates[cp_idx], color=C_CP, linewidth=1, linestyle='--', alpha=0.7,
           label='CP: 2025-W48')
axD.set_ylabel("Global Moran's I(t)")
axD.set_title("D. Moran's I time series with harmonic model", loc='left', pad=48)
axD.legend(fontsize=7, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
axD.grid(True, alpha=0.3)
set_epi_week_xaxis(axD, dates.iloc[0], dates.iloc[-1], n_ticks=8)

# E: Residual distribution
axE = fig.add_subplot(gs[2, :])
resid_post = I_t[post] - fit_I[post]
axE.hist(resid_I_pre, bins=30, color=C_PRE, alpha=0.5, label=f'Pre residuals (n={len(pre)})', density=True)
axE.hist(resid_post, bins=15, color=C_POST, alpha=0.7, label=f'Post residuals (n={len(post)})', density=True)
axE.axvline(0, color='k', linewidth=0.5)
axE.axvline(resid_post.mean(), color=C_POST, linestyle='--', linewidth=1.5,
           label=f'Post mean = {resid_post.mean():+.3f}')
axE.set_xlabel('Residual from pre-transition K=2 harmonic')
axE.set_ylabel('Density')
axE.set_title('E. Residual distribution from pre-transition harmonic model', loc='left', pad=22)
axE.legend(fontsize=8, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
axE.grid(True, alpha=0.3)

plt.savefig(f'{OUT}/Fig4_regime_transition.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/Fig4_regime_transition.pdf', bbox_inches='tight')
plt.close()
print('Fig4 done.')

# ============================================================
# FIGURE 5: Phase-plane representation
# A: pre-transition coloured by epi week
# B: post-transition vs pre-transition scatter
# ============================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))

# Phase-plane coords
x_pp = KL_north - KL_kyushu

# A: pre coloured by epi week
cm = plt.cm.twilight
sc = axA.scatter(x_pp[pre], I_t[pre], c=epi_weeks[pre], cmap=cm,
                 s=22, alpha=0.85, vmin=1, vmax=53)
axA.axhline(0, color='k', linewidth=0.5, alpha=0.5)
axA.axvline(0, color='k', linewidth=0.5, alpha=0.5)
axA.set_xlabel('KL_north − KL_kyushu')
axA.set_ylabel("Global Moran's I")
axA.set_title('A. Pre-transition trajectory (135 wks), colour = epi week', loc='left')
axA.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=axA, shrink=0.7)
cbar.set_label('Epi week', fontsize=8)

# B: post vs pre
axB.scatter(x_pp[pre], I_t[pre], c='#bbbbbb', s=15, alpha=0.6, label='Pre-transition (n=135)')
axB.scatter(x_pp[post], I_t[post], c=C_POST, s=40, alpha=0.9, edgecolors='k', linewidths=0.5,
           label='Post-transition (n=24)')
axB.axhline(0, color='k', linewidth=0.5, alpha=0.5)
axB.axvline(0, color='k', linewidth=0.5, alpha=0.5)
# Quadrant labels
axB.annotate('Mode B-dominant', xy=(0.95, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=9, color=C_NORTH, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
axB.annotate('Mode A-dominant', xy=(0.05, 0.05), xycoords='axes fraction',
            ha='left', va='bottom', fontsize=9, color=C_KYUSHU, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
axB.set_xlabel('KL_north − KL_kyushu')
axB.set_ylabel("Global Moran's I")
axB.set_title('B. Post-transition (red) overlaid on pre-transition (grey)', loc='left')
axB.legend(fontsize=8, loc='lower right')
axB.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig5_phase_plane.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/Fig5_phase_plane.pdf', bbox_inches='tight')
plt.close()
print('Fig5 done.')

# ============================================================
# FIGURE 6: Time-lagged Moran's I — persistence profile
# A: Persistence profile I(τ)/I(0) at lags 0,1,2,4,8
# B: Mean I(τ) at each lag, by period
# ============================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))

# A: Persistence ratio at each lag, by period
ratios_pre = []
ratios_post = []
for tau in LAGS:
    r = I_lag[tau] / I_lag[0]
    r[I_lag[0] <= 0] = np.nan
    rp = r[pre]; rp = rp[np.isfinite(rp)]
    rpo = r[post]; rpo = rpo[np.isfinite(rpo)]
    ratios_pre.append((rp.mean(), rp.std() / np.sqrt(len(rp))))
    ratios_post.append((rpo.mean(), rpo.std() / np.sqrt(len(rpo))))

pre_m = np.array([x[0] for x in ratios_pre])
pre_se = np.array([x[1] for x in ratios_pre])
post_m = np.array([x[0] for x in ratios_post])
post_se = np.array([x[1] for x in ratios_post])

axA.errorbar(LAGS, pre_m, yerr=pre_se, marker='o', color=C_PRE, linewidth=2,
            markersize=8, capsize=4, label=f'Pre-transition (n={len(pre)})')
axA.errorbar(LAGS, post_m, yerr=post_se, marker='s', color=C_POST, linewidth=2,
            markersize=8, capsize=4, label=f'Post-transition (n={len(post)})')
axA.axhline(1, color='k', linestyle=':', alpha=0.5, label='Perfect persistence (=1)')
axA.axhline(0, color='k', linewidth=0.5, alpha=0.5)
axA.set_xlabel('Lag τ (weeks)')
axA.set_ylabel('Spatial-persistence ratio I(τ) / I(0)')
axA.set_title('A. Spatial-persistence profile (ratio of mean I)', loc='left', pad=44)
axA.legend(fontsize=9, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
axA.grid(True, alpha=0.3)
axA.set_xticks(LAGS)

# Annotate the τ=8 values
axA.annotate(f'{pre_m[-1]:.2f}', xy=(8, pre_m[-1]), xytext=(-13, -7),
            textcoords='offset points', ha='right', va='center', fontsize=9, color=C_PRE, fontweight='bold')
axA.annotate(f'{post_m[-1]:.2f}', xy=(8, post_m[-1]), xytext=(-13, -9),
            textcoords='offset points', ha='right', va='center', fontsize=9, color=C_POST, fontweight='bold')

# B: Mean Moran's I at each lag, by period
means_pre_I = [I_lag[tau][pre][np.isfinite(I_lag[tau][pre])].mean() for tau in LAGS]
se_pre_I = [I_lag[tau][pre][np.isfinite(I_lag[tau][pre])].std() / np.sqrt(np.isfinite(I_lag[tau][pre]).sum()) for tau in LAGS]
means_post_I = [I_lag[tau][post][np.isfinite(I_lag[tau][post])].mean() for tau in LAGS]
se_post_I = [I_lag[tau][post][np.isfinite(I_lag[tau][post])].std() / np.sqrt(np.isfinite(I_lag[tau][post]).sum()) for tau in LAGS]

axB.errorbar(LAGS, means_pre_I, yerr=se_pre_I, marker='o', color=C_PRE, linewidth=2,
            markersize=8, capsize=4, label=f'Pre-transition (n={len(pre)})')
axB.errorbar(LAGS, means_post_I, yerr=se_post_I, marker='s', color=C_POST, linewidth=2,
            markersize=8, capsize=4, label=f'Post-transition (n={len(post)})')
axB.axhline(0, color='k', linewidth=0.5, alpha=0.5)
axB.set_xlabel('Lag τ (weeks)')
axB.set_ylabel("Mean Moran's I(τ)")
axB.set_title('B. Time-lagged bivariate Moran\'s I by period', loc='left', pad=26)
axB.legend(fontsize=9, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
axB.grid(True, alpha=0.3)
axB.set_xticks(LAGS)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig6_lagged_moran.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/Fig6_lagged_moran.pdf', bbox_inches='tight')
plt.close()
print('Fig6 done.')

print('\nAll figures regenerated in', OUT)
