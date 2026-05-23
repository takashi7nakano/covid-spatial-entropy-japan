"""
Phase 1.6: Robustness checks for the late-2025 change-point in S(t).
1. PELT penalty sensitivity
2. Bayesian online change-point posterior
3. Smoothed series check
4. Confidence interval via bootstrap
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR, _RES_DIR, _FIG_DIR = str(_DATA_DIR), str(_RES_DIR), str(_FIG_DIR)

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

# Load
data = np.load(os.path.join(_RES_DIR, 'phase1_changepoint_results.npz'), allow_pickle=True)
dates = pd.to_datetime(data['dates'])
S = data['S']
I_t = data['I_t']
X = data['X_harmonic']
T = len(S)

# ---------- 1. PELT penalty sensitivity ----------
print('='*70)
print('1. PELT penalty sensitivity (S series)')
print('='*70)
sigma2 = np.var(S)
print(f'sigma² = {sigma2:.5f}, log(n) = {np.log(T):.3f}, BIC base = 2σ²log(n) = {2*sigma2*np.log(T):.4f}')
for factor in [0.5, 1.0, 2.0, 4.0, 8.0]:
    pen = factor * sigma2 * np.log(T)
    algo = rpt.Pelt(model='l2', min_size=15).fit(S)
    cps = algo.predict(pen=pen)
    cps = [c for c in cps if c < T]
    cp_str = ', '.join([f'{dates[c].date()}' for c in cps]) if cps else 'none'
    print(f'  pen = {factor:4.1f} × σ²log(n) = {pen:.4f}: breakpoints = [{cp_str}]')

# ---------- 2. Bayesian online change-point (Adams & MacKay 2007) ----------
print('\n' + '='*70)
print('2. Bayesian Online Change-Point Detection (BOCPD)')
print('='*70)

def bocpd_normal(y, lam=100.0, alpha0=1.0, beta0=1.0, kappa0=1.0, mu0=None):
    """
    Bayesian Online Change-Point Detection (Adams & MacKay 2007)
    with Gaussian likelihood, conjugate Normal-Inverse-Gamma prior.
    Hazard: 1/lam constant.
    Returns run-length posterior.
    """
    T = len(y)
    if mu0 is None:
        mu0 = y.mean()
    R = np.zeros((T+1, T+1))
    R[0, 0] = 1.0
    # Sufficient stats: keep per-run-length means and variances
    mu = np.full(T+1, mu0)
    kappa = np.full(T+1, kappa0)
    alpha = np.full(T+1, alpha0)
    beta = np.full(T+1, beta0)
    for t in range(T):
        x = y[t]
        # Predictive prob (Student-t)
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = stats.t.pdf(x, df, loc=mu, scale=scale)
        # Growth probabilities (run-length grows by 1)
        growth = R[t, :t+1] * pred[:t+1] * (1 - 1/lam)
        # Change-point probability (run-length resets to 0)
        cp_prob = (R[t, :t+1] * pred[:t+1] * (1/lam)).sum()
        # Update R
        R[t+1, 1:t+2] = growth
        R[t+1, 0] = cp_prob
        # Normalize
        R[t+1, :] /= R[t+1, :].sum()
        # Update sufficient stats (shift)
        mu_new = np.zeros(t+2)
        kappa_new = np.zeros(t+2)
        alpha_new = np.zeros(t+2)
        beta_new = np.zeros(t+2)
        mu_new[0] = mu0
        kappa_new[0] = kappa0
        alpha_new[0] = alpha0
        beta_new[0] = beta0
        for r in range(t+1):
            kappa_new[r+1] = kappa[r] + 1
            mu_new[r+1] = (kappa[r]*mu[r] + x) / kappa_new[r+1]
            alpha_new[r+1] = alpha[r] + 0.5
            beta_new[r+1] = beta[r] + 0.5 * kappa[r]*(x - mu[r])**2 / (kappa[r]+1)
        mu = mu_new[:T+1] if t+2 > T+1 else np.pad(mu_new, (0, max(0,T+1-len(mu_new))), mode='edge')[:T+1]
        kappa = kappa_new[:T+1] if t+2 > T+1 else np.pad(kappa_new, (0, max(0,T+1-len(kappa_new))), mode='edge')[:T+1]
        alpha = alpha_new[:T+1] if t+2 > T+1 else np.pad(alpha_new, (0, max(0,T+1-len(alpha_new))), mode='edge')[:T+1]
        beta = beta_new[:T+1] if t+2 > T+1 else np.pad(beta_new, (0, max(0,T+1-len(beta_new))), mode='edge')[:T+1]
    return R

# Use S residuals (de-seasonalized) for BOCPD
resid_S = data['resid_S']
R = bocpd_normal(resid_S, lam=80.0, alpha0=1.0, beta0=np.var(resid_S))
# Probability of change-point at each time = R[t+1, 0]
cp_prob = R[1:, 0]
top5 = np.argsort(cp_prob)[-5:][::-1]
print('  Top-5 weeks with highest change-point posterior probability (S residuals):')
for rank, t in enumerate(top5, 1):
    print(f'    #{rank}: t={t:3d} ({dates[t].date()}), P(CP) = {cp_prob[t]:.4f}')

# ---------- 3. Smoothed series ----------
print('\n' + '='*70)
print('3. Smoothed S(t) (4-week MA) → PELT on smoothed')
print('='*70)
window = 4
S_smooth = pd.Series(S).rolling(window, center=True).mean().bfill().ffill().values
for factor in [0.5, 1.0, 2.0, 4.0]:
    pen = factor * np.var(S_smooth) * np.log(T)
    algo = rpt.Pelt(model='l2', min_size=15).fit(S_smooth)
    cps = algo.predict(pen=pen)
    cps = [c for c in cps if c < T]
    cp_str = ', '.join([f'{dates[c].date()}' for c in cps]) if cps else 'none'
    print(f'  pen = {factor:.1f} × σ²log(n): [{cp_str}]')

# ---------- 4. Confidence interval via block bootstrap ----------
print('\n' + '='*70)
print('4. Block-bootstrap CI for PELT-detected change-point (S series)')
print('='*70)

# Detect on original
pen_main = 2 * np.var(S) * np.log(T)
algo = rpt.Pelt(model='l2', min_size=15).fit(S)
cps_main = [c for c in algo.predict(pen=pen_main) if c < T]
cp_main = cps_main[0] if cps_main else None
print(f'  Original PELT: t* = {cp_main} ({dates[cp_main].date() if cp_main is not None else "N/A"})')

# Bootstrap
rng = np.random.default_rng(2026)
B = 500
b_size = 8  # block length (matches lag-8 in original paper)
cp_distribution = []
n_blocks = T // b_size
for b in range(B):
    # Block bootstrap
    block_idx = rng.integers(0, T - b_size + 1, size=n_blocks)
    S_boot = np.concatenate([S[i:i+b_size] for i in block_idx])[:T]
    try:
        cps_b = rpt.Pelt(model='l2', min_size=15).fit(S_boot).predict(pen=pen_main)
        cps_b = [c for c in cps_b if c < T]
        if cps_b:
            cp_distribution.append(cps_b[0])
    except Exception:
        pass

if cp_distribution:
    cp_arr = np.array(cp_distribution)
    print(f'  Bootstrap distribution (B={len(cp_distribution)}):')
    print(f'    median t* = {int(np.median(cp_arr))} ({dates[int(np.median(cp_arr))].date()})')
    print(f'    95% CI: [{int(np.percentile(cp_arr, 2.5))}, {int(np.percentile(cp_arr, 97.5))}]')
    print(f'           [{dates[int(np.percentile(cp_arr, 2.5))].date()}, {dates[int(np.percentile(cp_arr, 97.5))].date()}]')

# ---------- 5. Three-method consensus figure ----------
print('\n' + '='*70)
print('5. Building consensus figure...')
print('='*70)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# Panel 1: S(t) with all candidate change-points
ax = axes[0]
ax.plot(dates, S, 'b-', linewidth=1.2, label='S(t)')
ax.plot(dates, S_smooth, 'b-', linewidth=2, alpha=0.4, label='4-week MA')
ax.axvline(dates[cp_main] if cp_main else dates[0], color='red', linestyle='--',
           linewidth=2, label=f'PELT: {dates[cp_main].date() if cp_main else "N/A"}')
ax.axvline(dates[109], color='green', linestyle=':', alpha=0.6, label='CUSUM: 2025-05-29')
ax.axvline(dates[112], color='purple', linestyle=':', alpha=0.6, label='Bai-Perron sup-F: 2025-06-19')
ax.set_ylabel('Shannon entropy S(t)', fontsize=11)
ax.set_title(f'Data-driven change-point detection (n=19, {T} weeks)', fontsize=11)
ax.legend(loc='lower left', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: BOCPD posterior
ax = axes[1]
ax.plot(dates, cp_prob, 'navy', linewidth=1)
ax.fill_between(dates, 0, cp_prob, alpha=0.3, color='navy')
ax.set_ylabel('P(change-point)\nposterior (BOCPD)', fontsize=11)
ax.axvline(dates[cp_main] if cp_main else dates[0], color='red', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(cp_prob.max() * 1.1, 0.05))

# Panel 3: bootstrap histogram
ax = axes[2]
if cp_distribution:
    counts, edges = np.histogram(cp_distribution, bins=30, range=(0, T))
    bin_dates = [dates[int(e)] if int(e) < T else dates[-1] for e in (edges[:-1]+edges[1:])/2]
    ax.bar(bin_dates, counts, width=20, color='steelblue', alpha=0.7)
    ax.axvline(dates[cp_main] if cp_main else dates[0], color='red', linestyle='--', label='Point estimate')
    ax.axvline(dates[int(np.percentile(cp_arr, 2.5))], color='black', linestyle=':',
               alpha=0.5, label='95% CI')
    ax.axvline(dates[int(np.percentile(cp_arr, 97.5))], color='black', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)
ax.set_ylabel('Bootstrap count', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.grid(True, alpha=0.3)

axes[-1].xaxis.set_major_locator(MonthLocator(bymonth=[1,4,7,10]))
axes[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(_FIG_DIR, 'phase1_robustness.png'), dpi=150, bbox_inches='tight')
print('Figure saved: phase1_robustness.png')
