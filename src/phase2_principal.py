"""
Phase 2: Principal re-analysis with new framework
- Principal split: pre-transition (135 weeks, t < 2025-11-27) vs post-transition (24 weeks)
- AR(1) null included from the start
- K=1..6 sensitivity table
- AR(1)-corrected R²
- Block bootstrap as ONLY principal inferential statistic (Welch removed)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR, _RES_DIR, _FIG_DIR = str(_DATA_DIR), str(_RES_DIR), str(_FIG_DIR)

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

# ---------- Load ----------
DATA = os.path.join(_DATA_DIR, 'pref_full_n19.csv')
ADJ  = os.path.join(_DATA_DIR, 'adjacency_edges.csv')
df = pd.read_csv(DATA, parse_dates=['date'])
adj_df = pd.read_csv(ADJ)
PREFS = [c for c in df.columns if c not in ['date', 'S']]
dates = pd.to_datetime(df['date'])
P = df[PREFS].values
T = len(df)
N = len(PREFS)

# Adjacency
idx = {p:i for i,p in enumerate(PREFS)}
W = np.zeros((N, N))
for _, row in adj_df.iterrows():
    a, b = idx[row['Prefecture A']], idx[row['Prefecture B']]
    W[a,b] = 1; W[b,a] = 1
S0 = W.sum()

# Principal change-point (PELT-identified)
CP_DATE = pd.Timestamp('2025-11-27')
cp_idx = int(np.argmax(dates >= CP_DATE))
print(f'Principal change-point: {CP_DATE.date()} (week index {cp_idx})')
print(f'  pre-transition:  weeks 0..{cp_idx-1} = {cp_idx} weeks')
print(f'  post-transition: weeks {cp_idx}..{T-1} = {T-cp_idx} weeks')
pre  = np.arange(cp_idx)
post = np.arange(cp_idx, T)

# ---------- 1. Bivariate Moran's I (lagged) ----------
def bivariate_moran(x, y, W):
    xm = x.mean(); ym = y.mean()
    x_z = x - xm; y_z = y - ym
    den = np.sqrt((x_z**2).sum() * (y_z**2).sum())
    if den == 0: return np.nan
    return (N / S0) * (W * np.outer(x_z, y_z)).sum() / den

lags = [0, 1, 2, 4, 8]
I_lag = {tau: np.full(T, np.nan) for tau in lags}
for tau in lags:
    for t in range(tau, T):
        I_lag[tau][t] = bivariate_moran(P[t], P[t-tau], W)

# ---------- 2. New principal split — spatial-persistence ratio ----------
print('\n' + '='*72)
print('SPATIAL-PERSISTENCE RATIO I(τ=8)/I(0)  (new split: 2025-11-27)')
print('='*72)
# Only include weeks with I(0) > 0 to avoid noise division
print(f'{"τ":<4} {"pre I(τ) mean ± sd":<28} {"post I(τ) mean ± sd":<28} {"post/pre ratio":<10}')
print('-'*72)
for tau in lags:
    Ipre  = I_lag[tau][pre]
    Ipost = I_lag[tau][post]
    pre_m  = np.nanmean(Ipre);  pre_sd  = np.nanstd(Ipre)
    post_m = np.nanmean(Ipost); post_sd = np.nanstd(Ipost)
    print(f'{tau:<4} {pre_m:>+.3f} ± {pre_sd:.3f}  ({np.isfinite(Ipre).sum():>3})        '
          f'{post_m:>+.3f} ± {post_sd:.3f}  ({np.isfinite(Ipost).sum():>3})        '
          f'{post_m/max(pre_m, 1e-9):.2f}')

# Persistence ratio I(τ=8)/I(0), per-week then group
ratio_per_wk = I_lag[8] / I_lag[0]
ratio_per_wk[I_lag[0] <= 0] = np.nan  # exclude degenerate weeks
ratio_pre  = ratio_per_wk[pre]
ratio_post = ratio_per_wk[post]
ratio_pre = ratio_pre[np.isfinite(ratio_pre)]
ratio_post = ratio_post[np.isfinite(ratio_post)]
print(f'\nPersistence ratio I(τ=8)/I(0):')
print(f'  pre-transition  (n={len(ratio_pre):3d}): mean = {ratio_pre.mean():.3f}, sd = {ratio_pre.std():.3f}')
print(f'  post-transition (n={len(ratio_post):3d}): mean = {ratio_post.mean():.3f}, sd = {ratio_post.std():.3f}')
print(f'  raw ratio (post/pre): {ratio_post.mean() / max(ratio_pre.mean(), 1e-9):.2f}')

# ---------- 3. Moving block bootstrap (PRINCIPAL inferential statistic, only) ----------
def block_bootstrap_diff(x_pre, x_post, b_length, B=10000, seed=2026):
    """Block bootstrap test for difference in means between two segments.
    Resamples blocks within each segment to preserve serial dependence."""
    rng = np.random.default_rng(seed)
    obs_diff = x_post.mean() - x_pre.mean()
    n_pre, n_post = len(x_pre), len(x_post)
    diffs = np.zeros(B)
    for b in range(B):
        nb_pre  = n_pre  // b_length + 1
        nb_post = n_post // b_length + 1
        starts_pre  = rng.integers(0, max(1, n_pre  - b_length + 1), size=nb_pre)
        starts_post = rng.integers(0, max(1, n_post - b_length + 1), size=nb_post)
        boot_pre  = np.concatenate([x_pre[s:s+b_length] for s in starts_pre])[:n_pre]
        boot_post = np.concatenate([x_post[s:s+b_length] for s in starts_post])[:n_post]
        diffs[b] = boot_post.mean() - boot_pre.mean()
    # Block bootstrap typically used in 2 ways:
    # (1) for sampling distribution of the difference (using observed data)
    # (2) for null distribution (using a permutation-style version)
    # Here we test H0: same mean by computing the proportion of bootstrap
    # diffs that are >= observed diff in absolute value (two-sided p)
    # via the "studentized" approach:
    se_boot = diffs.std()
    z = obs_diff / se_boot if se_boot > 0 else 0.0
    # Combine: 2-sided p value
    p_2s = 2 * (1 - stats.norm.cdf(abs(z)))
    return obs_diff, se_boot, z, p_2s, diffs

print('\n' + '='*72)
print('MOVING BLOCK BOOTSTRAP — persistence ratio difference (post − pre)')
print('  (principal inferential statistic; two-sided p)')
print('='*72)
print(f'{"block b":<10} {"obs diff":<12} {"SE":<10} {"z":<8} {"p (2-sided)":<12}')
print('-'*55)
results_bbb = {}
for b_len in [1, 2, 4, 6, 8, 10, 12]:
    obs, se, z, p, _ = block_bootstrap_diff(ratio_pre, ratio_post, b_len, B=10000)
    results_bbb[b_len] = (obs, se, z, p)
    print(f'  b={b_len:<7d} {obs:>+.3f}      {se:.3f}     {z:+.2f}    {p:.4f}')

# ---------- 4. Harmonic regression with AR(1) AUGMENTED null (Cochrane-Orcutt) ----------
def cochrane_orcutt(y, X, max_iter=20, tol=1e-7):
    """Iterative Cochrane-Orcutt estimation of regression with AR(1) errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho = 0.0
    for _ in range(max_iter):
        resid = y - X @ beta
        # Estimate rho from residuals
        rho_new = np.sum(resid[1:] * resid[:-1]) / np.sum(resid[:-1]**2)
        if abs(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new
        # Transform variables
        y_t = y[1:] - rho * y[:-1]
        X_t = X[1:] - rho * X[:-1]
        beta, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
    # Final residuals
    resid = y - X @ beta
    return beta, rho, resid

DOY = dates.dt.dayofyear.values
omega = 2 * np.pi / 365.25
def harmonic_X(K):
    cols = [np.ones(T)]
    for k in range(1, K+1):
        cols.append(np.cos(k * omega * DOY))
        cols.append(np.sin(k * omega * DOY))
    return np.column_stack(cols)

# Compute S series (Shannon entropy)
S_t = -(P * np.log(P + 1e-30)).sum(axis=1)
I_t = np.array([bivariate_moran(P[t], P[t], W) for t in range(T)])

print('\n' + '='*72)
print('K SENSITIVITY (K=1..6, AR(1)-augmented errors via Cochrane-Orcutt)')
print('  Fit on pre-transition only (135 weeks)')
print('='*72)
print(f'{"K":<3} {"# params":<10} {"ρ̂":<8} {"R² (OLS)":<10} {"R² (AR-corr)":<14} {"AIC":<10} {"BIC":<10}')
print('-'*60)
results_K = {}
for K in [1, 2, 3, 4, 5, 6]:
    Xk = harmonic_X(K)
    y = S_t[pre]
    Xp = Xk[pre]
    beta_ols, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    resid_ols = y - Xp @ beta_ols
    rss_ols = np.sum(resid_ols**2)
    r2_ols = 1 - rss_ols / np.sum((y - y.mean())**2)
    # AR(1)-augmented
    beta_co, rho_co, resid_co = cochrane_orcutt(y, Xp)
    # AR-corrected R²: variance of *innovation* over variance of de-trended y
    # = 1 - var(innov) / var(y)
    # where innovation = (1-rho L) y - (1-rho L) Xb
    # already in resid_co? actually we need the post-transformation residual
    e = y[1:] - rho_co * y[:-1]  # transformed y
    Xte = Xp[1:] - rho_co * Xp[:-1]
    pred_t = Xte @ beta_co
    innov = e - pred_t
    r2_ar = 1 - np.var(innov) / np.var(e)
    n_eff = len(y)
    p_param = Xk.shape[1] + 1  # +1 for rho
    rss_co = np.sum(innov**2)
    n_co = len(innov)
    aic = n_co * np.log(rss_co / n_co) + 2 * p_param
    bic = n_co * np.log(rss_co / n_co) + p_param * np.log(n_co)
    results_K[K] = (rho_co, r2_ols, r2_ar, aic, bic)
    print(f'{K:<3} {p_param:<10} {rho_co:.3f}   {r2_ols:.3f}     {r2_ar:.3f}         {aic:.2f}   {bic:.2f}')

# ΔAIC, ΔBIC relative to K=2
print(f'\n{"K":<3} ΔAIC vs K=2   ΔBIC vs K=2')
aic_k2 = results_K[2][3]
bic_k2 = results_K[2][4]
for K, (rho, r2o, r2a, aic, bic) in results_K.items():
    print(f'{K:<3} {aic - aic_k2:+.2f}        {bic - bic_k2:+.2f}')

# ---------- 5. Save ----------
np.savez(os.path.join(_RES_DIR, 'phase2_principal_results.npz'),
         dates=df['date'].values,
         cp_idx=cp_idx,
         S=S_t, I_t=I_t,
         I_lag_0=I_lag[0], I_lag_1=I_lag[1], I_lag_2=I_lag[2],
         I_lag_4=I_lag[4], I_lag_8=I_lag[8],
         ratio_pre=ratio_pre, ratio_post=ratio_post,
         results_bbb=results_bbb,
         results_K=results_K,
         W=W)
print('\nSaved: phase2_principal_results.npz')
