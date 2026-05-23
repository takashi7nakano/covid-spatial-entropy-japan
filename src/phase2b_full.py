"""
Phase 2b:
- K-sensitivity for Moran's I, KL_kyushu, KL_north (the paper's principal series)
- Phase coherence verification on new pre-transition period (135 weeks)
- 95% CI for persistence ratio difference via percentile bootstrap
- Leave-one-prefecture-out (K=2 vs K=3 advantage source)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR, _RES_DIR, _FIG_DIR = str(_DATA_DIR), str(_RES_DIR), str(_FIG_DIR)

import numpy as np
import pandas as pd
from scipy import stats

# Load
DATA = os.path.join(_DATA_DIR, 'pref_full_n19.csv')
ADJ  = os.path.join(_DATA_DIR, 'adjacency_edges.csv')
df = pd.read_csv(DATA, parse_dates=['date'])
adj_df = pd.read_csv(ADJ)
PREFS = [c for c in df.columns if c not in ['date', 'S']]
dates = pd.to_datetime(df['date'])
P = df[PREFS].values
T = len(df); N = len(PREFS)

idx = {p:i for i,p in enumerate(PREFS)}
W = np.zeros((N, N))
for _, row in adj_df.iterrows():
    a, b = idx[row['Prefecture A']], idx[row['Prefecture B']]
    W[a,b] = 1; W[b,a] = 1
S0 = W.sum()

CP_DATE = pd.Timestamp('2025-11-27')
cp_idx = int(np.argmax(dates >= CP_DATE))
pre = np.arange(cp_idx)
post = np.arange(cp_idx, T)
print(f'cp = {dates[cp_idx].date()}, pre = {cp_idx} wk, post = {T-cp_idx} wk')

# ---------- Time series for K-sensitivity ----------
def univariate_moran_t(P_t):
    z = P_t - P_t.mean()
    den = (z**2).sum() / N
    num = (W * np.outer(z, z)).sum()
    return num / den / S0 * N

I_t = np.array([univariate_moran_t(P[t]) for t in range(T)])

# KL decompositions (Kyushu+Okinawa and Northern Japan)
KYUSHU_OKI = ['Fukuoka','Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa']
NORTH = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima']
# population weights (approx, used for baseline)
# Simplified: use mean shares as baseline
baseline = P.mean(axis=0)
baseline = baseline / baseline.sum()

def regional_KL(P_t, group, baseline_shares):
    s_group = sum(P_t[idx[p]] for p in group)
    s_other = 1 - s_group
    b_group = sum(baseline_shares[idx[p]] for p in group)
    b_other = 1 - b_group
    eps = 1e-12
    return s_group * np.log((s_group+eps)/(b_group+eps)) + s_other * np.log((s_other+eps)/(b_other+eps))

KL_kyushu = np.array([regional_KL(P[t], KYUSHU_OKI, baseline) for t in range(T)])
KL_north  = np.array([regional_KL(P[t], NORTH, baseline) for t in range(T)])

S_t = -(P * np.log(P + 1e-30)).sum(axis=1)

# ---------- AR(1)-augmented harmonic regression: K sensitivity for each series ----------
DOY = dates.dt.dayofyear.values
omega = 2 * np.pi / 365.25
def harmonic_X(K):
    cols = [np.ones(T)]
    for k in range(1, K+1):
        cols.append(np.cos(k * omega * DOY))
        cols.append(np.sin(k * omega * DOY))
    return np.column_stack(cols)

def cochrane_orcutt(y, X, max_iter=20, tol=1e-7):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho = 0.0
    for _ in range(max_iter):
        resid = y - X @ beta
        rho_new = np.sum(resid[1:] * resid[:-1]) / np.sum(resid[:-1]**2)
        if abs(rho_new - rho) < tol:
            rho = rho_new; break
        rho = rho_new
        y_t = y[1:] - rho * y[:-1]
        X_t = X[1:] - rho * X[:-1]
        beta, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
    resid = y - X @ beta
    return beta, rho, resid

def k_sensitivity(series, label):
    print(f'\n--- K sensitivity for {label} (pre-transition, n={len(pre)}) ---')
    y = series[pre]
    print(f'{"K":<3} {"p":<4} {"ρ̂":<8} {"R²(OLS)":<10} {"R²(AR)":<10} {"AIC":<10} {"BIC":<10}')
    rows = {}
    for K in [1, 2, 3, 4, 5, 6]:
        Xk = harmonic_X(K)
        Xp = Xk[pre]
        beta_ols, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        rss_ols = np.sum((y - Xp @ beta_ols)**2)
        r2_ols = 1 - rss_ols / np.sum((y - y.mean())**2)
        beta_co, rho_co, _ = cochrane_orcutt(y, Xp)
        e = y[1:] - rho_co * y[:-1]
        Xte = Xp[1:] - rho_co * Xp[:-1]
        innov = e - Xte @ beta_co
        r2_ar = 1 - np.var(innov) / np.var(e)
        p_param = Xk.shape[1] + 1
        rss_co = np.sum(innov**2)
        n_co = len(innov)
        aic = n_co * np.log(rss_co / n_co) + 2 * p_param
        bic = n_co * np.log(rss_co / n_co) + p_param * np.log(n_co)
        rows[K] = (rho_co, r2_ols, r2_ar, aic, bic)
        print(f'{K:<3} {p_param:<4} {rho_co:+.3f}  {r2_ols:+.3f}    {r2_ar:+.3f}     {aic:+.2f}   {bic:+.2f}')
    aic_k2 = rows[2][3]; bic_k2 = rows[2][4]
    print(f'ΔAIC, ΔBIC vs K=2:')
    for K, (rho, r2o, r2a, aic, bic) in rows.items():
        print(f'  K={K}: ΔAIC = {aic-aic_k2:+.2f}, ΔBIC = {bic-bic_k2:+.2f}')
    return rows

results_S    = k_sensitivity(S_t,       'S(t) entropy')
results_I    = k_sensitivity(I_t,       "Moran's I(t)")
results_KK   = k_sensitivity(KL_kyushu, 'KL_kyushu')
results_KN   = k_sensitivity(KL_north,  'KL_north')

# ---------- Phase coherence on new pre-transition (S.9 re-verification) ----------
print('\n' + '='*72)
print('PHASE COHERENCE (S.9 re-verified on new pre-transition = 135 weeks)')
print('='*72)
def fit_K3_phases(y_pre, X3_pre):
    """Fit K=3 harmonic, return DOY of k=1 and k=3 peaks."""
    beta, *_ = np.linalg.lstsq(X3_pre, y_pre, rcond=None)
    # beta = [c0, a1, b1, a2, b2, a3, b3]
    # Peak DOY of k-th harmonic: arctan2(b_k, a_k) / (k*omega) → conver to DOY
    a1, b1 = beta[1], beta[2]
    a3, b3 = beta[5], beta[6]
    A1 = np.sqrt(a1**2 + b1**2)
    A3 = np.sqrt(a3**2 + b3**2)
    # Peak: maximum of A cos(kω(DOY - DOY_peak)) is at DOY_peak where
    # cos(kω DOY_peak) = a / A , sin(kω DOY_peak) = b / A
    # → DOY_peak = arctan2(b, a) / (k·ω); take positive modulo 365/k
    doy_k1 = (np.arctan2(b1, a1) / omega) % 365.25
    doy_k3 = (np.arctan2(b3, a3) / (3*omega)) % (365.25/3)
    # k=3 has 3 peaks within a year; report the one closest to k=1 peak
    candidates = [(doy_k3 + i*365.25/3) % 365.25 for i in range(3)]
    diffs = [(c - doy_k1 + 365.25/2) % 365.25 - 365.25/2 for c in candidates]
    closest_idx = int(np.argmin(np.abs(diffs)))
    return doy_k1, candidates[closest_idx], diffs[closest_idx], A1, A3

X3_full = harmonic_X(3)
X3_pre = X3_full[pre]
for label, series in [("KL_kyushu", KL_kyushu),
                      ("KL_north", KL_north),
                      ("Moran's I", I_t)]:
    doy1, doy3, diff, A1, A3 = fit_K3_phases(series[pre], X3_pre)
    d1 = pd.Timestamp('2026-01-01') + pd.Timedelta(days=doy1-1)
    d3 = pd.Timestamp('2026-01-01') + pd.Timedelta(days=doy3-1)
    print(f'  {label}:')
    print(f'    k=1 peak: DOY {doy1:.0f} ({d1.strftime("%d %b")}), A1 = {A1:.4f}')
    print(f'    k=3 peak: DOY {doy3:.0f} ({d3.strftime("%d %b")}), A3 = {A3:.4f}')
    print(f'    offset:   {diff:+.1f} days  ({"coherent" if abs(diff) < 15 else "incoherent"})')

# ---------- 95% CI for persistence ratio difference (percentile bootstrap) ----------
print('\n' + '='*72)
print('95% CI for persistence ratio difference (post − pre), percentile bootstrap')
print('='*72)
# Reload ratio data from phase 2
def bivariate_moran(x, y, W, S0, N):
    xm = x.mean(); ym = y.mean()
    x_z = x - xm; y_z = y - ym
    den = np.sqrt((x_z**2).sum() * (y_z**2).sum())
    if den == 0: return np.nan
    return (N / S0) * (W * np.outer(x_z, y_z)).sum() / den

I_lag0 = np.array([bivariate_moran(P[t], P[t], W, S0, N) for t in range(T)])
I_lag8 = np.array([np.nan]*8 + [bivariate_moran(P[t], P[t-8], W, S0, N) for t in range(8, T)])

ratio = I_lag8 / I_lag0
ratio[I_lag0 <= 0] = np.nan
r_pre = ratio[pre]; r_pre = r_pre[np.isfinite(r_pre)]
r_post = ratio[post]; r_post = r_post[np.isfinite(r_post)]

# Block bootstrap of mean difference, with 95% CI
rng = np.random.default_rng(2026)
B = 10000
b_length = 8
n_pre = len(r_pre); n_post = len(r_post)

# Calculate observed
obs_diff = r_post.mean() - r_pre.mean()
obs_ratio = r_post.mean() / max(r_pre.mean(), 1e-9)
print(f'Observed mean difference (post - pre): {obs_diff:+.3f}')
print(f'Observed ratio (post / pre):           {obs_ratio:.2f}×')

# Block bootstrap for CI
diffs = np.zeros(B)
ratios = np.zeros(B)
for b in range(B):
    starts_pre  = rng.integers(0, max(1, n_pre  - b_length + 1), size=n_pre  // b_length + 1)
    starts_post = rng.integers(0, max(1, n_post - b_length + 1), size=n_post // b_length + 1)
    bp = np.concatenate([r_pre[s:s+b_length] for s in starts_pre])[:n_pre]
    bo = np.concatenate([r_post[s:s+b_length] for s in starts_post])[:n_post]
    diffs[b] = bo.mean() - bp.mean()
    ratios[b] = bo.mean() / max(bp.mean(), 1e-9)

print(f'\nBlock bootstrap (b={b_length}, B={B}):')
print(f'  Difference: mean = {diffs.mean():+.3f}')
print(f'  95% CI for difference: [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]')
print(f'  95% CI for ratio:      [{np.percentile(ratios, 2.5):.2f}×, {np.percentile(ratios, 97.5):.2f}×]')

# ---------- Leave-one-prefecture-out for K=2 vs K=3 source ----------
print('\n' + '='*72)
print('LEAVE-ONE-PREFECTURE-OUT: K=3 advantage source (Moran I)')
print('  All 47 prefs excluded one at a time; report ΔAIC(K3 vs K2) ranking')
print('='*72)
def compute_dAIC_excluding(pref):
    """Drop one prefecture, renormalize, recompute Moran I_t, compute ΔAIC(K3 vs K2)."""
    j = idx[pref]
    P_sub = np.delete(P, j, axis=1)
    P_sub = P_sub / P_sub.sum(axis=1, keepdims=True)
    W_sub = np.delete(np.delete(W, j, axis=0), j, axis=1)
    S0_sub = W_sub.sum()
    N_sub = N - 1
    # Recompute Moran's I
    I_sub = np.zeros(T)
    for t in range(T):
        z = P_sub[t] - P_sub[t].mean()
        den = (z**2).sum() / N_sub
        num = (W_sub * np.outer(z, z)).sum()
        I_sub[t] = num / den / S0_sub * N_sub
    # Fit K=2 and K=3 with AR(1) on pre-transition
    y = I_sub[pre]
    X2 = harmonic_X(2)[pre]
    X3 = harmonic_X(3)[pre]
    _, rho2, _ = cochrane_orcutt(y, X2)
    _, rho3, _ = cochrane_orcutt(y, X3)
    # AIC for each (transformed)
    def aic_ar(y, X, rho):
        beta, *_ = np.linalg.lstsq(X[1:] - rho*X[:-1], y[1:] - rho*y[:-1], rcond=None)
        e = (y[1:] - rho*y[:-1]) - (X[1:] - rho*X[:-1]) @ beta
        n_e = len(e)
        rss = np.sum(e**2)
        p = X.shape[1] + 1
        return n_e * np.log(rss/n_e) + 2 * p
    aic2 = aic_ar(y, X2, rho2)
    aic3 = aic_ar(y, X3, rho3)
    return aic3 - aic2

dAIC = {p: compute_dAIC_excluding(p) for p in PREFS}
# Lower ΔAIC = K=3 still preferred, higher = K=2 catches up after removal
sorted_prefs = sorted(dAIC.items(), key=lambda kv: kv[1])
print('\nPrefectures whose REMOVAL most reduces K=3 advantage (top-5):')
print('(higher ΔAIC after removal = the prefecture was driving K=3 advantage)')
for p, dA in sorted_prefs[-5:]:
    print(f'  {p:<12} ΔAIC(K3 vs K2) after removal = {dA:+.2f}')
print('\nPrefectures whose REMOVAL most strengthens K=3 advantage (bottom-5):')
for p, dA in sorted_prefs[:5]:
    print(f'  {p:<12} ΔAIC(K3 vs K2) after removal = {dA:+.2f}')

# Save everything
np.savez(os.path.join(_RES_DIR, 'phase2b_results.npz'),
         dates=df['date'].values, cp_idx=cp_idx,
         S=S_t, I_t=I_t, KL_kyushu=KL_kyushu, KL_north=KL_north,
         I_lag0=I_lag0, I_lag8=I_lag8,
         r_pre=r_pre, r_post=r_post,
         obs_diff=obs_diff, obs_ratio=obs_ratio,
         ci_diff=[np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)],
         ci_ratio=[np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)],
         results_S=results_S, results_I=results_I,
         results_KK=results_KK, results_KN=results_KN,
         dAIC_LOPO=dAIC)
print('\nSaved: phase2b_results.npz')
