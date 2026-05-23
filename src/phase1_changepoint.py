"""
Phase 1: Data-driven change-point detection (n=19, 159 weeks)
Three methods: CUSUM, PELT, Bai-Perron
Series tested: S(t), Moran's I(t), residuals from harmonic regression
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA as _DATA_DIR, RESULTS as _RES_DIR, FIGURES as _FIG_DIR
_DATA_DIR, _RES_DIR, _FIG_DIR = str(_DATA_DIR), str(_RES_DIR), str(_FIG_DIR)

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

# ---------- 1. Load data ----------
DATA = os.path.join(_DATA_DIR, 'pref_full_n19.csv')
ADJ = os.path.join(_DATA_DIR, 'adjacency_edges.csv')

df = pd.read_csv(DATA, parse_dates=['date'])
adj_df = pd.read_csv(ADJ)
pref_cols = [c for c in df.columns if c not in ['date', 'S']]
N_WEEKS = len(df)
print(f'Loaded n=19 data: {N_WEEKS} weeks, {df.date.min().date()} → {df.date.max().date()}')

# ---------- 2. Build adjacency matrix W (Queen, row-standardized) ----------
n = len(pref_cols)
idx = {p: i for i, p in enumerate(pref_cols)}
W = np.zeros((n, n))
for _, row in adj_df.iterrows():
    a, b = idx[row['Prefecture A']], idx[row['Prefecture B']]
    W[a, b] = 1
    W[b, a] = 1
# Row-standardize
W_rs = W / W.sum(axis=1, keepdims=True)
print(f'Adjacency: {int(W.sum()/2)} undirected edges, {n} prefectures')

# ---------- 3. Compute time series ----------
shares = df[pref_cols].values  # (T, 47)

# Shannon entropy
S = -(shares * np.log(shares + 1e-30)).sum(axis=1)

# Moran's I per week
def morans_I(x, W_rs):
    z = x - x.mean()
    num = (W_rs * np.outer(z, z)).sum()
    den = (z**2).sum() / len(z)
    return num / den / W_rs.sum() * len(z)

I_t = np.array([morans_I(shares[t], W_rs) for t in range(N_WEEKS)])

# Regional shares
NORTH = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima']
KYUSHU_OKI = ['Fukuoka','Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa']
share_north = df[NORTH].sum(axis=1).values
share_kyushu = df[KYUSHU_OKI].sum(axis=1).values

print(f'\nTime series ranges:')
print(f'  S:           [{S.min():.4f}, {S.max():.4f}]')
print(f'  Moran I:     [{I_t.min():.4f}, {I_t.max():.4f}]')
print(f'  North share: [{share_north.min():.4f}, {share_north.max():.4f}]')
print(f'  Kyushu+Oki:  [{share_kyushu.min():.4f}, {share_kyushu.max():.4f}]')

# ---------- 4. Fit K=2 harmonic regression on ALL 159 weeks (no a priori split) ----------
DOY = df['date'].dt.dayofyear.values
omega = 2 * np.pi / 365.25

def harmonic_X(t_doy, K=2):
    X = [np.ones(len(t_doy))]
    for k in range(1, K+1):
        X.append(np.cos(k * omega * t_doy))
        X.append(np.sin(k * omega * t_doy))
    return np.column_stack(X)

X = harmonic_X(DOY, K=2)
# Fit on full 159 weeks
beta_S, *_ = np.linalg.lstsq(X, S, rcond=None)
beta_I, *_ = np.linalg.lstsq(X, I_t, rcond=None)
resid_S = S - X @ beta_S
resid_I = I_t - X @ beta_I

print(f'\nHarmonic regression (K=2, full 159 weeks):')
print(f'  S residuals:  mean={resid_S.mean():.4f}, std={resid_S.std():.4f}')
print(f'  I residuals:  mean={resid_I.mean():.4f}, std={resid_I.std():.4f}')

# ---------- 5. Change-point detection: three methods ----------
# We test multiple series. For each, locate the most likely single change-point.

def cusum_changepoint(x):
    """Classical CUSUM: cumulative deviation from mean, find argmax of |CUSUM|."""
    x_demean = x - x.mean()
    cusum = np.cumsum(x_demean)
    # Sup-CUSUM statistic (Page 1954-type)
    abs_cusum = np.abs(cusum)
    cp = int(np.argmax(abs_cusum))
    # Bootstrap-based p-value (empirical)
    B = 5000
    rng = np.random.default_rng(42)
    null_stats = np.zeros(B)
    for b in range(B):
        x_perm = rng.permutation(x_demean)
        null_stats[b] = np.abs(np.cumsum(x_perm)).max()
    p_val = (null_stats >= abs_cusum.max()).mean()
    return cp, abs_cusum.max(), p_val

def pelt_changepoint(x, model='rbf', pen=None):
    """PELT via ruptures library."""
    # rbf model = kernel change-point; flexible for distributional shifts
    algo = rpt.Pelt(model=model, min_size=10).fit(x)
    if pen is None:
        # Default penalty: BIC-based
        pen = 3 * np.log(len(x)) * np.var(x)
    cps = algo.predict(pen=pen)
    # Returns: list of breakpoints; last element = len(x) (end marker)
    cps = [c for c in cps if c < len(x)]
    return cps

def baiperron_single(x, X_harm, min_frac=0.10):
    """
    Bai-Perron single break test on regression model y = X*beta + e.
    Returns: argmax F-statistic location, sup-F value, asymptotic p-value approximation.
    """
    T = len(x)
    lo = int(min_frac * T)
    hi = T - lo
    F_stats = np.full(T, np.nan)
    n_params = X_harm.shape[1]
    # Full model SSR (no break)
    beta_full, *_ = np.linalg.lstsq(X_harm, x, rcond=None)
    SSR_full = np.sum((x - X_harm @ beta_full)**2)
    for tau in range(lo, hi):
        # Pre-break model
        b1, *_ = np.linalg.lstsq(X_harm[:tau], x[:tau], rcond=None)
        SSR1 = np.sum((x[:tau] - X_harm[:tau] @ b1)**2)
        # Post-break model
        b2, *_ = np.linalg.lstsq(X_harm[tau:], x[tau:], rcond=None)
        SSR2 = np.sum((x[tau:] - X_harm[tau:] @ b2)**2)
        SSR_break = SSR1 + SSR2
        # Chow F-statistic
        df_num = n_params
        df_den = T - 2 * n_params
        if SSR_break > 0 and df_den > 0:
            F = (SSR_full - SSR_break) / df_num / (SSR_break / df_den)
            F_stats[tau] = F
    # sup-F statistic
    cp = int(np.nanargmax(F_stats))
    supF = F_stats[cp]
    # Andrews (1993) critical values for sup-F with trimming pi_0 = 0.10
    # For q (n_params restricted) ≈ 5 (intercept + 4 harmonic coefs): 5% cv ≈ 18.85, 1% cv ≈ 24.50
    # Approximate p-value via Hansen (1997) approximation isn't trivial; we use bootstrap
    B = 1000
    rng = np.random.default_rng(123)
    null_supF = np.zeros(B)
    resid = x - X_harm @ beta_full
    for b in range(B):
        x_boot = X_harm @ beta_full + rng.permutation(resid)
        F_b = np.full(T, np.nan)
        for tau in range(lo, hi):
            b1, *_ = np.linalg.lstsq(X_harm[:tau], x_boot[:tau], rcond=None)
            SSR1 = np.sum((x_boot[:tau] - X_harm[:tau] @ b1)**2)
            b2, *_ = np.linalg.lstsq(X_harm[tau:], x_boot[tau:], rcond=None)
            SSR2 = np.sum((x_boot[tau:] - X_harm[tau:] @ b2)**2)
            SSR_b = SSR1 + SSR2
            if SSR_b > 0:
                F_b[tau] = (SSR_full - SSR_b) / df_num / (SSR_b / df_den)
        null_supF[b] = np.nanmax(F_b)
    p_val = (null_supF >= supF).mean()
    return cp, supF, p_val, F_stats

# ---------- 6. Apply to multiple series ----------
results = {}

print('\n' + '='*70)
print('CHANGE-POINT DETECTION (single break, lowest-allowed-fraction = 0.10)')
print('='*70)

series_dict = {
    'S (Shannon entropy)': S,
    'Moran I': I_t,
    'North-Japan share': share_north,
    'Kyushu+Oki share': share_kyushu,
    'S residuals (K=2 harmonic)': resid_S,
    'I residuals (K=2 harmonic)': resid_I,
}

dates = df['date'].dt.date.values

for name, y in series_dict.items():
    print(f'\n--- {name} ---')
    # CUSUM
    cp_c, stat_c, p_c = cusum_changepoint(y)
    print(f'  CUSUM:        t* = {cp_c:3d} ({dates[cp_c]}), stat = {stat_c:.3f}, p ≈ {p_c:.4f}')
    # PELT (with mid-strength penalty)
    cps_p = pelt_changepoint(y, model='rbf', pen=10*np.var(y))
    cp_p_str = ', '.join([f'{c} ({dates[c]})' for c in cps_p]) if cps_p else 'none'
    print(f'  PELT (rbf):   breakpoints = {cp_p_str}')
    # Bai-Perron (only on harmonic residuals or de-seasonalized series — use directly on residuals or raw)
    if 'residuals' in name or name in ['S (Shannon entropy)', 'Moran I']:
        # Need de-seasonalized regression
        cp_bp, supF, p_bp, _ = baiperron_single(y, X, min_frac=0.10)
        print(f'  Bai-Perron:   t* = {cp_bp:3d} ({dates[cp_bp]}), sup-F = {supF:.3f}, p ≈ {p_bp:.4f}')
        results[name] = {'cusum_cp': cp_c, 'cusum_p': p_c,
                         'pelt_cps': cps_p,
                         'bp_cp': cp_bp, 'bp_supF': supF, 'bp_p': p_bp}
    else:
        results[name] = {'cusum_cp': cp_c, 'cusum_p': p_c, 'pelt_cps': cps_p}

# ---------- 7. Save results ----------
np.savez(os.path.join(_RES_DIR, 'phase1_changepoint_results.npz'),
         dates=df['date'].values,
         S=S, I_t=I_t,
         share_north=share_north, share_kyushu=share_kyushu,
         resid_S=resid_S, resid_I=resid_I,
         X_harmonic=X,
         beta_S=beta_S, beta_I=beta_I,
         results=results)
print(f'\nSaved to phase1_changepoint_results.npz')
