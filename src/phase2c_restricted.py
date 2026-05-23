"""
Phase 2c: Restricted change-point detection.
Approach: define entropy ceiling failure as the operational signature of regime transition.
1. Identify t_last = last week with S(t) >= threshold (3.80)
2. Restrict change-point search to t > t_last
3. Apply PELT, Bai-Perron, BOCPD within this restricted region
4. Sensitivity to threshold choice (3.79, 3.80, 3.81, 3.82)
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
from matplotlib.dates import DateFormatter, MonthLocator

# Load data
data = np.load(os.path.join(_RES_DIR, 'phase1_changepoint_results.npz'), allow_pickle=True)
dates = pd.to_datetime(data['dates'])
S = data['S']
T = len(S)

print(f'Total: {T} weeks, {dates[0].date()} → {dates[-1].date()}')

# ---------- 1. Find t_last for various thresholds ----------
print('\n' + '='*72)
print('LAST WEEK WHERE S(t) >= threshold')
print('='*72)
print(f'{"Threshold":<12} {"t_last":<8} {"date":<12} {"S(t_last)":<10} {"# post weeks":<12}')
print('-'*70)
for threshold in [3.79, 3.80, 3.81, 3.82, 3.83]:
    above = np.where(S >= threshold)[0]
    if len(above) == 0:
        print(f'{threshold:<12} no week')
        continue
    t_last = above[-1]
    n_post = T - t_last - 1
    print(f'{threshold:<12} {t_last:<8} {dates[t_last].date()} {S[t_last]:<10.4f} {n_post:<12}')

# Choose 3.80 as principal threshold
THRESHOLD = 3.80
above = np.where(S >= THRESHOLD)[0]
t_last = above[-1]
print(f'\nUsing principal threshold = {THRESHOLD}')
print(f't_last = {t_last} ({dates[t_last].date()}), S({t_last}) = {S[t_last]:.4f}')
print(f'Restricted region: {t_last+1} .. {T-1} = {T - t_last - 1} weeks')

# ---------- 2. PELT on restricted region ----------
print('\n' + '='*72)
print('PELT on RESTRICTED region (weeks after last entropy-ceiling attainment)')
print('='*72)
S_restricted = S[t_last+1:]
T_r = len(S_restricted)
print(f'Restricted series length: {T_r} weeks')

# Penalty: BIC-style on restricted region variance
sigma2_r = np.var(S_restricted)
print(f'σ²(restricted) = {sigma2_r:.5f}, log(T_r) = {np.log(T_r):.3f}')
for factor in [0.5, 1.0, 2.0, 4.0]:
    pen = factor * sigma2_r * np.log(T_r)
    try:
        cps_local = rpt.Pelt(model='l2', min_size=5).fit(S_restricted).predict(pen=pen)
        cps_local = [c for c in cps_local if c < T_r]
        # Convert back to global indices
        cps_global = [c + t_last + 1 for c in cps_local]
        cp_str = ', '.join([f'{c} ({dates[c].date()})' for c in cps_global]) if cps_global else 'none'
        print(f'  pen = {factor:.1f}× σ²log(T_r) = {pen:.5f}: {cp_str}')
    except Exception as e:
        print(f'  pen = {factor:.1f}× σ²log(T_r): ERROR - {e}')

# ---------- 3. Bai-Perron sup-F on restricted region ----------
print('\n' + '='*72)
print('Bai-Perron sup-F on RESTRICTED region')
print('='*72)

# Need to also restrict the harmonic regressors
X = data['X_harmonic']
X_restricted = X[t_last+1:]

def baiperron_supF(y, X, min_frac=0.10):
    T = len(y)
    lo = max(int(min_frac * T), X.shape[1] + 1)
    hi = T - lo
    n_params = X.shape[1]
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    SSR_full = np.sum((y - X @ beta_full)**2)
    F_stats = np.full(T, np.nan)
    df_num = n_params
    df_den = T - 2 * n_params
    for tau in range(lo, hi):
        try:
            b1, *_ = np.linalg.lstsq(X[:tau], y[:tau], rcond=None)
            SSR1 = np.sum((y[:tau] - X[:tau] @ b1)**2)
            b2, *_ = np.linalg.lstsq(X[tau:], y[tau:], rcond=None)
            SSR2 = np.sum((y[tau:] - X[tau:] @ b2)**2)
            SSR_break = SSR1 + SSR2
            if SSR_break > 0 and df_den > 0:
                F_stats[tau] = (SSR_full - SSR_break) / df_num / (SSR_break / df_den)
        except Exception:
            pass
    return F_stats

F_r = baiperron_supF(S_restricted, X_restricted, min_frac=0.10)
if not np.all(np.isnan(F_r)):
    cp_local = int(np.nanargmax(F_r))
    cp_global = cp_local + t_last + 1
    supF = F_r[cp_local]
    print(f'  sup-F = {supF:.2f} at local t = {cp_local} → global t = {cp_global} ({dates[cp_global].date()})')

    # Bootstrap p-value (simple permutation)
    B = 1000
    rng = np.random.default_rng(42)
    beta_full, *_ = np.linalg.lstsq(X_restricted, S_restricted, rcond=None)
    resid = S_restricted - X_restricted @ beta_full
    null_supF = np.zeros(B)
    for b in range(B):
        y_boot = X_restricted @ beta_full + rng.permutation(resid)
        F_b = baiperron_supF(y_boot, X_restricted, min_frac=0.10)
        null_supF[b] = np.nanmax(F_b) if not np.all(np.isnan(F_b)) else 0
    p_val = (null_supF >= supF).mean()
    print(f'  Bootstrap p-value (B={B}) ≈ {p_val:.4f}')
else:
    print('  Insufficient weeks for Bai-Perron')

# ---------- 4. CUSUM on restricted region ----------
print('\n' + '='*72)
print('CUSUM on RESTRICTED region')
print('='*72)
x_demean = S_restricted - S_restricted.mean()
cusum = np.cumsum(x_demean)
abs_cusum = np.abs(cusum)
cp_local = int(np.argmax(abs_cusum))
cp_global = cp_local + t_last + 1
B = 5000
rng = np.random.default_rng(123)
null_stats = np.zeros(B)
for b in range(B):
    perm = rng.permutation(x_demean)
    null_stats[b] = np.abs(np.cumsum(perm)).max()
p_val = (null_stats >= abs_cusum.max()).mean()
print(f'  CUSUM peak at local t = {cp_local} → global t = {cp_global} ({dates[cp_global].date()})')
print(f'  Stat = {abs_cusum.max():.3f}, bootstrap p = {p_val:.4f}')

# ---------- 5. Sensitivity to threshold ----------
print('\n' + '='*72)
print('Threshold sensitivity: PELT-detected CP on restricted region for each threshold')
print('='*72)
print(f'{"Threshold":<12} {"t_last":<20} {"# wks":<8} {"PELT CP (BIC pen)":<30}')
print('-'*70)
for threshold in [3.78, 3.79, 3.80, 3.81, 3.82]:
    above = np.where(S >= threshold)[0]
    if len(above) == 0 or T - above[-1] - 1 < 10:
        print(f'{threshold:<12} {above[-1] if len(above) > 0 else "none":<20} {T - (above[-1] if len(above) > 0 else 0) - 1:<8} too few wks')
        continue
    t_last_th = above[-1]
    S_r = S[t_last_th+1:]
    pen = 2 * np.var(S_r) * np.log(len(S_r))
    cps_local = rpt.Pelt(model='l2', min_size=5).fit(S_r).predict(pen=pen)
    cps_local = [c for c in cps_local if c < len(S_r)]
    cps_global = [c + t_last_th + 1 for c in cps_local]
    cp_str = ', '.join([f'{dates[c].date()}' for c in cps_global]) if cps_global else 'NO CP (single regime)'
    print(f'{threshold:<12} {t_last_th} ({dates[t_last_th].date()})   {T - t_last_th - 1:<8} {cp_str}')

# ---------- 6. Build figure ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# Panel 1: full series with threshold and restricted region
ax = axes[0]
ax.plot(dates, S, 'b-', linewidth=1.2, label='S(t)')
ax.axhline(THRESHOLD, color='red', linestyle='--', alpha=0.6, label=f'threshold = {THRESHOLD}')
ax.axhline(np.log(47), color='black', linestyle=':', alpha=0.4, label='ln 47 (theoretical max)')
ax.axvline(dates[t_last], color='green', linestyle='-', linewidth=2, alpha=0.7,
           label=f't_last (last S≥{THRESHOLD}) = {dates[t_last].date()}')
ax.axvspan(dates[t_last], dates[-1], color='red', alpha=0.1, label='Restricted CP search region')
ax.set_ylabel('S(t)', fontsize=11)
ax.set_title('Phase 1d: Entropy-ceiling-failure restricted change-point detection', fontsize=11)
ax.legend(loc='lower left', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Restricted region zoomed
ax = axes[1]
ax.plot(dates[t_last+1:], S_restricted, 'b-', linewidth=1.5)
ax.set_ylabel('S(t) within restricted region', fontsize=11)
# Recompute PELT for visualization on principal threshold
pen = 2 * np.var(S_restricted) * np.log(len(S_restricted))
cps_local = rpt.Pelt(model='l2', min_size=5).fit(S_restricted).predict(pen=pen)
cps_local = [c for c in cps_local if c < len(S_restricted)]
for c in cps_local:
    cp_global_v = c + t_last + 1
    ax.axvline(dates[cp_global_v], color='red', linestyle='--', linewidth=2,
               label=f'PELT CP: {dates[cp_global_v].date()}')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlabel('Date', fontsize=11)

axes[-1].xaxis.set_major_locator(MonthLocator(bymonth=[1,4,7,10]))
axes[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(_FIG_DIR, 'phase2c_restricted_cp.png'), dpi=150, bbox_inches='tight')
print('\nFigure: phase2c_restricted_cp.png')
