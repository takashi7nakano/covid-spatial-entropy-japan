#!/usr/bin/env python3
"""Exploratory and descriptive measures reported in the manuscript.

Stage 12 of reproduce.sh. Everything here is *descriptive or exploratory*: none of
it enters the confirmatory entropy-ceiling test, and none of it feeds the
change-point estimation of stages 1-5. It is included so that every number that
appears in the manuscript can be regenerated from the archived data.

Inputs : data/pref_full_n33.csv, data/combined_n33.csv  (173 weeks)
Outputs: results/exploratory_measures.npz + printed table

Sections
  1. Cross-correlation between S(t) and log national amplitude (pre-transition)
  2. Post-transition summary of S and the moving-block-bootstrap difference
  3. Amplitude adjustment: OLS era regression, boundary sensitivity,
     amplitude-matched subset and permutation test
  4. Coefficient of variation and synchrony fraction phi
  5. Off-season participation ratios (Supplementary Table S11.1)
  6. Elapsed intervals below reference entropy levels (Supplementary Table S14.1)
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from config import DATA, RESULTS

PREFS = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima','Ibaraki','Tochigi','Gunma',
         'Saitama','Chiba','Tokyo','Kanagawa','Niigata','Toyama','Ishikawa','Fukui','Yamanashi','Nagano',
         'Gifu','Shizuoka','Aichi','Mie','Shiga','Kyoto','Osaka','Hyogo','Nara','Wakayama','Tottori',
         'Shimane','Okayama','Hiroshima','Yamaguchi','Tokushima','Kagawa','Ehime','Kochi','Fukuoka',
         'Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa']
IDX   = {p: i for i, p in enumerate(PREFS)}
KY7   = ['Fukuoka','Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima']
NORTH = ['Hokkaido','Aomori','Iwate','Miyagi','Akita','Yamagata','Fukushima','Niigata','Nagano','Toyama']

CP    = '2025-11-27'   # principal change point, 2025-W48 (mid-week Thursday dating)
ERA   = '2026-01-01'   # principal era boundary for the amplitude-adjusted regression
CEIL  = 3.80
ACTIVE = 3.0           # off-season "active" threshold (multiple of baseline)
WAVES = [('Summer 2023','2023-08-31','summer'), ('Winter 2023-24','2024-02-01','winter'),
         ('Summer 2024','2024-07-25','summer'), ('Winter 2024-25','2025-01-09','winter'),
         ('Summer 2025','2025-08-21','summer'), ('Winter 2025-26','2026-02-05','winter'),
         ('Summer 2026','2026-07-16','summer')]

def h(t): print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)

# ---------------------------------------------------------------- load
p = pd.read_csv(os.path.join(str(DATA), 'pref_full_n33.csv'))
c = pd.read_csv(os.path.join(str(DATA), 'combined_n33.csv'))
for df in (p, c):
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
cols = [x for x in p.columns if x not in ('date', 'S')]
assert cols == PREFS, 'unexpected prefecture column order'
m = p.merge(c[['date', 'cases', 'window']], on='date')
M = m[cols].values * m['cases'].values[:, None]      # per-sentinel rate, 173 x 47
d = list(m['date'])
P = M / M.sum(axis=1, keepdims=True)
S = -(P * np.log(P + 1e-30)).sum(axis=1)
amp = M.mean(axis=1)
i_cp  = d.index(CP)
i_end = d.index('2026-05-07')                        # end of the 159-week estimation window
print(f'Loaded {len(d)} weeks, {d[0]} -> {d[-1]};  change point {CP} (index {i_cp})')

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))

# ------------------------------------------------------- 1. cross-correlation
h('1. CROSS-CORRELATION  S(t) vs log national amplitude, pre-transition weeks')
pre_S, pre_L = S[:i_cp], np.log(amp[:i_cp])
prof = {lag: pearson(pre_L[:len(pre_L) - lag], pre_S[lag:]) for lag in range(0, 9)}
best = max(prof, key=lambda k: prof[k])
print(f'  pre-transition n = {i_cp} weeks ({d[0]} .. {d[i_cp-1]})')
print('  lag (wk) ' + ''.join(f'{k:>8d}' for k in prof))
print('  r        ' + ''.join(f'{v:>8.3f}' for v in prof.values()))
print(f'  maximised at lag +{best} weeks, r = {prof[best]:.3f}  (n = {i_cp - best} pairs)')

# --------------------------------------------- 2. post-transition summary of S
h('2. POST-TRANSITION SUMMARY OF S  (estimation window only)')
pre, post = S[:i_cp], S[i_cp:i_end + 1]
j = i_cp + int(np.argmax(post))
diff = post.mean() - pre.mean()
print(f'  pre  n = {len(pre):3d}  mean {pre.mean():.4f}  SD {pre.std():.4f}  max {pre.max():.4f}')
print(f'  post n = {len(post):3d}  mean {post.mean():.4f}  SD {post.std():.4f}  max {post.max():.4f} @ {d[j]}'
      f'  = {post.max()/math.log(47)*100:.1f}% of ln 47')
print(f'  difference (post - pre) = {diff:+.4f}')

def ar1(v):
    return pearson(v[:-1], v[1:])

def mbb(v, b, rng):
    n = len(v); k = math.ceil(n / b); starts = max(1, n - b + 1)
    out = []
    for _ in range(k):
        s = rng.integers(starts); out.extend(v[s:s + b])
    return np.array(out[:n])

b_pre, b_post = max(1, round(1/(1-ar1(pre)))), max(1, round(1/(1-ar1(post))))
rng = np.random.default_rng(20260824); B = 10000
ds = np.array([mbb(post, min(b_post, len(post)), rng).mean() - mbb(pre, b_pre, rng).mean() for _ in range(B)])
lo, hi = np.percentile(ds, [2.5, 97.5])
print(f'  block lengths (Politis-Romano, per group): pre b = {b_pre}, post b = {b_post}')
print(f'  moving block bootstrap ({B} reps): 95% CI {lo:+.4f} .. {hi:+.4f}   SE {ds.std():.4f}   z = {diff/ds.std():.2f}')

# ------------------------------------------------------- 3. amplitude adjustment
h('3. AMPLITUDE ADJUSTMENT  (all 173 weeks; era boundary = %s)' % ERA)
def era_fit(bound):
    e = (np.array(d) >= bound).astype(float)
    X = np.column_stack([np.ones(len(S)), np.log(amp), e])
    b, *_ = np.linalg.lstsq(X, S, rcond=None)
    r = S - X @ b; s2 = r @ r / (len(S) - 3)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b[2], se[2], b[2] / se[2], int(e.sum())
co, se, t, n_post = era_fit(ERA)
print(f'  S ~ log(amplitude) + era :  era = {co:+.4f}  SE {se:.4f}  t = {t:.2f}  (post n = {n_post})')
print('  boundary sensitivity:')
for bnd in ['2025-10-01','2025-11-24','2025-12-01','2026-01-01','2026-02-01']:
    co_b, se_b, t_b, n_b = era_fit(bnd)
    print(f'    {bnd}: era = {co_b:+.4f}  t = {t_b:.2f}  (post n = {n_b})')
i_era = d.index(ERA)
lo_a, hi_a = amp[i_era:].min(), amp[i_era:].max()
mp = [k for k in range(i_era) if lo_a <= amp[k] <= hi_a]
obs = S[i_era:].mean() - S[mp].mean()
lab = np.array([0]*len(mp) + [1]*(len(S)-i_era))
vals = np.concatenate([S[mp], S[i_era:]])
rng2 = np.random.default_rng(42); NP = 20000
cnt = sum(1 for _ in range(NP)
          if abs(vals[(pm := rng2.permutation(lab)) == 1].mean() - vals[pm == 0].mean()) >= abs(obs))
print(f'  post-transition amplitude range {lo_a:.2f}-{hi_a:.2f} is spanned by {len(mp)} pre-transition weeks'
      f'  (mean S = {S[mp].mean():.3f}, SD {S[mp].std(ddof=1):.3f})')
print(f'  amplitude-matched difference = {obs:+.4f}   permutation p = {(cnt+1)/(NP+1):.5f}  ({NP} permutations)')

# ------------------------------------------------------------- 4. CV and phi
h('4. COEFFICIENT OF VARIATION AND SYNCHRONY FRACTION')
CV  = M.std(axis=1) / M.mean(axis=1)
phi = np.exp(S) / 47.0
ceil_wk = []
for name, pk, _ in WAVES[:5]:
    i = d.index(pk); a, b = max(0, i-8), min(len(d), i+9)
    ceil_wk.append(a + int(np.argmax(S[a:b])))
print('  wave ceiling weeks (max S within +/-8 wk of the national peak):')
for (name, pk, _), k in zip(WAVES[:5], ceil_wk):
    print(f'    {name:15s} peak {pk} -> ceiling {d[k]}  S = {S[k]:.4f}  CV = {CV[k]:.3f}  phi = {phi[k]:.3f}'
          f'  prefectures > 1.0 per sentinel: {int((M[k] > 1.0).sum())}/47')
print(f'  CV  at the five ceiling weeks : {min(CV[ceil_wk]):.3f} - {max(CV[ceil_wk]):.3f}')
print(f'  phi at the five ceiling weeks : {min(phi[ceil_wk]):.3f} - {max(phi[ceil_wk]):.3f}')
print(f'  CV  over the {len(S)-i_cp} post-change-point weeks : {CV[i_cp:].min():.3f} - {CV[i_cp:].max():.3f}')
k1 = d.index('2026-07-16') + 1
print(f'  phi one week after the summer 2026 peak ({d[k1]}) : {phi[k1]:.3f}'
      f'   deficit {47*(1-phi[k1]):.1f} prefectures')

# --------------------------------------------------- 5. off-season participation
h('5. OFF-SEASON PARTICIPATION RATIO  (Supplementary Table S11.1)')
era_lab = np.array([0 if x < ERA else 1 for x in d])   # baselines use the era boundary, not the change point
base = np.zeros((2, 47))
for e in (0, 1):
    base[e] = np.percentile(M[era_lab == e], 20, axis=0)
print(f'  active threshold = {ACTIVE}x the era-specific 20th-percentile baseline')
print(f'  {"Wave":16s}{"Region":20s}' + ''.join(f'{"Peak+"+str(q) if q else "Peak":>9s}' for q in range(5)))
tab = {}
for name, pk, season in WAVES:
    reg = NORTH if season == 'summer' else KY7
    ids = [IDX[x] for x in reg]; i = d.index(pk); row = []
    for q in range(5):
        k = i + q
        row.append(float(M[k][ids].mean() / base[era_lab[k]][ids].mean()) if k < len(d) else float('nan'))
    tab[name] = row
    print(f'  {name:16s}{("Northern" if season=="summer" else "Kyushu (mainland)"):20s}'
          + ''.join(f'{v:9.2f}' for v in row))
for q, lab_q in ((1, 'peak+1'),):
    pre_n = sum(1 for name, _, _ in WAVES[:5] if tab[name][q] >= ACTIVE)
    post_n = sum(1 for name, _, _ in WAVES[5:] if tab[name][q] >= ACTIVE)
    print(f'  at {lab_q}, threshold {ACTIVE}: pre-transition {pre_n}/5 active, post-transition {post_n}/2 active')

# ------------------------------------------------- 6. elapsed intervals (S14)
h('6. ELAPSED INTERVALS BELOW REFERENCE ENTROPY LEVELS  (Supplementary Table S14.1)')
print(f'  {"level":>7s}{"pre weeks >= level":>22s}{"max pre gap":>13s}{"last attained":>15s}{"elapsed":>9s}{"ratio":>8s}')
for lev in (3.75, 3.70, 3.65):
    hits_pre = [k for k in range(i_cp) if S[k] >= lev]
    gaps = [hits_pre[q+1] - hits_pre[q] - 1 for q in range(len(hits_pre)-1)]
    mx = max(gaps) if gaps else 0
    last = max(k for k in range(len(S)) if S[k] >= lev)
    el = len(S) - 1 - last
    ratio = f'{el/mx:.1f}' if mx and el else '—'
    print(f'  {lev:>7.2f}{f"{len(hits_pre)}/{i_cp}":>22s}{mx:>13d}{d[last]:>15s}{el:>9d}{ratio:>8s}')
last80 = max(k for k in range(len(S)) if S[k] >= CEIL)
print(f'\n  entropy ceiling {CEIL}: last attained {d[last80]}, {len(S)-1-last80} weeks to the end of the record;'
      f' post-change-point weeks attaining it: {int((S[i_cp:] >= CEIL).sum())}/{len(S)-i_cp}')

os.makedirs(str(RESULTS), exist_ok=True)
np.savez(os.path.join(str(RESULTS), 'exploratory_measures.npz'),
         dates=np.array(d), S=S, amp=amp, CV=CV, phi=phi,
         xcorr_lags=np.array(list(prof)), xcorr_r=np.array(list(prof.values())),
         era_coef=co, era_se=se, matched_diff=obs)
print('\nSaved: results/exploratory_measures.npz')
