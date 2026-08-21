"""
Robustness analysis: test whether the (A)/(B) anti-phase finding and 2026 regime change
are sensitive to the choice of adjacency matrix W.

Variants:
1. Baseline: 92 edges (land + sea bridges/tunnels)
2. Land only: remove all 6 sea crossings
3. Drop Kagoshima-Okinawa only (test sensitivity of (A) summer mode)
4. Drop Hokkaido-Aomori only (test sensitivity of (B) winter mode)
5. Baseline + major air routes (Hokkaido↔Tokyo, Okinawa↔Tokyo/Osaka/Fukuoka)
6. k-nearest neighbors (k=5 by centroid distance)
7. Distance decay W_ij = 1/(1+(d/d0)²) with d0=200km
"""
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

from config import PREF_FULL, RESULTS

from prefecture_coords import PREFS, idx, N, COORDS

# Baseline edges (from previous analysis)
edges_baseline = [
    ('Hokkaido','Aomori'),
    ('Aomori','Akita'),('Aomori','Iwate'),
    ('Iwate','Akita'),('Iwate','Miyagi'),
    ('Miyagi','Akita'),('Miyagi','Yamagata'),('Miyagi','Fukushima'),
    ('Akita','Yamagata'),
    ('Yamagata','Fukushima'),('Yamagata','Niigata'),
    ('Fukushima','Niigata'),('Fukushima','Gunma'),('Fukushima','Tochigi'),('Fukushima','Ibaraki'),
    ('Ibaraki','Tochigi'),('Ibaraki','Saitama'),('Ibaraki','Chiba'),
    ('Tochigi','Gunma'),('Tochigi','Saitama'),
    ('Gunma','Saitama'),('Gunma','Niigata'),('Gunma','Nagano'),
    ('Saitama','Tokyo'),('Saitama','Yamanashi'),('Saitama','Nagano'),('Saitama','Chiba'),
    ('Chiba','Tokyo'),
    ('Tokyo','Kanagawa'),('Tokyo','Yamanashi'),
    ('Kanagawa','Yamanashi'),('Kanagawa','Shizuoka'),
    ('Niigata','Nagano'),('Niigata','Toyama'),
    ('Toyama','Nagano'),('Toyama','Gifu'),('Toyama','Ishikawa'),
    ('Ishikawa','Gifu'),('Ishikawa','Fukui'),
    ('Fukui','Gifu'),('Fukui','Shiga'),('Fukui','Kyoto'),
    ('Yamanashi','Shizuoka'),('Yamanashi','Nagano'),
    ('Nagano','Shizuoka'),('Nagano','Aichi'),('Nagano','Gifu'),
    ('Gifu','Aichi'),('Gifu','Mie'),('Gifu','Shiga'),
    ('Shizuoka','Aichi'),
    ('Aichi','Mie'),
    ('Mie','Shiga'),('Mie','Kyoto'),('Mie','Nara'),('Mie','Wakayama'),
    ('Shiga','Kyoto'),
    ('Kyoto','Osaka'),('Kyoto','Hyogo'),('Kyoto','Nara'),
    ('Osaka','Hyogo'),('Osaka','Nara'),('Osaka','Wakayama'),
    ('Hyogo','Tottori'),('Hyogo','Okayama'),('Hyogo','Tokushima'),  # Akashi+Onaruto
    ('Nara','Wakayama'),
    ('Tottori','Shimane'),('Tottori','Okayama'),('Tottori','Hiroshima'),
    ('Shimane','Hiroshima'),('Shimane','Yamaguchi'),
    ('Okayama','Hiroshima'),
    ('Okayama','Kagawa'),                                            # Seto Ohashi
    ('Hiroshima','Yamaguchi'),
    ('Hiroshima','Ehime'),                                           # Shimanami
    ('Yamaguchi','Fukuoka'),                                         # Kanmon
    ('Tokushima','Kagawa'),('Tokushima','Ehime'),('Tokushima','Kochi'),
    ('Kagawa','Ehime'),
    ('Ehime','Kochi'),
    ('Fukuoka','Saga'),('Fukuoka','Oita'),('Fukuoka','Kumamoto'),
    ('Saga','Nagasaki'),
    ('Kumamoto','Oita'),('Kumamoto','Miyazaki'),('Kumamoto','Kagoshima'),
    ('Oita','Miyazaki'),
    ('Miyazaki','Kagoshima'),
    ('Kagoshima','Okinawa'),                                         # via Amami
]

sea_crossings = {('Hokkaido','Aomori'), ('Hyogo','Tokushima'), ('Okayama','Kagawa'),
                 ('Hiroshima','Ehime'), ('Yamaguchi','Fukuoka'), ('Kagoshima','Okinawa')}
sea_crossings_norm = {tuple(sorted(e)) for e in sea_crossings}

def build_W(edges):
    W = np.zeros((N, N))
    for a, b in edges:
        i, j = idx[a], idx[b]
        W[i,j] = 1; W[j,i] = 1
    return W

def haversine(c1, c2):
    lat1, lon1 = c1; lat2, lon2 = c2
    R = 6371
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

D = np.zeros((N, N))
for i, p1 in enumerate(PREFS):
    for j, p2 in enumerate(PREFS):
        D[i,j] = haversine(COORDS[p1], COORDS[p2])

# Variants
W_baseline = build_W(edges_baseline)

edges_land = [e for e in edges_baseline if tuple(sorted(e)) not in sea_crossings_norm]
W_land = build_W(edges_land)

edges_noOK = [e for e in edges_baseline if set(e) != {'Kagoshima','Okinawa'}]
W_noOK = build_W(edges_noOK)

edges_noSe = [e for e in edges_baseline if set(e) != {'Hokkaido','Aomori'}]
W_noSe = build_W(edges_noSe)

extra_air = [('Hokkaido','Tokyo'),('Okinawa','Tokyo'),('Okinawa','Osaka'),('Okinawa','Fukuoka')]
edges_air = list({tuple(sorted(e)) for e in edges_baseline} | {tuple(sorted(e)) for e in extra_air})
W_air = build_W(edges_air)

def knn_W(D, k):
    W = np.zeros_like(D)
    for i in range(N):
        nearest = np.argsort(D[i])[1:k+1]
        W[i, nearest] = 1
    return ((W + W.T) > 0).astype(float)

W_knn5 = knn_W(D, 5)

W_dist = 1.0 / (1.0 + (D/200)**2)
np.fill_diagonal(W_dist, 0)

variants = {
    'Baseline (92 edges)': W_baseline,
    'Land only (no sea bridges)': W_land,
    'Drop Kagoshima-Okinawa': W_noOK,
    'Drop Hokkaido-Aomori': W_noSe,
    'Baseline + 4 air routes': W_air,
    'k-nearest (k=5)': W_knn5,
    'Distance decay (d0=200km)': W_dist,
}

def morans_I(p, W):
    deg = W.sum(axis=1)
    mask = deg > 0
    p_a = p[mask]; W_a = W[mask][:,mask]
    n = mask.sum()
    z = p_a - p_a.mean()
    S0 = W_a.sum()
    if S0 == 0 or (z*z).sum() == 0:
        return np.nan
    return (n / S0) * (W_a * np.outer(z,z)).sum() / (z*z).sum()

# Load data
df = pd.read_csv(PREF_FULL, parse_dates=['date'])
P = df[PREFS].values
T = P.shape[0]
dates = pd.to_datetime(df['date'].values)
pre_2026 = (dates < pd.Timestamp('2026-01-01'))
era_2026 = ~pre_2026
month = np.array([d.month for d in dates])
summer_pre = pre_2026 & np.isin(month, [6,7,8])
winter_pre = pre_2026 & np.isin(month, [11,12,1,2])

results = {}
print(f"\n{'Variant':<32}{'edges':>8}{'pre-26':>10}{'2026':>10}{'ratio':>8}{'sum':>10}{'win':>10}{'win/sum':>10}")
print("-"*98)
for name, W in variants.items():
    I_t = np.array([morans_I(P[t], W) for t in range(T)])
    results[name] = I_t
    n_edges = int(W[W>0].size / 2) if (W==0).sum() + (W>0).sum() == N*N else int((W>0).sum()/2)
    if 'decay' in name.lower():
        n_edges_str = "weighted"
    else:
        n_edges_str = str(int((W>0).sum()/2))
    pre_m = I_t[pre_2026].mean()
    e26_m = I_t[era_2026].mean()
    sum_m = I_t[summer_pre].mean()
    win_m = I_t[winter_pre].mean()
    print(f"{name:<32}{n_edges_str:>8}{pre_m:>10.3f}{e26_m:>10.3f}{e26_m/pre_m:>8.2f}"
          f"{sum_m:>10.3f}{win_m:>10.3f}{win_m/sum_m:>10.2f}")

# Save results
np.savez(RESULTS / 'robustness_results.npz',
         variant_names=list(variants.keys()),
         **{f'I_{i}': results[name] for i, name in enumerate(results.keys())},
         dates=df['date'].values, pre_2026=pre_2026,
         W_baseline=W_baseline, W_land=W_land, W_dist=W_dist)
print("\nSaved robustness_results.npz")
