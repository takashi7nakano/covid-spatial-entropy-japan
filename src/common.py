"""
Common building blocks for the spatial-entropy pipeline.

These functions implement the core quantities promised in the manuscript's
"Software and reproducibility" section and are imported by every stage so the
published numbers are reproduced exactly:

  - data ingestion        -> load_data()
  - spatial weights W      -> build_W()
  - Shannon pseudo-entropy -> shannon_entropy()
  - Moran's I (per week)   -> morans_I() / morans_I_series()
  - harmonic regression    -> harmonic_X() / fit_harmonic()

Definitions are kept byte-for-byte identical to the frozen analysis scripts.
"""
import numpy as np
import pandas as pd

from config import PREF_FULL, ADJACENCY, HARMONIC_K

# Regional groupings used for KL / share decompositions (manuscript Methods)
NORTH = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima']
KYUSHU_OKI = ['Fukuoka', 'Saga', 'Nagasaki', 'Kumamoto', 'Oita', 'Miyazaki',
              'Kagoshima', 'Okinawa']


def load_data(pref_full=PREF_FULL, adjacency=ADJACENCY):
    """Load the prefecture-share table and adjacency edge list.

    Returns
    -------
    df       : DataFrame with columns [date, S, <47 prefecture shares>]
    prefs    : list of 47 prefecture column names (data order)
    dates    : DatetimeIndex of the 159 weekly observations
    P        : (T, 47) array of prefecture shares p_i(t), each row sums to 1
    adj_df   : DataFrame with columns [Prefecture A, Prefecture B]
    """
    df = pd.read_csv(pref_full, parse_dates=['date'])
    adj_df = pd.read_csv(adjacency)
    prefs = [c for c in df.columns if c not in ('date', 'S')]
    dates = pd.to_datetime(df['date'])
    P = df[prefs].values
    return df, prefs, dates, P, adj_df


def build_W(prefs, adj_df, row_standardize=False):
    """Build the binary symmetric Queen-contiguity weights matrix W.

    Set row_standardize=True for the row-standardised W used in Moran's I.
    """
    n = len(prefs)
    idx = {p: i for i, p in enumerate(prefs)}
    W = np.zeros((n, n))
    for _, row in adj_df.iterrows():
        a, b = idx[row['Prefecture A']], idx[row['Prefecture B']]
        W[a, b] = 1.0
        W[b, a] = 1.0
    if row_standardize:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0          # leave isolated nodes (e.g. Okinawa) as zero rows
        W = W / rs
    return W


def shannon_entropy(P):
    """Shannon pseudo-entropy S(t) of the prefecture-share distribution.

    P may be a (47,) vector or a (T, 47) array; returns scalar or (T,) array.
    """
    P = np.asarray(P)
    return -(P * np.log(P + 1e-30)).sum(axis=-1)


def morans_I(x, W_rs):
    """Global Moran's I for one week, using row-standardised weights W_rs."""
    z = x - x.mean()
    num = (W_rs * np.outer(z, z)).sum()
    den = (z ** 2).sum() / len(z)
    return num / den / W_rs.sum() * len(z)


def morans_I_series(P, W_rs):
    """Moran's I for every week; returns (T,) array."""
    return np.array([morans_I(P[t], W_rs) for t in range(P.shape[0])])


def harmonic_X(dates, K=HARMONIC_K):
    """Design matrix for a K-harmonic annual regression on day-of-year."""
    doy = pd.to_datetime(dates).dt.dayofyear.values
    omega = 2 * np.pi / 365.25
    cols = [np.ones(len(doy))]
    for k in range(1, K + 1):
        cols.append(np.cos(k * omega * doy))
        cols.append(np.sin(k * omega * doy))
    return np.column_stack(cols)


def fit_harmonic(y, X):
    """OLS fit y = X beta + e; returns (beta, fitted, residuals)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    return beta, fitted, y - fitted
