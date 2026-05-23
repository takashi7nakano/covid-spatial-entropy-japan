"""
Central path configuration for the covid-spatial-entropy-japan pipeline.

All paths are resolved relative to the repository root, so the pipeline runs
identically regardless of the working directory or machine.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Canonical frozen inputs (n19 = 159 weeks, 2023-W17 .. 2026-W19)
PREF_FULL = DATA / "pref_full_n19.csv"          # date, S, <47 prefecture shares p_i(t)>
ADJACENCY = DATA / "adjacency_edges.csv"          # 92 undirected Queen-contiguity edges

# Intermediate / output artefacts
CHANGEPOINT_NPZ = RESULTS / "phase1_changepoint_results.npz"

# Frozen analysis constants (must match the published manuscript)
CP_DATE = "2025-11-27"      # data-driven change-point, epi week 2025-W48
ENTROPY_THRESHOLD = 3.80    # entropy-ceiling threshold for restricted CP search
RANDOM_SEED = 42            # fixed seed for all permutation / bootstrap tests
HARMONIC_K = 2              # principal harmonic order

for d in (RESULTS, FIGURES):
    d.mkdir(exist_ok=True)
