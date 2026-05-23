#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the full analysis pipeline end-to-end from the frozen n19 data.
# Runs every stage in dependency order and writes all outputs to results/
# and figures/. Safe to re-run; later stages depend on earlier outputs.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/src"

echo "==> [1/7] data ingestion + S(t) + Moran's I + harmonic regression + change-point"
python3 phase1_changepoint.py            # -> results/phase1_changepoint_results.npz

echo "==> [2/7] change-point robustness (PELT penalty, BOCPD, bootstrap CI)"
python3 phase1c_robustness.py            # -> figures/phase1_robustness.png

echo "==> [3/7] principal pre/post re-analysis (block bootstrap, K-sensitivity, AR(1))"
python3 phase2_principal.py              # -> results/phase2_principal_results.npz

echo "==> [4/7] full sensitivity analyses (K, persistence-ratio CI, leave-one-prefecture-out)"
python3 phase2b_full.py                  # -> results/phase2b_results.npz

echo "==> [5/7] restricted entropy-ceiling change-point (threshold sensitivity)"
python3 phase2c_restricted.py            # -> figures/phase2c_restricted_cp.png

echo "==> [6/7] main figures 1-6 (regenerate_figs) + Fig 1A per-sentinel overlay"
python3 regenerate_figs.py               # -> figures/Fig1..Fig6 (Fig1 = base version)
python3 make_fig1_overlay.py             # -> figures/Fig1_wave_template.* (published Fig 1A)

echo "==> [7/7] Figure 7 change-point panel"
python3 make_fig7.py                     # -> figures/Fig7_changepoint.{png,pdf}

echo ""
echo "Done. See results/ for .npz artefacts and figures/ for all figures."
