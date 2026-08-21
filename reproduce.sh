#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the full analysis pipeline end-to-end.
#
# Stages 1-7 and 9-10 use only the frozen 159-week estimation window (n19). Stage 8 is
# the only stage that reads the 14 out-of-sample weeks (n33, 2026-W20..W33);
# it performs no estimation. Stages 9-10 rebuild Supplementary Figures S1 and S2
# (the adjacency specification and the seven adjacency-matrix variants). Stage 11
# assembles figures/submission/ under the figure numbers used in the manuscript.
#
# All outputs go to results/ (.npz) and figures/ (PNG/PDF). Safe to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/src"

echo "==> [1/9] data ingestion + S(t) + Moran's I + harmonic regression + change-point"
python3 phase1_changepoint.py            # -> results/phase1_changepoint_results.npz

echo "==> [2/9] change-point robustness (PELT penalty, BOCPD, bootstrap CI)"
python3 phase1c_robustness.py            # -> figures/phase1_robustness.png

echo "==> [3/9] principal pre/post re-analysis (block bootstrap, K-sensitivity, AR(1))"
python3 phase2_principal.py              # -> results/phase2_principal_results.npz

echo "==> [4/9] full sensitivity analyses (K, persistence-ratio CI, leave-one-prefecture-out)"
python3 phase2b_full.py                  # -> results/phase2b_results.npz

echo "==> [5/9] restricted entropy-ceiling change-point (threshold sensitivity)"
python3 phase2c_restricted.py            # -> figures/phase2c_restricted_cp.png

echo "==> [6/9] figures 1-6 in preprint numbering (159-week window)"
python3 regenerate_figs.py               # -> figures/Fig1..Fig6 (Fig1 = base version)
python3 make_fig1_overlay.py             # -> figures/Fig1_wave_template.* (per-sentinel overlay)

echo "==> [7/9] figure 7 in preprint numbering, change-point panel (159-week window)"
python3 make_fig7.py                     # -> figures/Fig7_changepoint.{png,pdf}

echo "==> [8/9] figures 1, 6 and 8 of the submission, from the full 170-week record"
python3 make_figs_extended.py            # -> figures/submission/Fig{1,6,8}_*.{png,pdf}

echo "==> [9/11] supplementary figure S1 (adjacency specification)"
python3 make_figS1.py                    # -> figures/FigS1_adjacency.png

echo "==> [10/11] supplementary figure S2 (7 adjacency-matrix variants, 159-week window)"
python3 make_figS2.py                    # -> figures/FigS2_robustness.png

echo "==> [11/11] assemble figures/submission/ in submission numbering"
python3 export_submission_figures.py     # -> figures/submission/Fig1..Fig8

echo ""
echo "Done."
echo "  results/            .npz artefacts"
echo "  figures/            preprint numbering, 159-week window (incl. FigS1/FigS2)"
echo "  figures/submission/ manuscript numbering, Figures 1-8 as submitted"
