# covid-spatial-entropy-japan

Reproducible analysis pipeline for the study:

> **Two anti-phase spatial modes and a candidate spatial-persistence regime
> transition of SARS-CoV-2 in Japan: a prefecture-level sentinel surveillance
> study.**
> Nakano T, Onozuka D, Ikeda Y, Washiyama K, Takashima Y.

This repository contains the data and code needed to reproduce every
quantitative result and figure in the manuscript.

> Preprint: medRxiv DOI: 10.64898/2026.05.24.26353972 (https://doi.org/10.64898/2026.05.24.26353972), posted 26 May 2026
> Archived release v1.1.0: Zenodo DOI: 10.5281/zenodo.22051486 (this version — the one the manuscript cites)
> All versions: Zenodo concept DOI: 10.5281/zenodo.20359404 (always resolves to the latest version)

See `CHANGELOG.md` for what changed in the current release. Version 1.1.0
corrects a defect in the Moran's I implementation used by two of the pipeline
modules; no published number is affected, but versions before 1.1.0 do not
reproduce Table 2.

## Two data windows

The distinction matters for reading the code, so it is worth stating plainly.

| | weeks | file | used for |
| --- | --- | --- | --- |
| **Estimation window** | 159 (2023-W17 .. 2026-W19) | `data/*_n19.csv` | everything: change-point detection, harmonic regression, Moran's I, every between-period comparison |
| **Full record** | 173 (2023-W17 .. 2026-W33) | `data/*_n33.csv` | the out-of-sample test and the exploratory measures only |

The estimation window is frozen and identical to that of version 1 of the
preprint. Nothing was re-estimated when the further 14 weeks became available.
Only `src/make_figs_extended.py` reads those weeks, and it fits nothing.

## What the pipeline does

The stages map one-to-one onto the manuscript's "Software and reproducibility"
description:

1. **Data ingestion** — load the prefecture-share table and adjacency edge list.
2. **Shannon pseudo-entropy `S(t)`** — spatial concentration of weekly reports.
3. **Moran's I** — global spatial autocorrelation, per week, using
   `I = (n/S0)·z'Wz/z'z` with binary symmetric Queen-contiguity weights.
4. **Harmonic regression** — K=2 annual harmonic model on day-of-year; residual
   series for change-point tests.
5. **Change-point detection** — CUSUM, PELT, Bai–Perron, and BOCPD; the
   principal estimate uses a restricted entropy-ceiling-failure search.
6. **Sensitivity analyses** — alternative weights, harmonic order K∈{1..6},
   moving-block-bootstrap block length, leave-one-prefecture-out, threshold
   sensitivity.
7. **Figure generation** — Figures 1–8 of the submitted manuscript.

All permutation/bootstrap routines use a fixed random seed (42); the principal
change-point is **2025-W48 (2025-11-27)**.

## Repository layout

```
covid-spatial-entropy-japan/
├── data/
│   ├── pref_full_n19.csv        # date, S, 47 prefecture shares p_i(t); 159 weeks
│   ├── combined_n19.csv         # date, S, cases (= Σ_i x_i), era; 159 weeks
│   ├── pref_full_n33.csv        # same columns, 173 weeks
│   ├── combined_n33.csv         # date, S, cases, window; 173 weeks
│   └── adjacency_edges.csv      # 92 undirected Queen-contiguity edges
├── src/
│   ├── config.py                # repo-relative paths + frozen constants
│   ├── common.py                # data ingestion, W, S, Moran's I, harmonic regression
│   ├── phase1_changepoint.py    # stage 1: series + 3-method change-point  -> results npz
│   ├── phase1c_robustness.py    # stage 2: CP robustness
│   ├── phase2_principal.py      # stage 3: principal pre/post re-analysis
│   ├── phase2b_full.py          # stage 4: full sensitivity analyses
│   ├── phase2c_restricted.py    # stage 5: restricted entropy-ceiling CP
│   ├── regenerate_figs.py       # stage 6: figures in preprint numbering
│   ├── make_fig1_overlay.py     # stage 6: per-sentinel overlay panel
│   ├── make_fig7.py             # stage 7: change-point panel
│   ├── make_figs_extended.py    # stage 8: submission Figures 1, 6, 8 (173 weeks)
│   ├── export_submission_figures.py  # stage 11: assemble figures/submission/
│   ├── exploratory_measures.py  # stage 12: exploratory / descriptive measures
│   └── epi_week_axis.py         # epidemiological-week x-axis helper
├── results/                     # generated .npz artefacts (created on run)
├── figures/                     # generated figures (created on run)
│   └── submission/              # the eight figures as submitted
├── requirements.txt
├── reproduce.sh                 # run the whole pipeline in order
├── CHANGELOG.md
├── CITATION.cff
└── LICENSE                      # MIT (code) + CC-BY 4.0 (data)
```

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
bash reproduce.sh
```

Outputs land in `results/` (`.npz`), `figures/` (preprint numbering) and
`figures/submission/` (manuscript numbering). Individual stages can also be run
directly, e.g. `cd src && python3 phase1_changepoint.py`; stages 2–7 depend on
`results/phase1_changepoint_results.npz` produced by stage 1, and stage 9
depends on stages 6–8.

Sanity checks after a full run:

| quantity | expected |
| --- | --- |
| `S(t)` range over the 159-week window | 3.5412 – 3.8361 |
| restricted change-point (stage 5) | 2025-11-27 (2025-W48), stable across penalties and thresholds |
| Moran's I, τ = 0 (Table 2) | 0.313 pre / 0.468 post |
| Moran's I, τ = 8 (Table 2) | 0.074 pre / 0.418 post |
| persistence-ratio difference | +0.619, 95% CI [+0.315, +0.897] |
| within-wave max `S`, five pre-transition waves | 3.8361, 3.8268, 3.8273, 3.8196, 3.8262 (all ≥ 3.80) |
| within-wave max `S`, two post-transition waves | 3.6890, 3.6770 (neither ≥ 3.80) |

## Figure numbering

Two numbering schemes exist for the same images. The figures in the submitted
manuscript are numbered in order of first mention in the main text, as BMC
formatting requires; the preprint numbered them in a different order, and the
analysis scripts still emit the preprint names. `figures/submission/` holds the
submission-numbered set, and is what the manuscript refers to.

| Script output (preprint numbering) | Submission | Content | Window |
| --- | --- | --- | --- |
| `Fig1_wave_template` | **Figure 1** | Wave template, S(t) and national per-sentinel mean | 173 wk |
| `Fig7_changepoint` | **Figure 2** | Entropy-ceiling-failure change-point detection | 159 wk |
| `Fig2_two_modes` | **Figure 3** | Two spatial modes (S vs Moran's I) | 159 wk |
| `Fig3_seasonal_antiphase` | **Figure 4** | Seasonal anti-phase structure | 159 wk |
| `Fig4_regime_transition` | **Figure 5** | Post-transition departure | 159 wk |
| `Fig5_phase_plane` | **Figure 6** | Phase plane | 173 wk |
| `Fig6_lagged_moran` | **Figure 7** | Time-lagged Moran's I | 159 wk |
| — (new) | **Figure 8** | Out-of-sample test of the pre-specified criterion | 173 wk |

Figures 1, 6 and 8 are generated from the full record by
`src/make_figs_extended.py`, which writes straight into `figures/submission/`.
The 159-week versions of Figures 1 and 6 that `regenerate_figs.py` emits into
`figures/` are superseded and are not part of the submission.

## Data provenance

`pref_full_n19.csv` and `pref_full_n33.csv` are derived from publicly available
weekly COVID-19 sentinel surveillance reports published by the Japanese Ministry
of Health, Labour and Welfare (MHLW). Values are per-sentinel report rates
normalised to weekly prefecture shares `p_i(t)` (each row sums to 1). The
`combined_*` files additionally provide the unweighted national total
`cases = Σ_i x_i(t)`; because each prefecture value is published to two
decimals, this total is exact and the rate matrix can be recovered as
`x_i(t) = p_i(t) · cases(t)`. No population weights, sentinel counts, or
absolute case counts are included.

`adjacency_edges.csv` is the baseline land-contiguity (Queen) adjacency used
throughout; Okinawa is an isolated node (no land neighbours) by construction.

The `era` column of `combined_n19.csv` is a legacy three-valued label carried
over from the preprint and does not mark the change-point. Use the `window`
column of `combined_n33.csv` (`estimation` / `out-of-sample`) for the
distinction that matters in the manuscript.

## Citing

Please cite the preprint (medRxiv DOI: 10.64898/2026.05.24.26353972) and the archived code
release. Cite the specific version you used — for the release accompanying the submitted
manuscript this is Zenodo DOI 10.5281/zenodo.22051486 (v1.1.0). To refer to the software
irrespective of version, use the concept DOI 10.5281/zenodo.20359404, which always resolves
to the latest release. See `CITATION.cff` for machine-readable metadata.

## Licence

Code is released under the MIT License; data under CC-BY 4.0. See `LICENSE`.
