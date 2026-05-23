# covid-spatial-entropy-japan

Reproducible analysis pipeline for the study:

> **Two anti-phase spatial modes and a candidate spatial-persistence regime
> transition of SARS-CoV-2 in Japan: a 159-week prefecture-level sentinel
> surveillance study.**
> Nakano T, Onozuka D, Ikeda Y, Washiyama K, Takashima Y.

This repository contains the data and code needed to reproduce every
quantitative result and figure in the manuscript from the frozen input data
(159 weeks, 2023-W17 to 2026-W19; "n19").

> Preprint: medRxiv DOI — *to be added once posted*.
> Archived release: Zenodo DOI — *to be added on tagging* (see "Citing" below).

## What the pipeline does

The stages map one-to-one onto the manuscript's "Software and reproducibility"
description:

1. **Data ingestion** — load the prefecture-share table and adjacency edge list.
2. **Shannon pseudo-entropy `S(t)`** — spatial concentration of weekly reports.
3. **Moran's I** — global spatial autocorrelation, per week (row-standardised
   Queen contiguity).
4. **Harmonic regression** — K=2 annual harmonic model on day-of-year; residual
   series for change-point tests.
5. **Change-point detection** — CUSUM, PELT, Bai–Perron, and BOCPD; the
   principal estimate uses a restricted entropy-ceiling-failure search.
6. **Sensitivity analyses** — alternative weights, harmonic order K∈{1..6},
   moving-block-bootstrap block length, leave-one-prefecture-out, threshold
   sensitivity.
7. **Figure generation** — Figures 1–7, including the Figure 1A per-sentinel
   case-magnitude overlay.

All permutation/bootstrap routines use a fixed random seed (42); the principal
change-point is **2025-W48 (2025-11-27)**.

## Repository layout

```
covid-spatial-entropy-japan/
├── data/
│   ├── pref_full_n19.csv        # date, S, 47 prefecture shares p_i(t); 159 weeks
│   ├── adjacency_edges.csv      # 92 undirected Queen-contiguity edges
│   └── combined_n19.csv         # date, S, cases (= Σ_i x_i), era; for Fig 1A overlay
├── src/
│   ├── config.py                # repo-relative paths + frozen constants
│   ├── common.py                # data ingestion, W, S, Moran's I, harmonic regression
│   ├── phase1_changepoint.py    # stage 1: series + 3-method change-point  -> results npz
│   ├── phase1c_robustness.py    # stage 2: CP robustness
│   ├── phase2_principal.py      # stage 3: principal pre/post re-analysis
│   ├── phase2b_full.py          # stage 4: full sensitivity analyses
│   ├── phase2c_restricted.py    # stage 5: restricted entropy-ceiling CP
│   ├── regenerate_figs.py       # stage 6: Figures 1–6
│   ├── make_fig1_overlay.py     # stage 6: published Figure 1A (per-sentinel overlay)
│   ├── make_fig7.py             # stage 7: Figure 7
│   └── epi_week_axis.py         # epidemiological-week x-axis helper
├── results/                     # generated .npz artefacts (created on run)
├── figures/                     # generated figures (created on run)
├── requirements.txt
├── reproduce.sh                 # run the whole pipeline in order
├── CITATION.cff
└── LICENSE                      # MIT (code) + CC-BY 4.0 (data)
```

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
bash reproduce.sh
```

Outputs land in `results/` (`.npz`) and `figures/` (PNG/PDF). Individual stages
can also be run directly, e.g. `cd src && python3 phase1_changepoint.py`; stages
2–7 depend on `results/phase1_changepoint_results.npz` produced by stage 1.

A sanity check after running stage 1: `S(t)` should span **3.5412–3.8361** over
159 weeks, and the restricted change-point (stage 5) should be **2025-11-27
(2025-W48)**, stable across penalties and entropy thresholds.

## Data provenance

`pref_full_n19.csv` is derived from publicly available weekly COVID-19 sentinel
surveillance reports published by the Japanese Ministry of Health, Labour and
Welfare (MHLW). Values are per-sentinel report rates normalised to weekly
prefecture shares `p_i(t)` (each row sums to 1). `combined_n19.csv` additionally
provides the unweighted national mean magnitude `cases = Σ_i x_i(t)`. No
population weights, sentinel counts, or absolute case counts are included.
`adjacency_edges.csv` is the baseline land-contiguity (Queen) adjacency used
throughout; Okinawa is an isolated node (no land neighbours) by construction.

## Citing

Please cite the preprint (medRxiv DOI, to be added) and the archived code
release (Zenodo DOI, to be added). See `CITATION.cff` for machine-readable
metadata.

## Licence

Code is released under the MIT License; data under CC-BY 4.0. See `LICENSE`.
