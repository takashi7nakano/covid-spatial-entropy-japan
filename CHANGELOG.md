# Changelog

## 1.1.1 — 2026-08-24

Maintenance release. No analysis result changes; the numbers reported in the
manuscript are unchanged. This release makes the pipeline run end to end from a
clean clone and extends it to cover every quantity the manuscript reports.

### Fixed

- **`src/make_figs_extended.py` aborted `reproduce.sh`.** The loader asserted
  eleven out-of-sample weeks, a value left over from the pre-freeze record. The
  archived data contain fourteen (2026-W20 to 2026-W33), so stage 8 raised
  `AssertionError: expected 11 out-of-sample weeks` and the run stopped there.
  A clean clone therefore could not reproduce Figures 1, 6 and 8, or any later
  stage. The assertion now matches the archived data.
- Stale references to a "170-week record" and to "11 out-of-sample weeks" in
  `src/make_figs_extended.py`, `src/export_submission_figures.py`,
  `reproduce.sh` and `README.md`. The frozen record is 173 weeks.
- Stage numbering in `reproduce.sh` was inconsistent (`[1/9]` … `[11/11]`).

### Added

- **`src/exploratory_measures.py` (stage 12).** The previous releases covered
  the 159-week estimation pipeline and the figures, but not the exploratory and
  descriptive quantities the manuscript also reports, so "the complete analysis
  code" overstated what was archived. This script regenerates, from the archived
  data alone: the cross-correlation profile between *S* and log national
  amplitude; the post-transition summary of *S* with its moving-block-bootstrap
  difference; the amplitude-adjusted era regression with boundary sensitivity
  and the amplitude-matched permutation test; the coefficient of variation and
  the synchrony fraction φ; the off-season participation ratios of
  Supplementary Table S11.1; and the elapsed intervals of Supplementary
  Table S14.1.
- The principal era boundary for the amplitude-adjusted regression
  (1 January 2026) is now stated explicitly in the code. Taking the change point
  instead gives −0.160 rather than the reported −0.170.

## 1.1.0 — 2026-08-21

Release accompanying the journal submission of the manuscript. It contains the
record as frozen for submission: 173 weeks, ending four weeks after the summer
2026 national peak (peak + 4), the phase at which spatial uniformity is maximal
on average in the pre-transition wave template.

### Fixed

- **Moran's I in `src/common.py` and `src/phase1_changepoint.py`.** These two
  modules used an estimator that took row-standardised weights, divided the
  variance term by *n* a second time, and normalised by the row sum of the
  standardised matrix. Running `reproduce.sh` therefore did **not** reproduce
  Table 2 of the manuscript: the values came out roughly 55× too large, and the
  discrepancy was not a constant factor — the ratio to the correct value ranged
  from −22 to +145 across weeks and changed sign, so ordering and significance
  were not preserved either. The post/pre ratio it produced was 1.399 against
  the published 1.496.

  Both modules now use the standard estimator

  ```
  I = (n / S0) * (z' W z) / (z' z),   z = x - mean(x)
  ```

  with binary symmetric weights `W` and `S0 = sum(W)`, which is what
  `src/regenerate_figs.py::moran_t` always used and what produced the published
  figures and Table 2. After the fix `reproduce.sh` reproduces Table 2 exactly:
  τ = 0 gives 0.313 (pre) and 0.468 (post); τ = 8 gives 0.074 and 0.418; the
  persistence-ratio difference is +0.619 with a 95% bootstrap CI of
  [+0.315, +0.897].

  No published number changes as a result of this fix. The defect was confined
  to the two modules above and never entered the manuscript.

### Added

- `data/pref_full_n33.csv`, `data/combined_n33.csv` — the full 173-week record
  (2023-W17 .. 2026-W33). The first 159 weeks are identical to the `n19` files;
  the additional 14 weeks are the out-of-sample confirmation set and are flagged
  as such in the `window` column.
- `src/make_figs_extended.py` — Figures 1, 6 and 8 of the submitted manuscript,
  the only script that reads the out-of-sample weeks. It performs no estimation.
- `src/export_submission_figures.py` — assembles `figures/submission/` under the
  figure numbers used in the submitted manuscript.

### Changed

- `reproduce.sh` now has nine stages rather than seven, and states for each
  stage which data window it uses.
- Figures are written at **300 dpi** with a PDF companion for every panel; the
  earlier 180/200 dpi PNG-only output did not meet the journal's figure
  requirements. The figure content is unchanged — the pre-fix and post-fix PNGs
  are byte-identical at the original resolution.
- Manuscript title updated: "159-week" removed, since the analysis now reports
  an out-of-sample test on weeks outside that window.
- README rewritten to document both figure numbering schemes and the split
  between the estimation window and the out-of-sample weeks.

### Note on the manuscript numbers

The estimation window is unchanged from 1.0.1: the same 159 weeks, the same
change-point, the same harmonic regression, Moran's *I* and between-period
comparisons. Only the out-of-sample window is longer. Quantities that are fitted
over all available weeks therefore differ slightly from any figure quoted in the
preprint: the amplitude-adjusted era coefficient is −0.170 (SE 0.019), the
post-change-point count is 38 weeks, and the elapsed interval since the entropy
ceiling was last attained is 43 weeks. The entropy-ceiling criterion itself is
unchanged: 0 of 38 post-change-point weeks attain *S* ≥ 3.80, against 49 of the
135 pre-transition weeks.

## 1.0.1 — 2026-05-27

Initial archived release accompanying the medRxiv preprint
(doi:10.64898/2026.05.24.26353972, posted 26 May 2026). Frozen 159-week
estimation window, 2023-W17 .. 2026-W19.
