#!/usr/bin/env python3
"""Assemble figures/submission/ using the figure numbers of the submitted manuscript.

Two numbering schemes exist for the same images:

  * the preprint numbering, which the analysis scripts emit into `figures/`;
  * the submission numbering, in which figures appear in order of first mention
    in the main text, as BMC formatting requires.

This script copies the five figures that come from the frozen 159-week window
into `figures/submission/` under their submission numbers. Figures 1, 6 and 8
are written there directly by `make_figs_extended.py`, which must be run first.

The result is a directory whose contents are byte-identical to the figure files
uploaded with the manuscript.
"""
import shutil
import sys

from config import FIGURES

SUBMISSION = FIGURES / 'submission'

# preprint file stem -> submission file stem
COPY = {
    'Fig7_changepoint':        'Fig2_changepoint',
    'Fig2_two_modes':          'Fig3_two_modes',
    'Fig3_seasonal_antiphase': 'Fig4_seasonal_antiphase',
    'Fig4_regime_transition':  'Fig5_regime_transition',
    'Fig6_lagged_moran':       'Fig7_lagged_moran',
}

# written directly by make_figs_extended.py, from the full 173-week record
FROM_EXTENDED = [
    'Fig1_wave_template',
    'Fig6_phase_plane',
    'Fig8_prospective_test',
]

# Emitted by the 159-week scripts but NOT part of the submission:
#   Fig1_wave_template  (regenerate_figs / make_fig1_overlay) - superseded by the
#                        173-week panel A, which is submission Figure 1
#   Fig5_phase_plane     (regenerate_figs) - superseded by the 173-week panel B,
#                        which is submission Figure 6
SUPERSEDED = ['Fig1_wave_template', 'Fig5_phase_plane']


def main():
    SUBMISSION.mkdir(exist_ok=True)
    missing = []

    for src_stem, dst_stem in COPY.items():
        found = False
        for ext in ('.png', '.pdf'):
            src = FIGURES / (src_stem + ext)
            if src.exists():
                shutil.copy2(src, SUBMISSION / (dst_stem + ext))
                found = True
                print('  %-26s -> submission/%s' % (src_stem + ext, dst_stem + ext))
        if not found:
            missing.append(src_stem + '.png')

    for stem in FROM_EXTENDED:
        if not (SUBMISSION / (stem + '.png')).exists():
            missing.append('submission/' + stem + '.png (run make_figs_extended.py)')

    if missing:
        print('\nMISSING:')
        for m in missing:
            print('  ' + m)
        sys.exit(1)

    pngs = sorted(p.name for p in SUBMISSION.glob('Fig*.png'))
    print('\nfigures/submission/ contains %d PNGs:' % len(pngs))
    for n in pngs:
        print('  ' + n)
    expected = {'Fig%d' % i for i in range(1, 9)}
    got = {n.split('_')[0] for n in pngs}
    if got != expected:
        print('\nFigure numbers are not 1-8: %s' % sorted(got))
        sys.exit(1)
    print('\nSuperseded by the 173-week versions, not submitted: %s'
          % ', '.join(s + '.png' for s in SUPERSEDED))
    print('OK')


if __name__ == '__main__':
    main()
