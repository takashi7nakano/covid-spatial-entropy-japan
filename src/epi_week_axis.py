"""
Utility module: epi_week_axis
Functions to format a datetime axis with epi-week (YYYY-Www) tick labels.
"""
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


def date_to_epi(ts):
    """Return ISO epi-week string 'YYYY-Www' for a Timestamp."""
    iso = pd.Timestamp(ts).isocalendar()
    return f'{iso.year}-W{iso.week:02d}'


def set_epi_week_xaxis(ax, start_date, end_date, n_ticks=8):
    """Replace ax's X axis ticks with epi-week labels.
    start_date, end_date: pd.Timestamp range to cover.
    """
    # Generate evenly spaced tick positions
    tick_dates = pd.date_range(start_date, end_date, periods=n_ticks)
    tick_positions = [mdates.date2num(d) for d in tick_dates]
    tick_labels = [date_to_epi(d) for d in tick_dates]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha='right', fontsize=8)
