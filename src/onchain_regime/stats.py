"""Percentile and rarity statistics."""

from __future__ import annotations

import pandas as pd

VALUATION_METRICS: tuple[str, ...] = (
    "mvrv",
    "supply_in_profit_share",
    "sth_mvrv",
    "lth_mvrv",
    "price_to_realized",
)


def percentile_of(series: pd.Series, value: float) -> float:
    """Share of non-null observations at or below ``value``, in percent."""
    s = series.dropna()
    if s.empty or pd.isna(value):
        return float("nan")
    return float((s <= value).mean() * 100)


def expanding_percentile(series: pd.Series, min_periods: int = 365) -> pd.Series:
    """Point-in-time percentile: each value ranked only against history up to that date."""
    s = series.dropna()
    ranks = s.expanding(min_periods=min_periods).rank(method="max", pct=True) * 100
    return ranks.reindex(series.index)


def current_percentiles(df: pd.DataFrame, metrics: tuple[str, ...] = VALUATION_METRICS) -> pd.DataFrame:
    """Latest value of each metric and its full-history percentile."""
    rows = []
    for m in metrics:
        if m not in df:
            continue
        s = df[m].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "metric": m,
                "date": s.index[-1].date(),
                "value": s.iloc[-1],
                "percentile": percentile_of(s, s.iloc[-1]),
                "history_start": s.index[0].date(),
                "n_obs": len(s),
                "min": s.min(),
                "median": s.median(),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def joint_rarity(df: pd.DataFrame, a: str = "mvrv", b: str = "supply_in_profit_share") -> dict:
    """Share of days where both ``a`` and ``b`` were at or below their latest values."""
    both = df[[a, b]].dropna()
    cur_a, cur_b = both[a].iloc[-1], both[b].iloc[-1]
    hits = (both[a] <= cur_a) & (both[b] <= cur_b)
    return {
        "date": both.index[-1].date(),
        a: cur_a,
        b: cur_b,
        "matching_days": int(hits.sum()),
        "total_days": len(both),
        "joint_percentile": float(hits.mean() * 100),
    }
