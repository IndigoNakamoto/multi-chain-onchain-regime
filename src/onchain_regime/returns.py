"""Forward returns after regime episodes."""

from __future__ import annotations

import pandas as pd

HORIZONS: tuple[int, ...] = (30, 90, 180)


def _col(h: int) -> str:
    return f"fwd_{h}d"


def forward_returns(price: pd.Series, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Simple forward return over ``h`` calendar days; NaN where the window is incomplete.

    Requires a daily index; the series is reindexed to a contiguous calendar so
    ``shift(-h)`` means ``h`` days, not ``h`` rows.
    """
    p = price.asfreq("D")
    out = pd.DataFrame({_col(h): p.shift(-h) / p - 1 for h in horizons})
    return out.reindex(price.index)


def episode_forward_returns(
    df: pd.DataFrame, episodes: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Forward returns measured from each episode's first matching day."""
    fwd = forward_returns(df["price_close"], horizons)
    out = episodes[["start", "end", "matched_days", "start_price", "is_current"]].copy()
    for h in horizons:
        out[_col(h)] = fwd[_col(h)].reindex(out["start"]).to_numpy()
    out["incomplete"] = out[[_col(h) for h in horizons]].isna().any(axis=1)
    return out


def matching_day_forward_returns(
    df: pd.DataFrame, mask: pd.Series, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Forward returns from every matching day (heavily overlapping, not independent)."""
    fwd = forward_returns(df["price_close"], horizons)
    return fwd.loc[mask.reindex(fwd.index, fill_value=False).to_numpy()]


def summarize(returns: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Per-horizon stats over completed windows only."""
    rows = []
    for h in horizons:
        r = returns[_col(h)]
        done = r.dropna()
        rows.append(
            {
                "horizon_days": h,
                "n": len(done),
                "n_incomplete": int(r.isna().sum()),
                "median": done.median(),
                "mean": done.mean(),
                "hit_rate": (done > 0).mean() if len(done) else float("nan"),
                "min": done.min(),
                "max": done.max(),
            }
        )
    return pd.DataFrame(rows).set_index("horizon_days")


def forward_return_report(
    df: pd.DataFrame, episodes: pd.DataFrame, mask: pd.Series, horizons: tuple[int, ...] = HORIZONS
) -> dict[str, pd.DataFrame]:
    """Episode-start, all-matching-day, and unconditional baseline summaries."""
    ep = episode_forward_returns(df, episodes, horizons)
    days = matching_day_forward_returns(df, mask, horizons)
    base = forward_returns(df["price_close"], horizons)
    return {
        "episodes": ep,
        "episode_summary": summarize(ep, horizons),
        "matching_day_summary": summarize(days, horizons),
        "baseline_summary": summarize(base, horizons),
    }
