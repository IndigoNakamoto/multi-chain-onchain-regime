"""Cross-asset comparisons."""

from __future__ import annotations

import pandas as pd

from onchain_regime.stats import VALUATION_METRICS, current_percentiles, expanding_percentile, joint_rarity, percentile_of


def current_table(frames: dict[str, pd.DataFrame], metrics: tuple[str, ...] = VALUATION_METRICS) -> pd.DataFrame:
    """One row per asset: latest value and full-history percentile for each metric."""
    rows = {}
    for key, df in frames.items():
        cur = current_percentiles(df, metrics)
        row = {"date": df.index[-1].date(), "price_close": df["price_close"].iloc[-1], "realized_price": df["realized_price"].iloc[-1]}
        for m in cur.index:
            row[m] = cur.at[m, "value"]
            row[f"{m}_pct"] = cur.at[m, "percentile"]
        row["joint_pct"] = joint_rarity(df)["joint_percentile"]
        row["history_start"] = df.index[0].date()
        rows[key] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "asset"
    return out


def relative_mvrv(frames: dict[str, pd.DataFrame], num: str = "ltc", den: str = "btc", min_periods: int = 365) -> pd.DataFrame:
    """Ratio of two assets' MVRV on overlapping dates, with full-history and point-in-time percentiles.

    Above 1: ``num`` trades at a richer premium to its realized price than ``den``.
    """
    joined = pd.concat([frames[num]["mvrv"].rename(num), frames[den]["mvrv"].rename(den)], axis=1).dropna()
    ratio = (joined[num] / joined[den]).rename("relative_mvrv")
    sip_spread = (
        frames[num]["supply_in_profit_share"] - frames[den]["supply_in_profit_share"]
    ).reindex(joined.index).rename("sip_spread")
    out = pd.concat([joined.add_suffix("_mvrv"), ratio, sip_spread], axis=1)
    out["relative_mvrv_pct_pit"] = expanding_percentile(ratio, min_periods)
    out.attrs["current_pct"] = percentile_of(ratio, ratio.iloc[-1])
    out.attrs["pair"] = (num, den)
    return out


def relative_summary(rel: pd.DataFrame) -> dict:
    num, den = rel.attrs["pair"]
    r = rel["relative_mvrv"]
    return {
        "pair": f"{num}/{den}",
        "date": r.index[-1].date(),
        "relative_mvrv": r.iloc[-1],
        "percentile": rel.attrs["current_pct"],
        "median": r.median(),
        "sip_spread": rel["sip_spread"].iloc[-1],
        "sip_spread_pct": percentile_of(rel["sip_spread"], rel["sip_spread"].iloc[-1]),
    }


def regime_timing(episodes: dict[str, pd.DataFrame], ref: str = "btc") -> pd.DataFrame:
    """Pair each non-reference episode with a reference episode and report the start lag.

    Pairs with the most-overlapping reference episode, else the nearest start.
    Positive ``lag_days``: the asset entered its episode after the reference asset.
    """
    ref_eps = episodes[ref]
    rows = []
    for key, eps in episodes.items():
        if key == ref or eps.empty or ref_eps.empty:
            continue
        for _, ep in eps.iterrows():
            gaps = (ep["start"] - ref_eps["start"]).dt.days
            overlaps = (
                (ref_eps["end"].where(ref_eps["end"] < ep["end"], ep["end"])
                 - ref_eps["start"].where(ref_eps["start"] > ep["start"], ep["start"])).dt.days + 1
            ).clip(lower=0)
            j = overlaps.idxmax() if overlaps.max() > 0 else gaps.abs().idxmin()
            ref_ep = ref_eps.loc[j]
            overlap = int(overlaps.loc[j])
            rows.append(
                {
                    "asset": key,
                    "start": ep["start"].date(),
                    "end": ep["end"].date(),
                    f"{ref}_start": ref_ep["start"].date(),
                    f"{ref}_end": ref_ep["end"].date(),
                    "lag_days": int(gaps.loc[j]),
                    "overlap_days": overlap,
                }
            )
    return pd.DataFrame(rows)


def mask_overlap(masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Days each asset was in-regime, and days all were simultaneously."""
    joined = pd.concat({k: m for k, m in masks.items()}, axis=1).dropna().astype(bool)
    stats = {k: int(joined[k].sum()) for k in joined}
    stats["all_simultaneous"] = int(joined.all(axis=1).sum())
    stats["common_days"] = len(joined)
    return pd.DataFrame([stats])
