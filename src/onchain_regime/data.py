"""Assemble per-asset metric frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from onchain_regime.assets import CORE_METRICS, AssetConfig, get_asset
from onchain_regime.client import DEFAULT_CACHE_DIR, BRKClient


def trim_to_valid(
    df: pd.DataFrame,
    required: tuple[str, ...] = CORE_METRICS,
    max_flat_days: int = 14,
) -> pd.DataFrame:
    """Drop leading rows that lack ``required`` data or carry a stale placeholder price.

    Early-chain rows (e.g. BTC supply_in_profit_share in 2009, before any price;
    LTC's constant $3.15 price from 2011-10 to 2013-05) would otherwise distort
    percentiles, since MVRV and supply-in-profit are derived from that price.
    """
    valid = df[list(required)].notna().all(axis=1)
    if not valid.any():
        return df.iloc[0:0]
    df = df.loc[valid.idxmax():]
    price = df["price_close"]
    flat = price.diff().eq(0)
    flat.iloc[0] = True
    if flat.iloc[: max_flat_days + 1].all():
        first_move = (~flat).idxmax()
        df = df.loc[first_move:]
    return df


def load_asset(
    asset: AssetConfig | str,
    index: str = "day1",
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    client: BRKClient | None = None,
) -> pd.DataFrame:
    """One column per canonical metric plus ``price_to_realized``, trimmed to valid history."""
    if isinstance(asset, str):
        asset = get_asset(asset)
    client = client or BRKClient(asset, cache_dir=cache_dir, refresh=refresh)
    cols = {metric: client.fetch(series_id, index) for metric, series_id in asset.series.items()}
    df = pd.DataFrame(cols)
    df["price_to_realized"] = df["price_close"] / df["realized_price"]
    df = trim_to_valid(df)
    df.attrs["asset"] = asset.key
    df.attrs["index"] = index
    return df


def load_assets(
    keys: list[str],
    index: str = "day1",
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    return {k: load_asset(k, index=index, cache_dir=cache_dir, refresh=refresh) for k in keys}
