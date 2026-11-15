"""End-to-end analysis shared by the CLI and the notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from onchain_regime.assets import get_asset
from onchain_regime.backtest import backtest, performance, trades
from onchain_regime.client import DEFAULT_CACHE_DIR
from onchain_regime.compare import current_table, mask_overlap, regime_timing, relative_mvrv, relative_summary
from onchain_regime.data import load_asset
from onchain_regime.episodes import Rule, find_episodes
from onchain_regime.returns import HORIZONS, forward_return_report
from onchain_regime.snapshot import build_snapshot
from onchain_regime.stats import current_percentiles

REFERENCE_ASSET = "btc"


@dataclass
class AssetResult:
    daily: pd.DataFrame
    monthly: pd.DataFrame
    percentiles: pd.DataFrame
    episodes: pd.DataFrame
    mask: pd.Series
    returns: dict[str, pd.DataFrame]
    backtest: pd.DataFrame
    performance: pd.DataFrame
    trades: pd.DataFrame


@dataclass
class Results:
    rule: Rule
    assets: dict[str, AssetResult]
    current: pd.DataFrame
    relative: dict[str, pd.DataFrame] = field(default_factory=dict)
    relative_summary: pd.DataFrame | None = None
    timing: pd.DataFrame | None = None
    overlap: pd.DataFrame | None = None
    snapshot: dict = field(default_factory=dict)

    @property
    def backtests(self) -> dict[str, pd.DataFrame]:
        return {k: r.backtest for k, r in self.assets.items()}

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return {k: r.daily for k, r in self.assets.items()}

    @property
    def episodes(self) -> dict[str, pd.DataFrame]:
        return {k: r.episodes for k, r in self.assets.items()}


def run_analysis(
    asset_keys: list[str],
    rule: Rule | str = "current",
    refresh: bool = False,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    gap_days: int = 30,
    horizons: tuple[int, ...] = HORIZONS,
    bt_threshold: float = 10.0,
    bt_exit: str = "hold:180",
) -> Results:
    if isinstance(rule, str):
        rule = Rule.parse(rule)
    for k in asset_keys:
        a = get_asset(k)
        if not a.enabled:
            raise NotImplementedError(f"{a.name} is not enabled yet; see TODO({a.key}) in assets.py")

    assets: dict[str, AssetResult] = {}
    for k in asset_keys:
        daily = load_asset(k, "day1", cache_dir=cache_dir, refresh=refresh)
        monthly = load_asset(k, "month1", cache_dir=cache_dir, refresh=refresh)
        eps, mask = find_episodes(daily, rule, gap_days)
        bt = backtest(daily, threshold=bt_threshold, exit=bt_exit)
        assets[k] = AssetResult(
            daily=daily,
            monthly=monthly,
            percentiles=current_percentiles(daily),
            episodes=eps,
            mask=mask,
            returns=forward_return_report(daily, eps, mask, horizons),
            backtest=bt,
            performance=performance(bt),
            trades=trades(bt),
        )

    frames = {k: r.daily for k, r in assets.items()}
    res = Results(
        rule=rule,
        assets=assets,
        current=current_table(frames),
        snapshot=build_snapshot(frames, REFERENCE_ASSET, bt_threshold),
    )
    if REFERENCE_ASSET in assets and len(assets) > 1:
        others = [k for k in asset_keys if k != REFERENCE_ASSET]
        res.relative = {f"{k}_{REFERENCE_ASSET}": relative_mvrv(res.frames, k, REFERENCE_ASSET) for k in others}
        res.relative_summary = pd.DataFrame([relative_summary(r) for r in res.relative.values()]).set_index("pair")
        res.timing = regime_timing(res.episodes, REFERENCE_ASSET)
        res.overlap = mask_overlap({k: r.mask for k, r in assets.items()})
    return res
