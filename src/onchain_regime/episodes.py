"""Detect historical low-MVRV / low supply-in-profit episodes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from onchain_regime.stats import expanding_percentile

MVRV = "mvrv"
SIP = "supply_in_profit_share"


@dataclass(frozen=True)
class Rule:
    """Episode condition.

    ``kind="current"``: MVRV and supply-in-profit both at or below today's values.
    ``kind="pctl"``: both at or below ``threshold`` percentile, measured point-in-time
    (expanding window) so the flag never uses future data.
    """

    kind: str = "current"
    threshold: float = 10.0
    min_history: int = 365

    @classmethod
    def parse(cls, text: str) -> Rule:
        """``"current"`` or ``"pctl:<threshold>"``."""
        if text == "current":
            return cls("current")
        if text.startswith("pctl:"):
            return cls("pctl", float(text.split(":", 1)[1]))
        raise ValueError(f"Unknown rule {text!r}; use 'current' or 'pctl:<n>'")

    @property
    def label(self) -> str:
        return "at/below current" if self.kind == "current" else f"both <= p{self.threshold:g} (point-in-time)"


def matching_mask(df: pd.DataFrame, rule: Rule) -> pd.Series:
    both = df[[MVRV, SIP]].dropna()
    if rule.kind == "current":
        cur = both.iloc[-1]
        mask = (both[MVRV] <= cur[MVRV]) & (both[SIP] <= cur[SIP])
    elif rule.kind == "pctl":
        p_m = expanding_percentile(both[MVRV], rule.min_history)
        p_s = expanding_percentile(both[SIP], rule.min_history)
        mask = (p_m <= rule.threshold) & (p_s <= rule.threshold)
    else:
        raise ValueError(rule.kind)
    return mask.reindex(df.index, fill_value=False).astype(bool)


def cluster_episodes(df: pd.DataFrame, mask: pd.Series, gap_days: int = 30) -> pd.DataFrame:
    """Group matching days into episodes; days within ``gap_days`` of each other merge."""
    days = mask.index[mask.to_numpy()]
    cols = ["start", "end", "matched_days", "span_days", "min_mvrv", "min_sip", "start_price", "is_current"]
    if len(days) == 0:
        return pd.DataFrame(columns=cols)
    groups = (pd.Series(days).diff().dt.days.fillna(0) > gap_days).cumsum()
    last_day = df.index[-1]
    rows = []
    for _, g in pd.Series(days).groupby(groups.to_numpy()):
        start, end = g.iloc[0], g.iloc[-1]
        window = df.loc[start:end]
        rows.append(
            {
                "start": start,
                "end": end,
                "matched_days": len(g),
                "span_days": (end - start).days + 1,
                "min_mvrv": window.loc[g, MVRV].min(),
                "min_sip": window.loc[g, SIP].min(),
                "start_price": df.at[start, "price_close"],
                "is_current": (last_day - end).days <= gap_days,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    out.index.name = "episode"
    return out


def find_episodes(df: pd.DataFrame, rule: Rule | str = "current", gap_days: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Return (episodes table, daily match mask)."""
    if isinstance(rule, str):
        rule = Rule.parse(rule)
    mask = matching_mask(df, rule)
    return cluster_episodes(df, mask, gap_days), mask
