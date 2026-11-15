"""Long/flat backtest of the point-in-time low-valuation regime vs buy-and-hold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from onchain_regime.episodes import MVRV, Rule, matching_mask
from onchain_regime.stats import expanding_percentile

DAYS_PER_YEAR = 365


def regime_signal(df: pd.DataFrame, threshold: float = 10.0, min_history: int = 365) -> pd.Series:
    """True on days where MVRV and supply-in-profit are both <= ``threshold`` point-in-time percentile."""
    return matching_mask(df, Rule("pctl", threshold, min_history))


def positions(df: pd.DataFrame, signal: pd.Series, exit: str = "hold:180", min_history: int = 365) -> pd.Series:
    """Position (1 long / 0 flat) for each day, using only information through the prior day.

    ``exit="hold:N"``: stay long until N days after the last signal day.
    ``exit="mvrv_pct:P"``: stay long until MVRV's point-in-time percentile reaches P.
    """
    sig = signal.reindex(df.index, fill_value=False).astype(float)
    kind, _, arg = exit.partition(":")
    if kind == "hold":
        days = int(arg)
        pos = sig.shift(1, fill_value=0.0).rolling(f"{days}D").max()
    elif kind == "mvrv_pct":
        level = float(arg)
        hot = (expanding_percentile(df[MVRV], min_history) >= level).to_numpy()
        s = sig.to_numpy().astype(bool)
        out = np.zeros(len(df))
        holding = False
        for i in range(1, len(df)):
            if holding and hot[i - 1]:
                holding = False
            if not holding and s[i - 1]:
                holding = True
            out[i] = holding
        pos = pd.Series(out, index=df.index)
    else:
        raise ValueError(f"Unknown exit {exit!r}; use 'hold:<days>' or 'mvrv_pct:<pct>'")
    return pos.fillna(0.0).rename("position")


def backtest(
    df: pd.DataFrame,
    threshold: float = 10.0,
    exit: str = "hold:180",
    min_history: int = 365,
) -> pd.DataFrame:
    """Daily equity and drawdown for the regime strategy and buy-and-hold.

    Starts once point-in-time percentiles exist, so both legs cover the same window.
    Flat days earn zero.
    """
    signal = regime_signal(df, threshold, min_history)
    pos = positions(df, signal, exit, min_history)
    start = expanding_percentile(df[MVRV], min_history).first_valid_index()
    price = df["price_close"].loc[start:]
    ret = price.pct_change().fillna(0.0)
    pos = pos.loc[start:]
    out = pd.DataFrame({"price_close": price, "signal": signal.loc[start:], "position": pos})
    out["strategy_ret"] = pos * ret
    out["buy_hold_ret"] = ret
    for leg in ("strategy", "buy_hold"):
        eq = (1 + out[f"{leg}_ret"]).cumprod()
        out[f"{leg}_equity"] = eq
        out[f"{leg}_drawdown"] = eq / eq.cummax() - 1
    out.attrs.update(threshold=threshold, exit=exit)
    return out


def performance(bt: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per leg over the backtest window."""
    years = (bt.index[-1] - bt.index[0]).days / DAYS_PER_YEAR
    rows = {}
    for leg in ("strategy", "buy_hold"):
        r = bt[f"{leg}_ret"]
        eq = bt[f"{leg}_equity"]
        in_mkt = bt["position"] if leg == "strategy" else pd.Series(1.0, index=bt.index)
        rows[leg] = {
            "start": bt.index[0].date(),
            "end": bt.index[-1].date(),
            "multiple": eq.iloc[-1],
            "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
            "max_drawdown": bt[f"{leg}_drawdown"].min(),
            "ann_vol": r.std() * np.sqrt(DAYS_PER_YEAR),
            "time_in_market": in_mkt.mean(),
            "trades": int((in_mkt.diff() > 0).sum() + (in_mkt.iloc[0] > 0)),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["calmar"] = out["cagr"] / out["max_drawdown"].abs()
    out.index.name = "leg"
    return out


def trades(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per long stint: entry/exit dates and return."""
    pos = bt["position"].to_numpy() > 0
    rows = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i]:
            j = i
            while j + 1 < n and pos[j + 1]:
                j += 1
            entry_px = bt["price_close"].iloc[i - 1] if i > 0 else bt["price_close"].iloc[i]
            rows.append(
                {
                    "entry": bt.index[i].date(),
                    "exit": bt.index[j].date(),
                    "days": j - i + 1,
                    "return": bt["price_close"].iloc[j] / entry_px - 1,
                    "open": j == n - 1,
                }
            )
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows, columns=["entry", "exit", "days", "return", "open"])
