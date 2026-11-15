"""Comparative plots. Every function returns a matplotlib Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from onchain_regime.assets import get_asset


def _shade(ax, episodes: pd.DataFrame | None, color: str, alpha: float = 0.15) -> None:
    if episodes is None:
        return
    for _, ep in episodes.iterrows():
        ax.axvspan(ep["start"], ep["end"] + pd.Timedelta(days=1), color=color, alpha=alpha, lw=0)


def _overlay(
    frames: dict[str, pd.DataFrame],
    column: str,
    episodes: dict[str, pd.DataFrame] | None,
    title: str,
    ylabel: str,
    log: bool = False,
    hline: float | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for key, df in frames.items():
        a = get_asset(key)
        s = df[column].dropna()
        ax.plot(s.index, s, color=a.color, lw=1.1, label=f"{a.name} (now {s.iloc[-1]:.3g})")
        _shade(ax, (episodes or {}).get(key), a.color)
        ax.scatter([s.index[-1]], [s.iloc[-1]], color=a.color, zorder=5, s=25)
    if hline is not None:
        ax.axhline(hline, color="grey", ls="--", lw=0.8)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_mvrv_overlay(frames, episodes=None) -> Figure:
    return _overlay(frames, "mvrv", episodes, "MVRV (shaded: matched low-MVRV / low supply-in-profit episodes)", "MVRV (log)", log=True, hline=1.0)


def plot_sip_overlay(frames, episodes=None) -> Figure:
    return _overlay(frames, "supply_in_profit_share", episodes, "Supply in profit (shaded: matched episodes)", "% of supply in profit", hline=50.0)


def plot_relative_mvrv(rel: pd.DataFrame, episodes: dict[str, pd.DataFrame] | None = None) -> Figure:
    num, den = rel.attrs["pair"]
    a_num, a_den = get_asset(num), get_asset(den)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, height_ratios=[2, 1])
    r = rel["relative_mvrv"]
    ax.plot(r.index, r, color="black", lw=1)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.axhline(r.median(), color="tab:blue", ls=":", lw=0.9, label=f"median {r.median():.2f}")
    ax.set_yscale("log")
    for key in (num, den):
        _shade(ax, (episodes or {}).get(key), get_asset(key).color, alpha=0.12)
    ax.set_title(
        f"Relative MVRV: {a_num.name} / {a_den.name} (now {r.iloc[-1]:.2f}, "
        f"p{rel.attrs['current_pct']:.0f} of history)"
    )
    ax.set_ylabel("MVRV ratio (log)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax2.plot(rel.index, rel["sip_spread"], color="tab:purple", lw=0.9)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel(f"SIP spread\n{num.upper()} - {den.upper()} (pp)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_price_vs_realized(frames, episodes=None) -> Figure:
    fig, axes = plt.subplots(len(frames), 1, figsize=(13, 4.2 * len(frames)), squeeze=False)
    for ax, (key, df) in zip(axes[:, 0], frames.items()):
        a = get_asset(key)
        ax.plot(df.index, df["price_close"], color=a.color, lw=1, label="Price close")
        ax.plot(df.index, df["realized_price"], color="black", lw=1.1, label="Realized price")
        ax.fill_between(df.index, df["price_close"], df["realized_price"], where=df["price_close"] < df["realized_price"], color="tab:red", alpha=0.2, label="Price < realized")
        _shade(ax, (episodes or {}).get(key), a.color)
        ax.set_yscale("log")
        ax.set_title(
            f"{a.name}: price \\${df['price_close'].iloc[-1]:,.2f} vs realized \\${df['realized_price'].iloc[-1]:,.2f}"
        )
        ax.set_ylabel("USD (log)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_equity_drawdown(backtests: dict[str, pd.DataFrame]) -> Figure:
    n = len(backtests)
    fig, axes = plt.subplots(2 * n, 1, figsize=(13, 5.5 * n), height_ratios=[2, 1] * n, squeeze=False)
    for i, (key, bt) in enumerate(backtests.items()):
        a = get_asset(key)
        ax, ax_dd = axes[2 * i, 0], axes[2 * i + 1, 0]
        ax.plot(bt.index, bt["buy_hold_equity"], color="grey", lw=1, label="Buy and hold")
        ax.plot(bt.index, bt["strategy_equity"], color=a.color, lw=1.4, label="Regime strategy")
        in_mkt = bt["position"] > 0
        ax.fill_between(bt.index, 0, 1, where=in_mkt, transform=ax.get_xaxis_transform(), color=a.color, alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_title(
            f"{a.name}: both <= p{bt.attrs['threshold']:g} point-in-time, exit {bt.attrs['exit']} "
            f"(strategy {bt['strategy_equity'].iloc[-1]:,.1f}x vs buy-and-hold {bt['buy_hold_equity'].iloc[-1]:,.1f}x)"
        )
        ax.set_ylabel("Equity (log)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
        ax_dd.fill_between(bt.index, bt["buy_hold_drawdown"], 0, color="grey", alpha=0.35, lw=0, label="Buy and hold")
        ax_dd.plot(bt.index, bt["strategy_drawdown"], color=a.color, lw=1, label="Regime strategy")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_dd.grid(alpha=0.25)
        ax_dd.legend(loc="lower left")
    fig.tight_layout()
    return fig


def all_plots(frames, episodes, rel, backtests=None) -> dict[str, Figure]:
    figs = {
        "mvrv_overlay": plot_mvrv_overlay(frames, episodes),
        "supply_in_profit_overlay": plot_sip_overlay(frames, episodes),
        "price_vs_realized": plot_price_vs_realized(frames, episodes),
    }
    if rel is not None:
        figs["relative_mvrv"] = plot_relative_mvrv(rel, episodes)
    if backtests:
        figs["equity_drawdown"] = plot_equity_drawdown(backtests)
    return figs
