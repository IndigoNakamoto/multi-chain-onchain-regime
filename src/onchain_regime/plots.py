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


def all_plots(frames, episodes, rel) -> dict[str, Figure]:
    figs = {
        "mvrv_overlay": plot_mvrv_overlay(frames, episodes),
        "supply_in_profit_overlay": plot_sip_overlay(frames, episodes),
        "price_vs_realized": plot_price_vs_realized(frames, episodes),
    }
    if rel is not None:
        figs["relative_mvrv"] = plot_relative_mvrv(rel, episodes)
    return figs
