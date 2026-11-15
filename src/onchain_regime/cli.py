"""Command line entry point: ``python -m onchain_regime run --assets ltc,btc``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from onchain_regime.assets import ASSETS  # noqa: E402
from onchain_regime.data import load_assets  # noqa: E402
from onchain_regime.pipeline import REFERENCE_ASSET, Results, run_analysis  # noqa: E402
from onchain_regime.plots import all_plots  # noqa: E402
from onchain_regime.snapshot import build_snapshot, to_json, to_markdown  # noqa: E402


def write_tables(res: Results, out: Path) -> list[Path]:
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(df: pd.DataFrame, name: str, index: bool = True) -> None:
        p = tdir / f"{name}.csv"
        df.to_csv(p, index=index)
        written.append(p)

    save(res.current, "current_snapshot")
    for k, r in res.assets.items():
        save(r.percentiles, f"{k}_percentiles")
        save(r.episodes, f"{k}_episodes")
        save(r.returns["episodes"], f"{k}_episode_forward_returns")
        for name in ("episode_summary", "matching_day_summary", "baseline_summary"):
            save(r.returns[name], f"{k}_{name}")
        save(r.daily, f"{k}_daily")
        save(r.monthly, f"{k}_monthly")
        save(r.backtest, f"{k}_backtest_daily")
        save(r.performance, f"{k}_backtest_performance")
        save(r.trades, f"{k}_backtest_trades", index=False)
    for pair, rel in res.relative.items():
        save(rel, f"relative_mvrv_{pair}")
    if res.relative_summary is not None:
        save(res.relative_summary, "relative_summary")
    if res.timing is not None:
        save(res.timing, "regime_timing", index=False)
    if res.overlap is not None:
        save(res.overlap, "regime_overlap", index=False)
    return written


def write_plots(res: Results, out: Path) -> list[Path]:
    pdir = out / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    rel = next(iter(res.relative.values()), None)
    written = []
    for name, fig in all_plots(res.frames, res.episodes, rel, res.backtests).items():
        p = pdir / f"{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)
    return written


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def summary_markdown(res: Results) -> str:
    lines = [
        "# On-chain regime summary",
        "",
        f"Episode rule: **{res.rule.label}**. Percentiles are full-history unless noted; "
        "forward returns use completed windows only.",
        "",
        "## Current snapshot",
        "",
    ]
    cols = ["date", "price_close", "realized_price", "mvrv", "mvrv_pct", "supply_in_profit_share",
            "supply_in_profit_share_pct", "sth_mvrv", "sth_mvrv_pct", "lth_mvrv", "lth_mvrv_pct", "joint_pct", "history_start"]
    lines += [_md(res.current[[c for c in cols if c in res.current]].T, ".3f"), ""]
    if res.relative_summary is not None:
        lines += ["## Relative valuation", "", _md(res.relative_summary), ""]
    for k, r in res.assets.items():
        lines += [f"## {ASSETS[k].name}", "", "### Matched episodes and forward returns", "",
                  _md(r.returns["episodes"]), "",
                  "### Forward returns from episode starts", "", _md(r.returns["episode_summary"]), "",
                  "### Forward returns from all matching days (overlapping windows)", "",
                  _md(r.returns["matching_day_summary"]), "",
                  "### Unconditional baseline (all days)", "", _md(r.returns["baseline_summary"]), ""]
    if res.timing is not None and not res.timing.empty:
        lines += ["## Regime timing vs BTC", "", res.timing.to_markdown(index=False), ""]
    if res.overlap is not None:
        lines += ["## Days in regime", "", res.overlap.to_markdown(index=False), ""]
    bt0 = next(iter(res.backtests.values()))
    lines += [f"## Backtest: both <= p{bt0.attrs['threshold']:g} point-in-time, exit {bt0.attrs['exit']}", "",
              "Long/flat, signal acted on the next day, zero return while flat.", ""]
    for k, r in res.assets.items():
        lines += [f"### {ASSETS[k].name}", "", _md(r.performance), "", r.trades.to_markdown(index=False, floatfmt=".3f"), ""]
    return "\n".join(lines)


def write_snapshot(snap: dict, out: Path) -> list[Path]:
    paths = [out / "current_regime.json", out / "current_regime.md"]
    paths[0].write_text(to_json(snap) + "\n")
    paths[1].write_text(to_markdown(snap))
    return paths


def _keys(text: str) -> list[str]:
    return [k.strip() for k in text.split(",") if k.strip()]


def cmd_run(args: argparse.Namespace) -> None:
    keys = _keys(args.assets)
    res = run_analysis(keys, rule=args.rule, refresh=args.refresh, cache_dir=args.cache_dir, gap_days=args.gap_days,
                       bt_threshold=args.bt_threshold, bt_exit=args.bt_exit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = write_tables(res, out)
    plots = write_plots(res, out)
    summary = out / "summary.md"
    summary.write_text(summary_markdown(res))
    write_snapshot(res.snapshot, out)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"Rule: {res.rule.label}\n")
        show = ["date", "mvrv", "mvrv_pct", "supply_in_profit_share", "supply_in_profit_share_pct",
                "sth_mvrv_pct", "lth_mvrv_pct", "joint_pct"]
        print(res.current[show].round(3).to_string(), "\n")
        if res.relative_summary is not None:
            print(res.relative_summary.round(3).to_string(), "\n")
        for k, r in res.assets.items():
            print(f"{ASSETS[k].name} episode-start forward returns:")
            print(r.returns["episode_summary"].round(3).to_string(), "\n")
            print(f"{ASSETS[k].name} backtest vs buy-and-hold:")
            print(r.performance.drop(columns=["start", "end"]).round(3).to_string(), "\n")
    print(f"Wrote {len(tables)} tables, {len(plots)} plots, {summary}, and current_regime.json/.md")


def cmd_snapshot(args: argparse.Namespace) -> None:
    frames = load_assets(_keys(args.assets), refresh=args.refresh, cache_dir=args.cache_dir)
    snap = build_snapshot(frames, REFERENCE_ASSET, args.bt_threshold)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_snapshot(snap, out)
    print(to_markdown(snap) if args.format == "md" else to_json(snap))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="onchain_regime")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--assets", default="ltc,btc", help="Comma-separated asset keys (doge once enabled)")
        sp.add_argument("--refresh", action="store_true", help="Ignore the raw CSV cache and re-download")
        sp.add_argument("--cache-dir", default="data/raw")
        sp.add_argument("--bt-threshold", type=float, default=10.0,
                        help="Point-in-time percentile threshold for the regime signal")

    run = sub.add_parser("run", help="Pull data, compute percentiles/episodes/returns/backtest, write outputs")
    common(run)
    run.add_argument("--rule", default="current", help="'current' or 'pctl:<n>' (point-in-time percentile)")
    run.add_argument("--gap-days", type=int, default=30, help="Merge matching days closer than this")
    run.add_argument("--bt-exit", default="hold:180", help="'hold:<days>' or 'mvrv_pct:<pct>'")
    run.add_argument("--out", default="outputs")
    run.set_defaults(func=cmd_run)

    snap = sub.add_parser("snapshot", help="Print the current-regime snapshot (daily data only)")
    common(snap)
    snap.add_argument("--format", choices=["json", "md"], default="json")
    snap.add_argument("--out", default=None, help="Also write current_regime.json/.md to this directory")
    snap.set_defaults(func=cmd_snapshot)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
