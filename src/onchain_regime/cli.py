"""Command line entry point: ``python -m onchain_regime run --assets ltc,btc``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from onchain_regime.assets import ASSETS  # noqa: E402
from onchain_regime.pipeline import Results, run_analysis  # noqa: E402
from onchain_regime.plots import all_plots  # noqa: E402

PCT_COLS_FMT = "{:.1f}"


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
    for name, fig in all_plots(res.frames, res.episodes, rel).items():
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
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> None:
    keys = [k.strip() for k in args.assets.split(",") if k.strip()]
    res = run_analysis(keys, rule=args.rule, refresh=args.refresh, cache_dir=args.cache_dir, gap_days=args.gap_days)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = write_tables(res, out)
    plots = write_plots(res, out)
    summary = out / "summary.md"
    summary.write_text(summary_markdown(res))

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
    print(f"Wrote {len(tables)} tables, {len(plots)} plots, and {summary}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="onchain_regime")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Pull data, compute percentiles/episodes/returns, write outputs")
    run.add_argument("--assets", default="ltc,btc", help="Comma-separated asset keys (doge once enabled)")
    run.add_argument("--rule", default="current", help="'current' or 'pctl:<n>' (point-in-time percentile)")
    run.add_argument("--refresh", action="store_true", help="Ignore the raw CSV cache and re-download")
    run.add_argument("--gap-days", type=int, default=30, help="Merge matching days closer than this")
    run.add_argument("--cache-dir", default="data/raw")
    run.add_argument("--out", default="outputs")
    run.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
