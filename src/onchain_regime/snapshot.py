"""Compact current-regime snapshot (JSON + markdown) for dashboards and bots."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from onchain_regime.assets import get_asset
from onchain_regime.backtest import regime_signal
from onchain_regime.compare import relative_mvrv, relative_summary
from onchain_regime.stats import VALUATION_METRICS, current_percentiles, joint_rarity


def classify(mvrv_pct: float, sip_pct: float) -> str:
    """Regime label from full-history percentiles of MVRV and supply in profit."""
    if mvrv_pct <= 10 and sip_pct <= 10:
        return "capitulation"
    if mvrv_pct <= 25 and sip_pct <= 25:
        return "undervalued"
    if mvrv_pct >= 90 or sip_pct >= 95:
        return "overheated"
    return "neutral"


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (dt.date, pd.Timestamp)):
        return x.isoformat()[:10]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 6)
    return x


def build_snapshot(frames: dict[str, pd.DataFrame], ref: str = "btc", threshold: float = 10.0) -> dict:
    assets = {}
    for key, df in frames.items():
        cur = current_percentiles(df, VALUATION_METRICS)
        joint = joint_rarity(df)
        assets[key] = {
            "name": get_asset(key).name,
            "date": df.index[-1],
            "price_close": df["price_close"].iloc[-1],
            "realized_price": df["realized_price"].iloc[-1],
            "metrics": {m: {"value": cur.at[m, "value"], "percentile": cur.at[m, "percentile"]} for m in cur.index},
            "joint_percentile": joint["joint_percentile"],
            "bottom_pct_signal": {"threshold": threshold, "active": bool(regime_signal(df, threshold).iloc[-1])},
            "regime": classify(cur.at["mvrv", "percentile"], cur.at["supply_in_profit_share", "percentile"]),
        }
    relative = {}
    if ref in frames:
        for key in frames:
            if key != ref:
                relative[f"{key}/{ref}"] = {k: v for k, v in relative_summary(relative_mvrv(frames, key, ref)).items() if k != "pair"}
    snap = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": max(df.index[-1] for df in frames.values()),
        "percentiles": "full history, at or below; joint = share of days with MVRV and supply in profit both at or below today",
        "assets": assets,
        "relative": relative,
    }
    return _clean(snap)


CHANGE_PCT_POINTS = 10.0
TRACKED_METRICS = ("mvrv", "supply_in_profit_share")


def compact(snap: dict) -> dict:
    """The fields change detection compares; one line per day in history.jsonl."""
    return {
        "as_of": snap["as_of"],
        "assets": {
            k: {
                "regime": a["regime"],
                "signal": a["bottom_pct_signal"]["active"],
                **{f"{m}_pct": a["metrics"][m]["percentile"] for m in TRACKED_METRICS},
            }
            for k, a in snap["assets"].items()
        },
        "relative": {pair: r["percentile"] for pair, r in snap["relative"].items()},
    }


def diff(prev: dict, cur: dict, threshold: float = CHANGE_PCT_POINTS) -> list[str]:
    """Human-readable changes between two compact snapshots."""
    items = []
    for k, c in cur["assets"].items():
        p = prev["assets"].get(k)
        name = k.upper()
        if p is None:
            items.append(f"{name} added to the monitor")
            continue
        if p["regime"] != c["regime"]:
            items.append(f"{name} regime changed: {p['regime']} -> {c['regime']}")
        if p["signal"] != c["signal"]:
            items.append(f"{name} bottom-percentile signal turned {'ON' if c['signal'] else 'OFF'}")
        for m in TRACKED_METRICS:
            a, b = p.get(f"{m}_pct"), c.get(f"{m}_pct")
            if a is not None and b is not None and abs(b - a) >= threshold:
                items.append(f"{name} {m} percentile moved {a:.1f} -> {b:.1f} ({b - a:+.1f})")
    for pair, b in cur["relative"].items():
        a = prev["relative"].get(pair)
        if a is not None and b is not None and abs(b - a) >= threshold:
            items.append(f"{pair.upper()} relative MVRV percentile moved {a:.1f} -> {b:.1f} ({b - a:+.1f})")
    return items


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def attach_changes(snap: dict, history: list[dict], lookback_days: int = 7) -> dict:
    """Add ``changes`` vs the previous recorded day and vs ``lookback_days`` ago."""
    cur = compact(snap)
    as_of = dt.date.fromisoformat(cur["as_of"])
    earlier = [h for h in history if dt.date.fromisoformat(h["as_of"]) < as_of]
    cutoff = as_of - dt.timedelta(days=lookback_days)
    week = [h for h in earlier if dt.date.fromisoformat(h["as_of"]) <= cutoff]
    snap["changes"] = {
        "since_previous": {"as_of": earlier[-1]["as_of"], "items": diff(earlier[-1], cur)} if earlier else None,
        f"since_{lookback_days}d": {"as_of": week[-1]["as_of"], "items": diff(week[-1], cur)} if week else None,
    }
    return snap


def update_history(path: Path, snap: dict) -> None:
    """Append today's compact record, replacing any earlier record for the same day."""
    cur = compact(snap)
    history = [h for h in load_history(path) if h["as_of"] != cur["as_of"]]
    history.append(cur)
    history.sort(key=lambda h: h["as_of"])
    path.write_text("".join(json.dumps(h) + "\n" for h in history))


def to_json(snap: dict) -> str:
    return json.dumps(snap, indent=2)


def _changes_md(snap: dict) -> list[str]:
    changes = snap.get("changes")
    if not changes:
        return []
    lines = ["", "**Changes**"]
    for key, label in (("since_previous", "Since previous snapshot"), ("since_7d", "Since 7 days ago")):
        c = changes.get(key)
        if c is None:
            lines.append(f"- {label}: no earlier snapshot recorded")
        elif not c["items"]:
            lines.append(f"- {label} ({c['as_of']}): none")
        else:
            lines.append(f"- {label} ({c['as_of']}):")
            lines += [f"  - {item}" for item in c["items"]]
    return lines


def to_markdown(snap: dict) -> str:
    signals = ", ".join(
        f"{a['name']} {'ON' if a['bottom_pct_signal']['active'] else 'off'}" for a in snap["assets"].values()
    )
    threshold = next(iter(snap["assets"].values()))["bottom_pct_signal"]["threshold"]
    lines = [
        f"**On-chain regime, {snap['as_of']}** (generated {snap['generated_at']})",
        "",
        "| Asset | Regime | Price | Realized | MVRV (pct) | Supply in profit (pct) | Joint rarity |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in snap["assets"].values():
        m = a["metrics"]
        lines.append(
            f"| {a['name']} | {a['regime']} | ${a['price_close']:,.2f} | ${a['realized_price']:,.2f} "
            f"| {m['mvrv']['value']:.3f} ({m['mvrv']['percentile']:.1f}) "
            f"| {m['supply_in_profit_share']['value']:.1f}% ({m['supply_in_profit_share']['percentile']:.1f}) "
            f"| {a['joint_percentile']:.2f}% |"
        )
    for pair, r in snap["relative"].items():
        lines += ["", f"{pair.upper()} relative MVRV {r['relative_mvrv']:.3f} (pct {r['percentile']:.1f}); "
                      f"supply-in-profit spread {r['sip_spread']:+.1f} pp (pct {r['sip_spread_pct']:.1f})."]
    lines += ["", f"Bottom-{threshold:g}% joint signal (MVRV and supply in profit, point-in-time): {signals}."]
    lines += _changes_md(snap)
    lines += ["", "Percentiles are full-history (share of days at or below today). Valuation context, not a trade signal."]
    return "\n".join(lines) + "\n"
