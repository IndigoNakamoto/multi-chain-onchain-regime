import json

import numpy as np
import pandas as pd
import pytest

from onchain_regime.assets import ASSETS
from onchain_regime.backtest import backtest, performance, positions, trades
from onchain_regime.snapshot import build_snapshot, classify, to_markdown
from onchain_regime.client import BRKClient
from onchain_regime.compare import regime_timing
from onchain_regime.data import load_asset, trim_to_valid
from onchain_regime.episodes import Rule, cluster_episodes, find_episodes
from onchain_regime.returns import episode_forward_returns, forward_returns, summarize
from onchain_regime.stats import expanding_percentile, joint_rarity, percentile_of


def frame(mvrv, sip, price=None, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(mvrv), freq="D", name="date")
    price = price if price is not None else np.arange(1, len(mvrv) + 1, dtype=float)
    return pd.DataFrame({"mvrv": mvrv, "supply_in_profit_share": sip, "price_close": price}, index=idx)


def test_percentile_of_counts_ties_as_at_or_below():
    s = pd.Series([1, 2, 2, 3, np.nan])
    assert percentile_of(s, 2) == 75.0
    assert percentile_of(s, 0) == 0.0


def test_expanding_percentile_has_no_lookahead():
    s = pd.Series([5.0, 4.0, 3.0, 10.0])
    pit = expanding_percentile(s, min_periods=1)
    assert pit.tolist() == [100.0, 50.0, pytest.approx(100 / 3), 100.0]


def test_joint_rarity():
    df = frame([1.0, 0.5, 0.4, 2.0, 0.5], [50, 10, 20, 90, 15])
    r = joint_rarity(df)
    # days with mvrv<=0.5 and sip<=15: index 1 (0.5,10) and 4 (0.5,15)
    assert r["matching_days"] == 2 and r["total_days"] == 5


def test_cluster_episodes_merges_within_gap():
    df = frame([1.0] * 100, [50] * 100)
    mask = pd.Series(False, index=df.index)
    mask.iloc[[5, 10, 50, 51, 99]] = True
    eps = cluster_episodes(df, mask, gap_days=30)
    assert len(eps) == 3
    assert eps.iloc[0]["start"] == df.index[5] and eps.iloc[0]["end"] == df.index[10]
    assert eps.iloc[0]["matched_days"] == 2 and eps.iloc[0]["span_days"] == 6
    assert eps.iloc[-1]["is_current"] and not eps.iloc[0]["is_current"]


def test_current_rule_matches_today():
    df = frame([2.0, 0.5, 1.0, 0.6], [80, 10, 50, 20])
    eps, mask = find_episodes(df, "current")
    assert mask.tolist() == [False, True, False, True]
    assert Rule.parse("pctl:5").threshold == 5.0


def test_forward_returns_use_calendar_days_and_flag_incomplete():
    price = pd.Series([100.0, 110.0, 121.0, 133.1], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    fwd = forward_returns(price, horizons=(1, 2))
    assert fwd["fwd_1d"].iloc[0] == pytest.approx(0.10)
    assert fwd["fwd_2d"].iloc[0] == pytest.approx(0.21)
    assert fwd["fwd_2d"].iloc[-2:].isna().all()
    # a missing row must not shift the horizon
    gappy = price.drop(price.index[1])
    assert forward_returns(gappy, horizons=(1,))["fwd_1d"].iloc[0] != pytest.approx(0.21)


def test_episode_forward_returns_from_start_and_summary():
    n = 60
    df = frame([1.0] * n, [50] * n, price=np.linspace(100, 159, n))
    eps = pd.DataFrame(
        {"start": [df.index[0], df.index[50]], "end": [df.index[2], df.index[55]],
         "matched_days": [3, 6], "start_price": [100.0, 150.0], "is_current": [False, True]}
    )
    ep = episode_forward_returns(df, eps, horizons=(10,))
    assert ep["fwd_10d"].iloc[0] == pytest.approx(110 / 100 - 1)
    assert ep["incomplete"].tolist() == [False, True]
    s = summarize(ep, horizons=(10,))
    assert s.loc[10, "n"] == 1 and s.loc[10, "n_incomplete"] == 1 and s.loc[10, "hit_rate"] == 1.0


def test_trim_to_valid_drops_missing_and_stale_prefix():
    price = [np.nan, 3.15] + [3.15] * 20 + [3.0, 3.2, 3.1]
    mvrv = [np.nan] + [1.0] * 24
    df = frame(mvrv, [50] * 25, price=price)
    out = trim_to_valid(df)
    assert out.index[0] == df.index[22]


def test_regime_timing_prefers_overlapping_episode():
    d = pd.Timestamp
    btc = pd.DataFrame({"start": [d("2014-09-19"), d("2015-08-18")], "end": [d("2015-06-14"), d("2015-09-27")]})
    ltc = pd.DataFrame({"start": [d("2015-03-21")], "end": [d("2015-05-06")]})
    t = regime_timing({"btc": btc, "ltc": ltc}, "btc")
    assert t.loc[0, "btc_start"] == d("2014-09-19").date() and t.loc[0, "overlap_days"] == 47


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, data):
        self.headers = {}
        self.data = data
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        series = url.split("/api/series/")[1].split("/")[0]
        return FakeResponse({"data": self.data[series]})


def test_client_masks_zero_prices_drops_partial_day_and_sets_ua():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    sess = FakeSession({"date": dates, "price_close": [0.0, 10.0, 11.0, 12.5], "mvrv": [None, 1.0, 1.1, 1.2]})
    c = BRKClient(ASSETS["btc"], cache_dir=None, session=sess)
    p = c.fetch("price_close")
    assert len(p) == 3 and np.isnan(p.iloc[0]) and p.iloc[-1] == 11.0
    assert p.index[-1] == pd.Timestamp("2020-01-03")
    assert "python" not in sess.headers["User-Agent"].lower()
    c.fetch("mvrv")
    assert sum("/date/" in u for u in sess.calls) == 1


def test_load_asset_with_fake_client_derives_price_to_realized():
    dates = [f"2020-01-{d:02d}" for d in range(1, 6)]
    data = {"date": dates, "price_close": [0, 10, 12, 9, 9.5], "realized_price": [0, 10, 10, 10, 10],
            "mvrv": [None, 1.0, 1.2, 0.9, 0.95], "supply_in_profit_share": [None, 100, 100, 0, 50],
            "sth_mvrv": [None, 1, 1, 1, 1], "lth_mvrv": [None, 1, 1, 1, 1]}
    c = BRKClient(ASSETS["ltc"], cache_dir=None, session=FakeSession(data))
    df = load_asset("ltc", client=c)
    assert df.index[0] == pd.Timestamp("2020-01-02") and len(df) == 3
    assert df["price_to_realized"].tolist() == pytest.approx([1.0, 1.2, 0.9])


def test_positions_hold_acts_next_day_and_expires():
    df = frame([1.0] * 10, [50] * 10)
    sig = pd.Series(False, index=df.index)
    sig.iloc[2] = True
    pos = positions(df, sig, "hold:3")
    assert pos.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_positions_mvrv_exit():
    mvrv = [1.0] * 5 + [0.1] + [0.5, 0.6, 50.0, 0.7, 0.8]
    df = frame(mvrv, [50] * 11)
    sig = pd.Series(False, index=df.index)
    sig.iloc[5] = True
    pos = positions(df, sig, "mvrv_pct:90", min_history=1)
    # enter day 6, MVRV hits 100th pct on day 8, flat from day 9
    assert pos.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]


def test_backtest_equity_and_performance():
    n = 400
    price = np.concatenate([np.full(200, 100.0), np.linspace(100, 50, 100), np.linspace(50, 200, 100)])
    mvrv = np.concatenate([np.full(200, 1.0), np.linspace(1.0, 0.3, 100), np.linspace(0.3, 2.0, 100)])
    sip = np.concatenate([np.full(200, 60.0), np.linspace(60, 5, 100), np.linspace(5, 99, 100)])
    df = frame(mvrv, sip, price=price)
    bt = backtest(df, threshold=10, exit="hold:30", min_history=50)
    assert bt.index[0] == df.index[49]
    assert bt["buy_hold_equity"].iloc[-1] == pytest.approx(2.0)
    flat = bt["position"] == 0
    assert (bt.loc[flat, "strategy_ret"] == 0).all()
    perf = performance(bt)
    assert perf.loc["buy_hold", "time_in_market"] == 1.0
    assert 0 < perf.loc["strategy", "time_in_market"] < 1
    assert perf.loc["strategy", "max_drawdown"] > perf.loc["buy_hold", "max_drawdown"]
    t = trades(bt)
    assert (1 + t["return"]).prod() == pytest.approx(bt["strategy_equity"].iloc[-1])


def test_snapshot_classify_and_json_safe():
    assert classify(5, 5) == "capitulation"
    assert classify(20, 3) == "undervalued"
    assert classify(95, 50) == "overheated"
    assert classify(50, 50) == "neutral"
    n = 500
    rng = np.random.default_rng(0)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    base = frame(rng.uniform(0.5, 2, n), rng.uniform(10, 90, n), price=price)
    frames = {}
    for k in ("ltc", "btc"):
        df = base.copy()
        df["realized_price"] = df["price_close"] / df["mvrv"]
        df["sth_mvrv"] = df["lth_mvrv"] = df["mvrv"]
        df["price_to_realized"] = df["mvrv"]
        frames[k] = df
    snap = build_snapshot(frames)
    json.dumps(snap)
    assert set(snap["assets"]) == {"ltc", "btc"} and "ltc/btc" in snap["relative"]
    assert snap["relative"]["ltc/btc"]["relative_mvrv"] == pytest.approx(1.0)
    assert "| Litecoin |" in to_markdown(snap)


def test_doge_stub_raises_clear_error():
    with pytest.raises(NotImplementedError, match="TODO\\(doge\\)"):
        BRKClient(ASSETS["doge"])
