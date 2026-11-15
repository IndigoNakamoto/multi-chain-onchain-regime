"""REST client for BRK-style on-chain APIs (Bitview, Litview, ...)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from onchain_regime.assets import AssetConfig

# Cloudflare in front of these APIs returns 403 for the default python UA.
USER_AGENT = "onchain-regime/0.1 (+https://github.com/IndigoNakamoto)"
DEFAULT_CACHE_DIR = Path("data/raw")
PRICE_SERIES = {"price_close", "price_open", "price_high", "price_low", "realized_price"}


class BRKClient:
    """Fetches one series at a time; bulk full-history requests are rejected (400)."""

    def __init__(
        self,
        asset: AssetConfig,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        refresh: bool = False,
        timeout: float = 90.0,
        session: requests.Session | None = None,
    ) -> None:
        if not asset.base_url:
            raise NotImplementedError(
                f"{asset.name} has no base_url configured yet; see TODO({asset.key}) in assets.py"
            )
        self.asset = asset
        self.base_url = asset.base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.refresh = refresh
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self._dates: dict[str, pd.DatetimeIndex] = {}

    def _get(self, path: str, **params) -> object:
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _raw(self, series: str, index: str) -> list:
        return self._get(f"/api/series/{series}/{index}", start=0)["data"]

    def dates(self, index: str = "day1") -> pd.DatetimeIndex:
        if index not in self._dates:
            self._dates[index] = pd.DatetimeIndex(pd.to_datetime(self._raw("date", index)), name="date")
        return self._dates[index]

    def latest(self, series: str, index: str = "day1") -> float | None:
        return self._get(f"/api/series/{series}/{index}/latest")

    def search(self, query: str) -> list[str]:
        return self._get("/api/series/search", q=query)

    def _cache_path(self, series: str, index: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.asset.key}_{series}_{index}.csv"

    def fetch(self, series: str, index: str = "day1", drop_partial: bool = True) -> pd.Series:
        """Full history of ``series`` indexed by date.

        Zero prices (pre-launch placeholders) become NaN, and the final row is
        dropped when ``drop_partial`` because it is the current, incomplete period.
        """
        path = self._cache_path(series, index)
        if path is not None and path.exists() and not self.refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            s.index.name = "date"
        else:
            values = self._raw(series, index)
            dates = self.dates(index)
            if len(values) != len(dates):
                raise ValueError(
                    f"{self.asset.key}:{series}/{index} length {len(values)} != date length {len(dates)}"
                )
            s = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(), index=dates, name=series)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                s.to_csv(path)
        s = s.rename(series).astype(float)
        if series in PRICE_SERIES:
            s = s.mask(s <= 0)
        if drop_partial:
            s = s.iloc[:-1]
        return s
