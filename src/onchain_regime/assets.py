"""Asset registry. Adding a chain means adding one entry to ``ASSETS``."""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical metric name -> API series id. All BRK-style deployments
# (Bitview, Litview) currently share these ids; override per asset if a
# deployment diverges.
DEFAULT_SERIES: dict[str, str] = {
    "price_close": "price_close",
    "realized_price": "realized_price",
    "mvrv": "mvrv",
    "supply_in_profit_share": "supply_in_profit_share",
    "sth_mvrv": "sth_mvrv",
    "lth_mvrv": "lth_mvrv",
}

# Metrics that must be present for a row to count as valid history.
CORE_METRICS: tuple[str, ...] = ("price_close", "mvrv")


@dataclass(frozen=True)
class AssetConfig:
    key: str
    name: str
    base_url: str | None
    mcp_url: str | None
    color: str
    series: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SERIES))
    enabled: bool = True


ASSETS: dict[str, AssetConfig] = {
    "ltc": AssetConfig(
        key="ltc",
        name="Litecoin",
        base_url="https://litview.space",
        mcp_url="https://mcp.litview.space/",
        color="#345d9d",
    ),
    "btc": AssetConfig(
        key="btc",
        name="Bitcoin",
        base_url="https://bitview.space",
        mcp_url="https://mcp.bitview.space/",
        color="#f7931a",
    ),
    # TODO(doge): dogeview.space currently returns Cloudflare error 1033.
    # Once the Dogeview API/MCP is live:
    #   1. set base_url="https://dogeview.space" and mcp_url to the MCP endpoint
    #   2. confirm series ids via GET /api/series/search?q=<metric> and override
    #      ``series`` here if any differ from DEFAULT_SERIES
    #   3. flip enabled=True and add "dogeview" to .cursor/mcp.json
    "doge": AssetConfig(
        key="doge",
        name="Dogecoin",
        base_url=None,
        mcp_url=None,
        color="#c2a633",
        enabled=False,
    ),
}


def get_asset(key: str) -> AssetConfig:
    try:
        return ASSETS[key]
    except KeyError:
        raise KeyError(f"Unknown asset {key!r}; known: {sorted(ASSETS)}") from None


def enabled_assets() -> list[AssetConfig]:
    return [a for a in ASSETS.values() if a.enabled]
