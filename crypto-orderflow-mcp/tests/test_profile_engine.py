import pytest
from unittest.mock import AsyncMock, MagicMock

from src.data.cache import MemoryCache
from src.mcp.tools import MCPTools
from src.indicators.profile_engine import VolumeProfileEngine


class DummyStorage:
    async def get_footprint_coverage(self, symbol, start_time, end_time):
        return {"minute_buckets": 3, "min_ts": start_time, "max_ts": end_time - 60_000}

    async def get_profile_range(self, symbol, start_time, end_time):
        return [
            {"price_level": 100.0, "buy_volume": 1.0, "sell_volume": 1.0, "trade_count": 2},
            {"price_level": 110.0, "buy_volume": 2.0, "sell_volume": 2.0, "trade_count": 3},
            {"price_level": 120.0, "buy_volume": 1.0, "sell_volume": 0.0, "trade_count": 1},
        ]


@pytest.mark.anyio
async def test_volume_profile_engine_value_area():
    storage = DummyStorage()
    cache = MemoryCache()
    engine = VolumeProfileEngine(storage, cache, coverage_threshold=0.2)

    res = await engine.build_profile(
        "BTCUSDT",
        window_start_ms=0,
        window_end_ms=180_000,
        bin_size=10.0,
        value_area_pct=70.0,
        include_levels=True,
        top_n_levels=2,
    )

    assert res["valueArea"]["vPOC"] == 110.0
    assert res["valueArea"]["VAL"] == 100.0
    assert res["levelsTruncated"] is True
    assert res["dataQuality"]["coveragePct"] == 1.0


class DummyFootprint:
    async def get_footprint_range(self, symbol, start_time, end_time, timeframe="1m"):
        return [
            {
                "timestamp": start_time,
                "timeframe": timeframe,
                "levels": [
                    {"price": 100.0, "buyVolume": 2.0, "sellVolume": 0.5, "tradeCount": 2},
                    {"price": 101.0, "buyVolume": 1.0, "sellVolume": 0.2, "tradeCount": 1},
                ],
            },
            {
                "timestamp": end_time - 60_000,
                "timeframe": timeframe,
                "levels": [
                    {"price": 102.0, "buyVolume": 3.0, "sellVolume": 1.0, "tradeCount": 2},
                    {"price": 103.0, "buyVolume": 0.5, "sellVolume": 0.1, "tradeCount": 1},
                ],
            },
        ]


@pytest.mark.anyio
async def test_get_footprint_levels_compression_and_pagination():
    storage = MagicMock()
    storage.get_footprint_coverage = AsyncMock(return_value={"minute_buckets": 10, "min_ts": 0, "max_ts": 120_000})
    tools = MCPTools(
        cache=MemoryCache(),
        storage=storage,
        orderbook=MagicMock(),
        rest_client=MagicMock(),
        vwap=MagicMock(),
        volume_profile=MagicMock(),
        tpo_profile=MagicMock(),
        session_levels=MagicMock(),
        footprint=DummyFootprint(),
        delta_cvd=MagicMock(),
        imbalance=MagicMock(),
        depth_delta=MagicMock(),
        heatmap=None,
    )

    res = await tools.get_footprint(
        "BTCUSDT",
        start_time=0,
        end_time=180_000,
        timeframe="1m",
        view="levels",
        bin_size=1.0,
        top_n_levels=1,
        limit=1,
    )

    assert res["bars"][0]["levelsDropped"] >= 1
    assert res["nextCursor"] == 1


@pytest.mark.anyio
async def test_heatmap_disabled_remediation():
    tools = MCPTools(
        cache=MemoryCache(),
        storage=MagicMock(),
        orderbook=MagicMock(),
        rest_client=MagicMock(),
        vwap=MagicMock(),
        volume_profile=MagicMock(),
        tpo_profile=MagicMock(),
        session_levels=MagicMock(),
        footprint=MagicMock(),
        delta_cvd=MagicMock(),
        imbalance=MagicMock(),
        depth_delta=MagicMock(),
        heatmap=None,
    )

    out = await tools.get_orderbook_heatmap("BTCUSDT")
    assert out["enabled"] is False
    assert "remediation" in out


class KeywordOnlyFootprint:
    def __init__(self) -> None:
        self.called_with = None

    async def get_footprint_range(self, *, symbol: str, timeframe: str, start_time: int, end_time: int):
        self.called_with = (symbol, timeframe, start_time, end_time)
        return []


class KeywordOnlyBackfill:
    def __init__(self) -> None:
        self.called = False
        self.args = None

    async def __call__(self, symbol: str, start_ms: int, end_ms: int, *, coverage_pct: float):
        self.called = True
        self.args = (symbol, start_ms, end_ms, coverage_pct)


@pytest.mark.anyio
async def test_get_footprint_levels_respects_keyword_params():
    storage = MagicMock()
    storage.get_footprint_coverage = AsyncMock(return_value={"minute_buckets": 10, "min_ts": 0, "max_ts": 120_000})
    footprint = KeywordOnlyFootprint()
    tools = MCPTools(
        cache=MemoryCache(),
        storage=storage,
        orderbook=MagicMock(),
        rest_client=MagicMock(),
        vwap=MagicMock(),
        volume_profile=MagicMock(),
        tpo_profile=MagicMock(),
        session_levels=MagicMock(),
        footprint=footprint,
        delta_cvd=MagicMock(),
        imbalance=MagicMock(),
        depth_delta=MagicMock(),
        heatmap=None,
    )

    await tools.get_footprint(
        "BTCUSDT",
        start_time=0,
        end_time=120_000,
        timeframe="5m",
        view="levels",
    )

    assert footprint.called_with == ("BTCUSDT", "5m", 0, 120_000)


@pytest.mark.anyio
async def test_profile_engine_backfill_callback_uses_keyword_coverage():
    cache = MemoryCache()
    storage = MagicMock()
    backfill_cb = KeywordOnlyBackfill()
    engine = VolumeProfileEngine(storage, cache, backfill_callback=backfill_cb, coverage_threshold=0.9)

    await engine._maybe_backfill("BTCUSDT", 0, 60_000, 0.5)

    assert backfill_cb.called is True
    assert backfill_cb.args == ("BTCUSDT", 0, 60_000, 0.5)
