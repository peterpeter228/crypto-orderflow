import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import get_settings
from src.data.cache import MemoryCache
from src.mcp.tools import MCPTools
from src.utils import timestamp_ms


@pytest.mark.anyio
async def test_heatmap_disabled_includes_how_to_enable():
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
    assert out["dataQuality"]["degraded"] is True
    assert "howToEnable" in out


@pytest.mark.anyio
async def test_heatmap_enabled_uses_cached_metadata_and_warmup(monkeypatch):
    settings = get_settings()
    original_enabled = settings.heatmap_enabled
    original_bin_ticks = settings.heatmap_bin_ticks
    original_percent = settings.heatmap_depth_percent
    settings.heatmap_enabled = True
    settings.heatmap_bin_ticks = 5
    settings.heatmap_depth_percent = 1.5

    try:
        now = timestamp_ms()
        storage = MagicMock()
        storage.get_latest_orderbook_heatmap_snapshot = AsyncMock(
            return_value=(
                now,
                [
                    {"price_bin": 100.0, "bid_volume": 2.0, "ask_volume": 1.0},
                    {"price_bin": 101.0, "bid_volume": 1.0, "ask_volume": 3.0},
                ],
            )
        )
        storage.get_orderbook_heatmap_coverage = AsyncMock(
            return_value={"uniqueSnapshots": 4, "latestTimestamp": now}
        )

        book = MagicMock()
        book.mid_price = 101.25
        orderbook = MagicMock()
        orderbook.get_orderbook = MagicMock(return_value=book)

        cache = MemoryCache()
        cache.set_heatmap_metadata(
            "BTCUSDT",
            {
                "uniqueSnapshots": 4,
                "latestTimestamp": now,
                "lookbackMinutes": 10,
                "sampledAtMs": now,
            },
        )

        tools = MCPTools(
            cache=cache,
            storage=storage,
            orderbook=orderbook,
            rest_client=MagicMock(),
            vwap=MagicMock(),
            volume_profile=MagicMock(),
            tpo_profile=MagicMock(),
            session_levels=MagicMock(),
            footprint=MagicMock(),
            delta_cvd=MagicMock(),
            imbalance=MagicMock(),
            depth_delta=MagicMock(),
            heatmap=MagicMock(),
        )

        out = await tools.get_orderbook_heatmap("BTCUSDT", lookback_minutes=10, max_levels=1)
        assert out["enabled"] is True
        assert out["coverage"]["uniqueSnapshots"] == 4
        assert out["coverage"]["latestTimestamp"] == now
        assert out["warmup"]["expectedMinutes"] == 10
        assert out["topBidBins"][0]["bidVolume"] == 2.0
        assert out["topAskBins"][0]["askVolume"] == 3.0
    finally:
        settings.heatmap_enabled = original_enabled
        settings.heatmap_bin_ticks = original_bin_ticks
        settings.heatmap_depth_percent = original_percent
