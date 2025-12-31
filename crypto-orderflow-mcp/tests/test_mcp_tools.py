"""Tests for MCPTools utilities."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.mcp.tools import MCPTools


@pytest.mark.anyio
async def test_get_orderflow_metrics_current_cvd_from_sequence():
    """currentCVD should reflect the last CVD value in the sequence."""
    cache = MagicMock()
    storage = MagicMock()
    orderbook = MagicMock()
    rest_client = MagicMock()
    vwap = MagicMock()
    volume_profile = MagicMock()
    tpo_profile = MagicMock()
    session_levels = MagicMock()
    footprint = MagicMock()
    delta_cvd = MagicMock()
    imbalance = MagicMock()
    depth_delta = MagicMock()

    cvd_sequence = [
        {"timestamp": 1, "cvd": 10.0},
        {"timestamp": 2, "cvd": 15.5},
        {"timestamp": 3, "cvd": 25.0},
    ]

    delta_cvd.get_delta_range = AsyncMock(
        return_value={
            "summary": {},
            "deltaSequence": [],
            "cvdSequence": cvd_sequence,
            "currentCVD": -999.0,  # Should be ignored in favor of last cvdSequence value
        }
    )
    footprint.get_footprint_range = AsyncMock(return_value=[])
    imbalance.analyze_footprint = MagicMock(return_value=None)
    storage.get_footprint_coverage = AsyncMock(
        return_value={"minute_buckets": 1, "min_ts": 0, "max_ts": 0}
    )

    tools = MCPTools(
        cache=cache,
        storage=storage,
        orderbook=orderbook,
        rest_client=rest_client,
        vwap=vwap,
        volume_profile=volume_profile,
        tpo_profile=tpo_profile,
        session_levels=session_levels,
        footprint=footprint,
        delta_cvd=delta_cvd,
        imbalance=imbalance,
        depth_delta=depth_delta,
    )

    result = await tools.get_orderflow_metrics(
        symbol="BTCUSDT",
        timeframe="1m",
        start_time=0,
        end_time=180000,
    )

    assert result["currentCVD"] == cvd_sequence[-1]["cvd"]


@pytest.mark.anyio
async def test_get_footprint_statistics_ok():
    """Statistics view should return dataQuality flags even when coverage is missing."""
    cache = MagicMock()
    storage = MagicMock()
    orderbook = MagicMock()
    rest_client = MagicMock()
    vwap = MagicMock()
    volume_profile = MagicMock()
    tpo_profile = MagicMock()
    session_levels = MagicMock()
    footprint = MagicMock()
    delta_cvd = MagicMock()
    imbalance = MagicMock()
    depth_delta = MagicMock()

    storage.get_footprint_statistics = AsyncMock(
        return_value=[
            {
                "bucket_start": 0,
                "vol_quote": 10.0,
                "delta_quote": 1.0,
                "delta_max_quote": 2.0,
                "delta_min_quote": -1.0,
                "buy_quote": 6.0,
                "sell_quote": 4.0,
                "trades": 5,
            }
        ]
    )
    storage.get_footprint_coverage = AsyncMock(return_value={"minute_buckets": 0, "min_ts": None, "max_ts": None})

    tools = MCPTools(
        cache=cache,
        storage=storage,
        orderbook=orderbook,
        rest_client=rest_client,
        vwap=vwap,
        volume_profile=volume_profile,
        tpo_profile=tpo_profile,
        session_levels=session_levels,
        footprint=footprint,
        delta_cvd=delta_cvd,
        imbalance=imbalance,
        depth_delta=depth_delta,
    )
    tools.on_demand_backfill_enabled = False

    result = await tools.get_footprint_statistics(
        symbol="BTCUSDT",
        start_time=0,
        end_time=60_000,
        timeframe="1m",
    )

    assert result["view"] == "statistics"
    assert result["timeframe"] == "1m"
    dq = result["dataQuality"]
    assert dq["degraded"] is True
    assert "low_coverage" in dq["qualityFlags"]


@pytest.mark.anyio
async def test_get_footprint_levels_ok():
    """Levels view should return bars and quality flags without raising."""
    cache = MagicMock()
    storage = MagicMock()
    orderbook = MagicMock()
    rest_client = MagicMock()
    vwap = MagicMock()
    volume_profile = MagicMock()
    tpo_profile = MagicMock()
    session_levels = MagicMock()
    delta_cvd = MagicMock()
    imbalance = MagicMock()
    depth_delta = MagicMock()

    footprint = MagicMock()
    footprint.get_footprint_range = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "timeframe": "1m",
                "timestamp": 0,
                "levels": [
                    {
                        "price": 100.0,
                        "buyVolume": 1.0,
                        "sellVolume": 0.0,
                        "tradeCount": 1,
                        "totalVolume": 1.0,
                    }
                ],
            }
        ]
    )
    storage.get_footprint_coverage = AsyncMock(return_value={"minute_buckets": 0, "min_ts": None, "max_ts": None})

    tools = MCPTools(
        cache=cache,
        storage=storage,
        orderbook=orderbook,
        rest_client=rest_client,
        vwap=vwap,
        volume_profile=volume_profile,
        tpo_profile=tpo_profile,
        session_levels=session_levels,
        footprint=footprint,
        delta_cvd=delta_cvd,
        imbalance=imbalance,
        depth_delta=depth_delta,
    )
    tools.on_demand_backfill_enabled = False

    result = await tools.get_footprint(
        symbol="BTCUSDT",
        start_time=0,
        end_time=60_000,
        timeframe="1m",
        view="levels",
        compress=False,
        limit=10,
    )

    assert result["view"] == "levels"
    assert result["timeframe"] == "1m"
    assert result["bars"] and result["bars"][0]["levels"][0]["price"] == 100.0
    dq = result["dataQuality"]
    assert dq["degraded"] is True
    assert "low_coverage" in dq["qualityFlags"]
