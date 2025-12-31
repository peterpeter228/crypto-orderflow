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
async def test_get_session_profile_handles_missing_footprint():
    """get_session_profile should degrade gracefully when no footprint/profile data exists."""
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
    profile_engine = MagicMock()

    # No footprint rows, no klines
    storage.get_profile_range = AsyncMock(return_value=[])
    storage.get_footprint_coverage = AsyncMock(
        return_value={"minute_buckets": 0, "min_ts": None, "max_ts": None}
    )
    rest_client.get_klines = AsyncMock(return_value=[])
    profile_engine.build_profile = AsyncMock(
        return_value={
            "valueArea": {"vPOC": None, "VAH": None, "VAL": None},
            "totals": {"totalVolume": 0.0, "binCount": 0},
            "dataQuality": {"coveragePct": 0.0, "minuteBuckets": 0, "expectedMinutes": 0},
            "levels": [],
        }
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
        profile_engine=profile_engine,
    )

    result = await tools.get_session_profile(symbol="BTCUSDT", date="1970-01-02")

    assert "sessions" in result and result["sessions"]
    first_session = next(iter(result["sessions"].values()))
    profile = first_session["profile"]
    assert profile["available"] is False or "insufficient_data" in profile.get("qualityFlags", [])
    assert profile["vPOC"] is None
    assert profile["vVAH"] is None
    assert profile["vVAL"] is None
