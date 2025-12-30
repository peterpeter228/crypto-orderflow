import pytest
from unittest.mock import AsyncMock, MagicMock

from src.mcp.tools import MCPTools


@pytest.mark.anyio
async def test_footprint_statistics_degraded_when_low_coverage():
    storage = MagicMock()
    storage.get_footprint_statistics = AsyncMock(
        return_value=[
            {
                "bucket_start": 0,
                "vol_quote": 100.0,
                "delta_quote": -10.0,
                "delta_max_quote": 5.0,
                "delta_min_quote": -15.0,
                "buy_quote": 50.0,
                "sell_quote": 60.0,
                "trades": 12,
            }
        ]
    )
    storage.get_footprint_coverage = AsyncMock(
        return_value={"minute_buckets": 1, "min_ts": 0, "max_ts": 0}
    )

    tools = MCPTools(
        cache=MagicMock(),
        storage=storage,
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
    )
    tools.on_demand_backfill_enabled = False

    result = await tools.get_footprint_statistics(
        symbol="BTCUSDT",
        start_time=0,
        end_time=600_000,  # 10 minutes window
        timeframe="1m",
    )

    dq = result["dataQuality"]
    assert dq["degraded"] is True
    assert dq["coveragePct"] < tools.coverage_threshold
    assert dq["barsReturned"] == 1
