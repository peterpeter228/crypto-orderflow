"""Tests for get_tpo_profile behavior when tick size config is missing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.mcp.tools import MCPTools


def test_get_tpo_profile_uses_default_tick_size_when_missing(monkeypatch):
    """Ensure get_tpo_profile falls back to symbol default and logs a warning."""

    cache = MagicMock()
    storage = MagicMock()
    orderbook = MagicMock()
    rest_client = MagicMock()
    rest_client.get_klines = AsyncMock(return_value=[])
    vwap = MagicMock()
    volume_profile = MagicMock()
    tpo_profile = MagicMock()
    tpo_profile.build_profile = AsyncMock(return_value={"totals": {"tpoTotalCount": 1}})
    session_levels = MagicMock()
    footprint = MagicMock()
    delta_cvd = MagicMock()
    imbalance = MagicMock()
    depth_delta = MagicMock()

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

    default_tick = tools.settings.get_default_tpo_tick_size("BTCUSDT")
    monkeypatch.setattr(
        tools.settings.__class__,
        "get_tpo_tick_size",
        MagicMock(side_effect=AttributeError("missing")),
    )

    tools.logger = MagicMock()

    result = asyncio.run(tools.get_tpo_profile("BTCUSDT"))

    assert tpo_profile.build_profile.await_args.kwargs["tick_size"] == default_tick
    assert any(
        "tick size not configured" in w.lower() for w in result.get("warnings", [])
    )
    tools.logger.warning.assert_called_once()
