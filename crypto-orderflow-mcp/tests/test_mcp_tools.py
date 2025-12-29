"""Tests for MCP tools safety around TPO tick size defaults."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import Settings
from src.mcp.tools import MCPTools


def test_get_tpo_profile_handles_missing_tick_sizes():
    """Ensure TPO profile works when tick size fields are absent in settings."""

    # Create settings without explicit TPO tick size fields to mimic older envs
    settings = Settings(_env_file=None)
    if hasattr(settings, "tpo_tick_size_btc"):
        delattr(settings, "tpo_tick_size_btc")
    if hasattr(settings, "tpo_tick_size_eth"):
        delattr(settings, "tpo_tick_size_eth")

    # Minimal mocks for dependencies
    cache = MagicMock()
    storage = MagicMock()
    orderbook = MagicMock()
    rest_client = MagicMock()
    rest_client.get_klines = AsyncMock(return_value=[])
    vwap = MagicMock()
    volume_profile = MagicMock()
    tpo_profile = MagicMock()
    tpo_profile.build_profile = AsyncMock(return_value={})
    session_levels = MagicMock()
    footprint = MagicMock()
    delta_cvd = MagicMock()
    imbalance = MagicMock()
    depth_delta = MagicMock()

    tools = MCPTools(
        cache,
        storage,
        orderbook,
        rest_client,
        vwap,
        volume_profile,
        tpo_profile,
        session_levels,
        footprint,
        delta_cvd,
        imbalance,
        depth_delta,
    )

    # Replace settings with the one missing tick size attributes
    tools.settings = settings

    import asyncio

    result = asyncio.run(tools.get_tpo_profile(symbol="BTCUSDT", tick_size=None))

    assert result is not None
    # build_profile should have been invoked with the fallback tick size for BTC
    assert tpo_profile.build_profile.await_args.kwargs["tick_size"] == pytest.approx(70.0)
