"""Tests for MCPTools key levels date handling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.mcp.tools import MCPTools


def make_tools() -> MCPTools:
    """Create a MCPTools instance with mocked dependencies."""
    return MCPTools(
        cache=MagicMock(),
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
        heatmap=MagicMock(),
        profile_engine=MagicMock(),
    )


def test_is_current_day_none_true():
    tools = make_tools()
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    day_start_ms, is_current = tools.parse_date_or_today(None, now_ms=int(now.timestamp() * 1000))

    assert is_current is True
    assert day_start_ms == int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)


def test_is_current_day_today_true():
    tools = make_tools()
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    day_start_ms, is_current = tools.parse_date_or_today(date_str, now_ms=int(now.timestamp() * 1000))

    assert is_current is True
    assert day_start_ms == int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)


def test_is_current_day_yesterday_false():
    tools = make_tools()
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    yesterday = now - timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")
    day_start_ms, is_current = tools.parse_date_or_today(date_str, now_ms=int(now.timestamp() * 1000))

    assert is_current is False
    assert day_start_ms == int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.mark.anyio
async def test_get_key_levels_invalid_date_returns_error_json():
    tools = make_tools()
    result = await tools.get_key_levels(symbol="BTCUSDT", date="2024-13-01")

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_date"
    assert result["error"]["input"] == "2024-13-01"
