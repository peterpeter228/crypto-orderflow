"""Tests for backfill fallbacks."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from src.binance.types import AggTrade
from src.config import get_settings
from src.data.backfill import AggTradesBackfiller
from src.data.storage import DataStorage
from src.utils.helpers import get_day_start_ms


class EmptyVision:
    """Vision stub that yields no trades."""

    async def iter_daily_aggtrades(self, *args, **kwargs):
        if False:
            yield None

    async def close(self):
        return None


def test_vision_empty_falls_back_to_rest(tmp_path, monkeypatch):
    """When Vision returns no rows, REST fallback should populate aggregates."""

    async def run():
        settings = get_settings()
        monkeypatch.setattr(settings, "cache_db_path", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "vision_cache_dir", str(tmp_path / "vision"))
        monkeypatch.setattr(settings, "backfill_source", "vision")

        storage = DataStorage()
        await storage.initialize()

        rest_client = AsyncMock()
        day_start = get_day_start_ms(1_600_000_000_000)
        trade_time = day_start + 1_000
        trades = [
            AggTrade(
                agg_trade_id=1,
                symbol="BTCUSDT",
                price=100.0,
                quantity=0.5,
                first_trade_id=1,
                last_trade_id=1,
                timestamp=trade_time,
                is_buyer_maker=False,
            ),
            AggTrade(
                agg_trade_id=2,
                symbol="BTCUSDT",
                price=100.5,
                quantity=0.25,
                first_trade_id=2,
                last_trade_id=2,
                timestamp=trade_time + 500,
                is_buyer_maker=True,
            ),
        ]
        rest_client.get_agg_trades = AsyncMock(return_value=trades)

        backfiller = AggTradesBackfiller(rest_client, storage)
        backfiller.vision = EmptyVision()

        end_time = day_start + 3_600_000  # one hour range
        try:
            await backfiller.backfill_symbol_range(
                symbol="BTCUSDT",
                start_time=day_start,
                end_time=end_time,
                pause_ms=0,
            )

            footprint_rows = await storage.get_footprint_range("BTCUSDT", day_start, end_time)
            daily_rows = await storage.get_daily_trades("BTCUSDT", day_start)

            assert rest_client.get_agg_trades.await_count > 0
            assert len(footprint_rows) > 0
            assert len(daily_rows) > 0
            assert sum(row["trade_count"] for row in footprint_rows) == len(trades)
            assert sum(row["trade_count"] for row in daily_rows) == len(trades)
        finally:
            await backfiller.close()
            await storage.close()

    asyncio.run(run())
