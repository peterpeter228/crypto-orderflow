import pytest
from unittest.mock import AsyncMock, MagicMock

from src.binance.types import AggTrade
from src.config import get_settings
from src.data.backfill import AggTradesBackfiller
from src.data.storage import DataStorage
from src.utils.helpers import get_day_start_ms


@pytest.mark.asyncio
async def test_backfill_retries_empty_day(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "cache_db_path", str(tmp_path / "retry.db"))
    monkeypatch.setattr(settings, "backfill_source", "rest")
    monkeypatch.setattr(settings, "backfill_empty_day_retries", 1)
    monkeypatch.setattr(settings, "backfill_empty_day_retry_delay_ms", 0)

    storage = DataStorage()
    await storage.initialize()

    rest_client = MagicMock()
    trade_ts = 1_000
    agg_trade = AggTrade(
        agg_trade_id=1,
        symbol="BTCUSDT",
        price=100.0,
        quantity=1.0,
        first_trade_id=1,
        last_trade_id=1,
        timestamp=trade_ts,
        is_buyer_maker=False,
    )
    rest_client.get_agg_trades = AsyncMock(side_effect=[[], [agg_trade]])

    backfiller = AggTradesBackfiller(rest_client, storage)

    day_start = get_day_start_ms(trade_ts)
    await backfiller.backfill_symbol_range(
        symbol="BTCUSDT",
        start_time=day_start,
        end_time=day_start + 60_000,
        pause_ms=0,
        clear_days=True,
    )

    counts = await storage.get_day_row_counts("BTCUSDT", day_start)
    assert counts["footprint_1m"] > 0
    assert rest_client.get_agg_trades.await_count == 2

    await backfiller.close()
    await storage.close()
