import pytest
import structlog
from unittest.mock import AsyncMock, MagicMock

from src.config import get_settings
from src.indicators.orderbook_heatmap import OrderbookHeatmapSampler


class _DummyLogger:
    def __init__(self):
        self.warning_calls = 0

    def bind(self, **kwargs):
        return self

    def warning(self, *args, **kwargs):
        self.warning_calls += 1


@pytest.mark.anyio
async def test_sampler_handles_dict_snapshot(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "heatmap_enabled", True)
    monkeypatch.setattr(settings, "heatmap_interval_sec", 0)
    monkeypatch.setattr(settings, "heatmap_depth_percent", 5)
    monkeypatch.setattr(settings, "heatmap_bin_ticks", 1)

    storage = MagicMock()
    storage.save_orderbook_heatmap_snapshot = AsyncMock()

    orderbook = MagicMock()
    orderbook.get_orderbook = MagicMock(
        return_value={
            "bids": [
                {"price": 99.5, "quantity": 2.0},
                {"price": 99.0, "quantity": 1.0},
            ],
            "asks": [
                {"price": 100.4, "quantity": 1.5},
                {"price": 100.6, "quantity": 1.0},
            ],
        }
    )

    sampler = OrderbookHeatmapSampler(
        storage=storage,
        orderbook=orderbook,
        settings=settings,
        logger=structlog.get_logger("test_heatmap_sampler"),
    )

    result = await sampler.maybe_snapshot("BTCUSDT", now_ms=10_000)
    assert result is True
    storage.save_orderbook_heatmap_snapshot.assert_awaited_once()

    _, _, rows = storage.save_orderbook_heatmap_snapshot.await_args.args
    # Ensure mid price was computed from best bid/ask and bins were built.
    assert len(rows) == 2
    assert rows[0][0] == 99.0
    assert rows[0][1] == 3.0  # bid volume aggregated into the 99 bin
    assert rows[1][0] == 100.0
    assert rows[1][2] == 2.5  # ask volume aggregated into the 100 bin


@pytest.mark.anyio
async def test_sampler_logs_missing_best_only_once(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "heatmap_enabled", True)
    monkeypatch.setattr(settings, "heatmap_interval_sec", 0)
    monkeypatch.setattr(settings, "heatmap_depth_percent", 5)
    monkeypatch.setattr(settings, "heatmap_bin_ticks", 1)

    storage = MagicMock()
    storage.save_orderbook_heatmap_snapshot = AsyncMock()

    orderbook = MagicMock()
    orderbook.get_orderbook = MagicMock(return_value={"bids": [], "asks": []})

    logger = _DummyLogger()
    sampler = OrderbookHeatmapSampler(
        storage=storage,
        orderbook=orderbook,
        settings=settings,
        logger=logger,
    )

    first = await sampler.maybe_snapshot("BTCUSDT", now_ms=1_000)
    second = await sampler.maybe_snapshot("BTCUSDT", now_ms=2_000)

    assert first is False
    assert second is False
    assert logger.warning_calls == 1
