"""Orderbook heatmap (liquidity surface) sampler.

This is a lightweight, data-only implementation inspired by the Rust project
`flowsurface`.

We periodically snapshot the in-memory orderbook, bin volumes by price, and
persist the result to SQLite. An MCP tool can then query the binned ladder for
visualization in an external charting tool.

Notes
-----
* This module intentionally stores *binned* levels instead of the full raw L2
  book to keep the database size manageable.
* The "bin size" is expressed in *price units* (e.g. USDT). For BTCUSDT, a
  bin size of 10 means: 87100, 87110, 87120 ...
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict

import structlog

from src.config import Settings
from src.data.orderbook import OrderbookManager
from src.data.storage import DataStorage
from src.utils.helpers import timestamp_ms


class OrderbookHeatmapSampler:
    """Periodically stores a binned depth ladder snapshot."""

    def __init__(
        self,
        storage: DataStorage,
        *,
        orderbook: OrderbookManager | None = None,
        orderbook_manager: OrderbookManager | None = None,
        settings: Settings,
        logger: structlog.BoundLogger,
    ):
        # Backwards compatibility: older code passed `orderbook_manager=`.
        if orderbook is None:
            orderbook = orderbook_manager
        if orderbook is None:
            raise TypeError(
                "OrderbookHeatmapSampler requires an `orderbook` (or legacy `orderbook_manager`) instance"
            )
        self.storage = storage
        self.orderbook = orderbook
        self.settings = settings
        self.logger = logger.bind(component="orderbook_heatmap")
        self._last_snapshot_ms: int = 0

    async def maybe_snapshot(self, symbol: str, now_ms: int | None = None) -> bool:
        """Take a snapshot if enough time has passed.

        Returns True if a snapshot was stored.
        """

        if not self.settings.heatmap_enabled:
            return False

        now_ms = int(now_ms or timestamp_ms())
        interval_ms = int(max(1, self.settings.heatmap_interval_sec) * 1000)
        if (now_ms - self._last_snapshot_ms) < interval_ms:
            return False

        book = self.orderbook.get_orderbook(symbol)
        if not book or not book.mid_price:
            return False

        mid = float(book.mid_price)
        pct = float(self.settings.heatmap_depth_percent)
        low = mid * (1.0 - pct / 100.0)
        high = mid * (1.0 + pct / 100.0)

        bin_size = float(self.settings.heatmap_bin_ticks)
        if bin_size <= 0:
            # Fallback: 1bp of price (coarse, but safe)
            bin_size = max(mid * 0.0001, 1e-8)

        bins: DefaultDict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])

        # Bids
        for price, qty in book.bids.items():
            if low <= price <= high:
                b = math.floor(price / bin_size) * bin_size
                bins[b][0] += float(qty)

        # Asks
        for price, qty in book.asks.items():
            if low <= price <= high:
                b = math.floor(price / bin_size) * bin_size
                bins[b][1] += float(qty)

        rows = [(float(p), float(v[0]), float(v[1])) for p, v in bins.items()]
        rows.sort(key=lambda x: x[0])

        try:
            await self.storage.save_orderbook_heatmap_snapshot(symbol, now_ms, rows)
            self._last_snapshot_ms = now_ms
            return True
        except Exception as e:
            self.logger.warning(
                "heatmap_snapshot_failed",
                symbol=symbol,
                error=str(e),
            )
            return False
