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
from dataclasses import dataclass
from typing import Any, DefaultDict, Mapping

import structlog

from src.config import Settings
from src.data.orderbook import OrderbookManager
from src.data.storage import DataStorage
from src.utils.helpers import timestamp_ms


@dataclass
class NormalizedOrderbookSnapshot:
    mid_price: float
    best_bid: float
    best_ask: float
    bids: dict[float, float]
    asks: dict[float, float]


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
        self._missing_best_logged = False

    def _normalize_snapshot(self, book: Any) -> NormalizedOrderbookSnapshot | None:
        """Normalize orderbook snapshots (dict or object) to a common shape."""
        if not book:
            return None

        mid_price: float | None = None
        best_bid: float | None = None
        best_ask: float | None = None
        bids_raw: Any = None
        asks_raw: Any = None

        if isinstance(book, Mapping):
            mid_price = book.get("mid_price") or book.get("midPrice")
            best_bid = book.get("best_bid") or book.get("bestBid")
            best_ask = book.get("best_ask") or book.get("bestAsk")
            bids_raw = book.get("bids")
            asks_raw = book.get("asks")
        else:
            mid_price = getattr(book, "mid_price", None)
            best_bid = getattr(book, "best_bid", None)
            best_ask = getattr(book, "best_ask", None)
            bids_raw = getattr(book, "bids", None)
            asks_raw = getattr(book, "asks", None)

        def _convert_side(side: Any) -> dict[float, float]:
            # Accept dict-like {price: qty} or list of {"price": x, "quantity": y}
            result: dict[float, float] = {}
            if isinstance(side, Mapping):
                for price, qty in side.items():
                    try:
                        result[float(price)] = float(qty)
                    except Exception:
                        continue
            elif isinstance(side, (list, tuple)):
                for row in side:
                    try:
                        price = float(row["price"]) if isinstance(row, Mapping) else float(row[0])
                        qty = float(row["quantity"]) if isinstance(row, Mapping) else float(row[1])
                        result[price] = qty
                    except Exception:
                        continue
            return result

        bids = _convert_side(bids_raw)
        asks = _convert_side(asks_raw)

        if best_bid is None and bids:
            best_bid = max(bids.keys())
        if best_ask is None and asks:
            best_ask = min(asks.keys())

        if mid_price is None and best_bid is not None and best_ask is not None:
            mid_price = (float(best_bid) + float(best_ask)) / 2.0

        if mid_price is None or best_bid is None or best_ask is None:
            if not self._missing_best_logged:
                self.logger.warning(
                    "heatmap_missing_best_levels",
                    has_mid=mid_price is not None,
                    has_best_bid=best_bid is not None,
                    has_best_ask=best_ask is not None,
                )
                self._missing_best_logged = True
            return None

        # Reset the one-shot warning once we eventually see valid data.
        self._missing_best_logged = False

        return NormalizedOrderbookSnapshot(
            mid_price=float(mid_price),
            best_bid=float(best_bid),
            best_ask=float(best_ask),
            bids=bids,
            asks=asks,
        )

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
        normalized = self._normalize_snapshot(book)
        if normalized is None:
            return False

        mid = normalized.mid_price
        pct = float(self.settings.heatmap_depth_percent)
        low = mid * (1.0 - pct / 100.0)
        high = mid * (1.0 + pct / 100.0)

        bin_size = float(self.settings.heatmap_bin_ticks)
        if bin_size <= 0:
            # Fallback: 1bp of price (coarse, but safe)
            bin_size = max(mid * 0.0001, 1e-8)

        bins: DefaultDict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])

        # Bids
        for price, qty in normalized.bids.items():
            if low <= price <= high:
                b = math.floor(price / bin_size) * bin_size
                bins[b][0] += float(qty)

        # Asks
        for price, qty in normalized.asks.items():
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
