"""Lightweight sampler for heatmap metadata coverage.

This sampler is intentionally lightweight: it only queries aggregated metadata
(count + latest timestamp) to avoid loading full binned ladders into memory.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog

from src.config import Settings
from src.data.cache import MemoryCache
from src.data.storage import DataStorage
from src.utils.helpers import timestamp_ms


class HeatmapMetadataSampler:
    """Periodically samples heatmap coverage and caches the metadata in-memory."""

    def __init__(
        self,
        storage: DataStorage,
        cache: MemoryCache,
        *,
        settings: Settings,
        logger: structlog.BoundLogger,
    ):
        self.storage = storage
        self.cache = cache
        self.settings = settings
        self.logger = logger.bind(component="heatmap_metadata")
        self._last_sample_ms: dict[str, int] = defaultdict(int)

    async def maybe_sample(self, symbol: str, now_ms: int | None = None) -> dict[str, Any] | None:
        """Sample metadata if enabled and interval elapsed."""

        if not getattr(self.settings, "heatmap_enabled", False):
            return None

        now = int(now_ms or timestamp_ms())
        interval_ms = max(1, int(getattr(self.settings, "heatmap_sample_interval_ms", 15_000)))
        last_ts = self._last_sample_ms.get(symbol, 0)
        if (now - last_ts) < interval_ms:
            return None

        lookback_minutes = int(getattr(self.settings, "heatmap_lookback_minutes", 180))
        start_ms = now - (lookback_minutes * 60_000)

        try:
            cov = await self.storage.get_orderbook_heatmap_coverage(symbol, start_ms, now)
            meta = {
                "symbol": symbol,
                "lookbackMinutes": lookback_minutes,
                "uniqueSnapshots": cov.get("uniqueSnapshots", 0),
                "latestTimestamp": cov.get("latestTimestamp"),
                "sampledAtMs": now,
            }
            self.cache.set_heatmap_metadata(symbol, meta)
            self._last_sample_ms[symbol] = now
            return meta
        except Exception as e:
            self.logger.debug("heatmap_metadata_sample_failed", symbol=symbol, error=str(e))
            return None
