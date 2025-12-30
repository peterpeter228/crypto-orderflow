"""Unified profile engines for Volume Profile and shared value-area logic.

The goal is to keep all profile-based outputs (Session/VR/TPO key levels)
aligned with Exocharts semantics:
- Explicit window (start/end in ms)
- Explicit bin_size (price bucket, e.g., T:70)
- Value area expansion from vPOC outward until value_area_pct of volume is covered
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from src.data.cache import MemoryCache
from src.data.storage import DataStorage
from src.utils import get_logger, timestamp_ms


def _as_fraction(value_area_pct: float) -> float:
    """Normalize value_area_pct to a 0-1 fraction."""
    if value_area_pct is None:
        return 0.7
    pct = float(value_area_pct)
    return pct / 100.0 if pct > 1 else pct


class ValueAreaCalculator:
    """Deterministic value area expansion used across VP/TPO."""

    @staticmethod
    def _poc(levels: dict[float, float]) -> float | None:
        if not levels:
            return None
        max_vol = max(levels.values())
        candidates = sorted(p for p, v in levels.items() if v == max_vol)
        return candidates[len(candidates) // 2]

    @classmethod
    def compute(cls, levels: dict[float, float], value_area_pct: float) -> tuple[float | None, float | None, float | None]:
        """Return (poc, vah, val) using outward expansion from vPOC."""
        if not levels:
            return None, None, None

        fraction = _as_fraction(value_area_pct)
        total = float(sum(levels.values()))
        if total <= 0:
            return None, None, None

        prices = sorted(levels.keys())
        poc = cls._poc(levels)
        if poc is None:
            return None, None, None
        if poc not in prices:
            poc = min(prices, key=lambda p: abs(p - poc))
        poc_idx = prices.index(poc)
        vah_idx = val_idx = poc_idx
        current = float(levels[poc])
        target = total * fraction

        while current < target and (vah_idx + 1 < len(prices) or val_idx - 1 >= 0):
            up_idx = vah_idx + 1
            down_idx = val_idx - 1

            up_vol = float(levels.get(prices[up_idx], 0.0)) if up_idx < len(prices) else -1.0
            down_vol = float(levels.get(prices[down_idx], 0.0)) if down_idx >= 0 else -1.0

            if up_vol <= 0 and down_vol <= 0:
                break

            if up_vol > down_vol:
                vah_idx = up_idx
                current += up_vol
            elif down_vol > up_vol:
                val_idx = down_idx
                current += down_vol
            else:
                if up_idx < len(prices):
                    vah_idx = up_idx
                    current += up_vol
                if down_idx >= 0 and down_idx != up_idx:
                    val_idx = down_idx
                    current += down_vol

        return poc, prices[vah_idx], prices[val_idx]


@dataclass
class ProfileWindow:
    start_ms: int
    end_ms: int
    bin_size: float
    value_area_pct: float
    mode: str
    normalize_seconds: int


class VolumeProfileEngine:
    """Builds deterministic volume profiles with explicit window/bin/value area parameters."""

    algo_version = "vp-eng-1"

    def __init__(
        self,
        storage: DataStorage,
        cache: MemoryCache,
        *,
        backfill_callback: Callable[[str, int, int, float], Any] | None = None,
        coverage_threshold: float = 0.8,
    ):
        self.storage = storage
        self.cache = cache
        self.logger = get_logger("indicators.volume_profile_engine")
        self.backfill_callback = backfill_callback
        self.coverage_threshold = coverage_threshold

    def _bin_price(self, price: float, bin_size: float) -> float:
        if bin_size <= 0:
            return price
        return math.floor(float(price) / bin_size) * bin_size

    async def _maybe_backfill(self, symbol: str, start_ms: int, end_ms: int, coverage_pct: float) -> None:
        if self.backfill_callback is None or coverage_pct >= self.coverage_threshold:
            return
        try:
            await self.backfill_callback(symbol, start_ms, end_ms, coverage_pct)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("profile_backfill_failed", symbol=symbol, error=str(exc))

    async def _coverage(self, symbol: str, start_ms: int, end_ms: int) -> dict[str, Any]:
        cov_raw = await self.storage.get_footprint_coverage(symbol, start_ms, end_ms)
        expected_minutes = max(1, int((end_ms - start_ms) / 60_000))
        buckets = int(cov_raw.get("minute_buckets") or cov_raw.get("minuteBuckets") or 0)
        coverage_pct = float(buckets / expected_minutes)
        return {
            "minuteBuckets": buckets,
            "expectedMinutes": expected_minutes,
            "coveragePct": round(coverage_pct, 4),
            "minTs": cov_raw.get("min_ts") or cov_raw.get("minTs"),
            "maxTs": cov_raw.get("max_ts") or cov_raw.get("maxTs"),
        }

    def _apply_mode(self, levels: dict[float, float], mode: str, normalize_seconds: int) -> dict[float, float]:
        if mode != "synthetic":
            return levels
        factor = max(0.01, float(normalize_seconds) / 60.0)
        return {p: v * factor for p, v in levels.items()}

    def _compress_levels(
        self,
        levels: dict[float, float],
        *,
        top_n: int | None,
        limit: int | None,
    ) -> tuple[list[dict[str, float]], bool]:
        if not levels:
            return [], False
        pairs = list(levels.items())
        truncated = False
        if top_n is not None:
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[: max(1, int(top_n))]
            truncated = len(pairs) < len(levels)
        elif limit is not None and len(pairs) > limit:
            pairs = sorted(pairs, key=lambda x: x[0])[:limit]
            truncated = True
        pairs = sorted(pairs, key=lambda x: x[0])
        return [{"price": float(p), "volume": float(v)} for p, v in pairs], truncated

    async def build_profile(
        self,
        symbol: str,
        *,
        window_start_ms: int,
        window_end_ms: int,
        bin_size: float,
        value_area_pct: float = 0.7,
        mode: str = "raw",
        normalize_seconds: int = 3,
        include_levels: bool = True,
        top_n_levels: int | None = None,
        price_level_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic profile aligned with Exocharts VR/Session expectations."""
        symbol = symbol.upper()
        if window_end_ms <= window_start_ms:
            raise ValueError("window_end_ms must be greater than window_start_ms")
        if bin_size is None or bin_size <= 0:
            raise ValueError("bin_size must be > 0 (Exocharts T value)")

        coverage = await self._coverage(symbol, window_start_ms, window_end_ms)
        await self._maybe_backfill(symbol, window_start_ms, window_end_ms, coverage["coveragePct"])
        coverage = await self._coverage(symbol, window_start_ms, window_end_ms)

        rows = await self.storage.get_profile_range(symbol, window_start_ms, window_end_ms)
        profile: dict[float, float] = {}
        for r in rows:
            price = float(r.get("price_level"))
            vol = float(r.get("buy_volume", 0.0) + r.get("sell_volume", 0.0))
            if vol <= 0:
                continue
            binned = self._bin_price(price, bin_size)
            profile[binned] = profile.get(binned, 0.0) + vol * price

        ws_trades = self.cache.get_recent_trades(symbol, since=window_start_ms)
        live_used = False
        for t in ws_trades:
            if t.timestamp < window_start_ms or t.timestamp > window_end_ms:
                continue
            live_used = True
            binned = self._bin_price(t.price, bin_size)
            profile[binned] = profile.get(binned, 0.0) + (t.quantity * t.price)

        prof_mode = self._apply_mode(profile, mode, normalize_seconds)
        poc, vah, val = ValueAreaCalculator.compute(prof_mode, value_area_pct)
        total = float(sum(prof_mode.values()))

        levels_out = None
        levels_truncated = False
        if include_levels:
            levels_out, levels_truncated = self._compress_levels(
                prof_mode,
                top_n=top_n_levels,
                limit=price_level_limit,
            )

        now = timestamp_ms()
        missing_ranges: list[dict[str, int]] = []
        if coverage["coveragePct"] < 1.0:
            if coverage.get("minTs") and coverage["minTs"] > window_start_ms:
                missing_ranges.append({"start": window_start_ms, "end": coverage["minTs"]})
            if coverage.get("maxTs") and coverage["maxTs"] < window_end_ms:
                missing_ranges.append({"start": coverage["maxTs"], "end": window_end_ms})

        data_source = "storage+ws" if live_used else "storage"
        dq = {
            "coveragePct": coverage["coveragePct"],
            "minuteBuckets": coverage["minuteBuckets"],
            "expectedMinutes": coverage["expectedMinutes"],
            "missingRanges": missing_ranges,
            "dataSource": data_source,
            "isDevelopingBar": window_end_ms > now,
            "dayIncomplete": coverage.get("maxTs") is not None and coverage["maxTs"] < window_end_ms and window_end_ms > now,
            "algoVersion": self.algo_version,
        }

        return {
            "symbol": symbol,
            "window": {
                "startTime": window_start_ms,
                "endTime": window_end_ms,
                "binSize": float(bin_size),
            },
            "mode": mode,
            "normalizeSeconds": int(normalize_seconds),
            "valueAreaPct": float(_as_fraction(value_area_pct)),
            "valueArea": {"vPOC": poc, "VAH": vah, "VAL": val},
            "totals": {"totalVolume": total, "binCount": len(prof_mode)},
            "levels": levels_out,
            "levelsTruncated": levels_truncated,
            "dataQuality": dq,
        }


__all__ = ["VolumeProfileEngine", "ValueAreaCalculator", "ProfileWindow"]
