"""Volume Profile calculator (POC, VAH, VAL)."""

from typing import Any
from collections import defaultdict

from src.data.storage import DataStorage
from src.config import get_settings
from src.utils import get_logger, timestamp_ms, round_to_tick
from src.utils.helpers import get_day_start_ms
from .profile_engine import ValueAreaCalculator

# Milliseconds in one day (UTC)
MS_IN_DAY = 86_400_000


class VolumeProfileCalculator:
    """Calculate Volume Profile with POC, VAH, VAL."""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.settings = get_settings()
        self.logger = get_logger("indicators.volume_profile")
        
        # In-memory *base* volume by price level for the current UTC day.
        #
        # IMPORTANT: this must be tied to a specific UTC day. Without tracking the
        # day, a long-running server would keep accumulating volume across day
        # boundaries, causing dPOC/dVAH/dVAL to drift.
        self._profiles: dict[str, dict[float, float]] = {}  # symbol -> {price_level -> base_volume}
        self._profile_day: dict[str, int] = {}  # symbol -> day_start_ms (UTC)
    
    async def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        buy_volume: float,
        sell_volume: float,
        timestamp: int,
    ) -> None:
        """Update volume profile with new trade data.
        
        Args:
            symbol: Trading pair symbol
            price: Trade price
            volume: Total trade volume
            buy_volume: Buy volume
            sell_volume: Sell volume
            timestamp: Trade timestamp in milliseconds
        """
        symbol = symbol.upper()
        tick_size = self.settings.get_tick_size(symbol)
        price_level = round_to_tick(price, tick_size)
        date = get_day_start_ms(timestamp)
        
        # Update in-memory profile
        # Reset automatically when we roll into a new UTC day.
        if self._profile_day.get(symbol) != date:
            self._profiles[symbol] = defaultdict(float)
            self._profile_day[symbol] = date
        
        # Track base volume for profile statistics (closer to exchange-reported volume/POC).
        base_volume = float(volume) * float(price)

        self._profiles[symbol][price_level] += base_volume
        
        # Persist to storage
        await self.storage.upsert_daily_trade(
            symbol=symbol,
            date=date,
            price_level=price_level,
            volume=volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            notional=price * volume,
        )
    
    def calculate_poc(self, profile: dict[float, float], mid_price: float | None = None) -> float | None:
        """Calculate Point of Control (price level with highest volume).

        If multiple prices tie for max volume, we break ties by choosing the level
        closest to `mid_price` (typically the midpoint of the profile range). If
        `mid_price` is not provided, we choose the middle price of the tied range.
        """
        if not profile:
            return None

        max_vol = max(profile.values())
        candidates = [p for p, v in profile.items() if v == max_vol]
        if len(candidates) == 1:
            return candidates[0]

        candidates_sorted = sorted(candidates)
        if mid_price is None:
            return candidates_sorted[len(candidates_sorted) // 2]

        return min(candidates_sorted, key=lambda p: (abs(p - mid_price), p))

    def calculate_value_area(
        self,
        profile: dict[float, float],
        value_area_percent: float = 70.0,
        *,
        percentage: float | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Calculate POC/VAH/VAL for a volume profile."""
        if percentage is not None:
            value_area_percent = float(percentage)
        return ValueAreaCalculator.compute(profile, value_area_percent)

    def build_volume_profile(self, rows: list[dict[str, Any]]) -> dict[float, float]:
        """Build a base-volume profile from storage rows.

        Expected row keys: price_level, buy_volume, sell_volume, total_volume or volume.
        """
        profile: dict[float, float] = {}
        for r in rows:
            try:
                price = float(r.get("price_level"))
            except Exception:
                continue

            buy_v = float(r.get("buy_volume") or 0.0)
            sell_v = float(r.get("sell_volume") or 0.0)
            total_v = float(r.get("total_volume") or r.get("volume") or 0.0)
            notional = float(r.get("notional") or 0.0)

            # Prefer explicit notional; otherwise multiply base volume by price level.
            base_v = notional if notional > 0 else (buy_v + sell_v)
            if base_v <= 0 and total_v > 0:
                base_v = total_v * price
            elif base_v <= 0:
                base_v = (buy_v + sell_v) * price

            if base_v == 0:
                continue

            profile[price] = float(profile.get(price, 0.0) + base_v)
        return profile
    
    async def get_today_profile(self, symbol: str) -> dict[float, float]:
        """Get today's volume profile.

        **Important:** prefer persisted storage (`daily_trades`) over the in-memory
        developing cache.

        Reason: after a backfill or a restart, the DB contains the full-day profile,
        while the in-memory cache may only contain *recent* live updates.
        """
        symbol = symbol.upper()
        today = get_day_start_ms(timestamp_ms())

        # Prefer storage snapshot because it survives restarts/backfills and is the
        # most complete source of truth.
        try:
            rows = await self.storage.get_daily_trades(symbol, today)
            if rows:
                profile = defaultdict(float)
                for row in rows:
                    base_v = float(row.get("buy_volume") or 0.0) + float(row.get("sell_volume") or 0.0)
                    if base_v <= 0:
                        base_v = float(row.get("volume") or 0.0)
                    if base_v <= 0:
                        continue
                    profile[row["price_level"]] = base_v

                # Keep cache in sync
                self._profiles[symbol] = profile
                self._profile_day[symbol] = today
                return dict(profile)
        except Exception as e:
            self.logger.warning(
                "volume_profile_storage_read_failed",
                symbol=symbol,
                error=str(e),
            )

        # Fallback: in-memory developing profile (may be partial).
        if (
            symbol in self._profiles
            and self._profiles[symbol]
            and self._profile_day.get(symbol) == today
        ):
            return dict(self._profiles[symbol])

        return {}

    async def get_yesterday_profile(self, symbol: str) -> dict[float, float]:
        """Get yesterday's volume profile.

        We prefer building the profile from the footprint (minute+price aggregates) because
        it is the most robust source and avoids relying on the per-day aggregation table
        (which can be incomplete after restarts if backfill was skipped).
        """
        symbol = symbol.upper()
        yesterday_start = get_day_start_ms(timestamp_ms()) - MS_IN_DAY
        yesterday_end = yesterday_start + MS_IN_DAY

        # 1) Preferred: footprint-derived profile (base volume)
        try:
            rows = await self.storage.get_profile_range(symbol, yesterday_start, yesterday_end)
            if rows:
                return self.build_volume_profile(rows)
        except Exception:
            # Fall back to daily_trades below.
            pass

        # 2) Fallback: daily_trades-derived profile (base volume)
        rows = await self.storage.get_daily_trades(symbol, yesterday_start)
        profile: dict[float, float] = {}
        for row in rows:
            base_v = float(row.get("buy_volume") or 0.0) + float(row.get("sell_volume") or 0.0)
            if base_v <= 0:
                base_v = float(row.get("volume") or 0.0)
            if base_v <= 0:
                continue
            profile[row["price_level"]] = base_v
        return profile
    
    async def get_key_levels(self, symbol: str, date: int | None = None) -> dict[str, Any]:
        """Get volume profile key levels.

        Notes:
            - Developing (d*) levels are calculated from today's footprint/daily aggregation.
            - Previous day (pd*) levels require footprint coverage for the full prior day.

        Returns:
            Dict with developing and previous-day volume profile levels.
        """
        symbol = symbol.upper()

        now_ms = timestamp_ms()

        # If `date` is provided, interpret it as the UTC day-start timestamp (ms)
        day_start = int(date) if date is not None else get_day_start_ms(now_ms)
        is_current_day = day_start == get_day_start_ms(now_ms)

        day_end = day_start + MS_IN_DAY
        dev_end = now_ms if is_current_day else day_end
        prev_day_start = day_start - MS_IN_DAY

        async def _load_profile(start_ms: int, end_ms: int, fallback_day: int | None = None) -> dict[float, float]:
            try:
                rows = await self.storage.get_profile_range(symbol, start_ms, end_ms)
                profile = self.build_volume_profile(rows)
                if profile:
                    return profile

                # Fallback: if we queried a full UTC day and footprint is empty, reuse daily snapshot.
                if fallback_day is not None:
                    daily_rows = await self.storage.get_daily_trades(symbol, fallback_day)
                    return self.build_volume_profile(daily_rows)
            except Exception as e:
                self.logger.warning(
                    "volume_profile_storage_read_failed",
                    symbol=symbol,
                    error=str(e),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
                return {}

        # Profiles (base-volume)
        today_profile = await _load_profile(day_start, dev_end, fallback_day=day_start)
        d_poc, d_vah, d_val = self.calculate_value_area(today_profile)

        yesterday_profile = await _load_profile(prev_day_start, day_start, fallback_day=prev_day_start)
        pd_poc, pd_vah, pd_val = self.calculate_value_area(yesterday_profile)

        # Coverage / completeness diagnostics
        def _coverage(info: dict[str, Any], start_ms: int, end_ms: int) -> dict[str, Any]:
            expected = int((end_ms - start_ms + 59999) // 60000)
            buckets = int(info.get('minuteBuckets') or info.get('minute_buckets') or 0)
            pct = float(buckets / expected) if expected > 0 else 0.0
            return {
                'minuteBuckets': buckets,
                'expectedMinutes': expected,
                'coveragePct': round(pct, 4),
                'minTs': info.get('minTs') or info.get('min_ts'),
                'maxTs': info.get('maxTs') or info.get('max_ts'),
            }

        try:
            cov_today_raw = await self.storage.get_footprint_coverage(symbol, day_start, dev_end)
            cov_yesterday_raw = await self.storage.get_footprint_coverage(symbol, prev_day_start, day_start)
        except Exception:
            cov_today_raw = {'minuteBuckets': 0, 'minTs': None, 'maxTs': None}
            cov_yesterday_raw = {'minuteBuckets': 0, 'minTs': None, 'maxTs': None}

        cov_today = _coverage(cov_today_raw, day_start, dev_end)
        cov_yesterday = _coverage(cov_yesterday_raw, prev_day_start, day_start)
        pd_complete = cov_yesterday['coveragePct'] >= 0.95

        return {
            'symbol': symbol,
            'timestamp': now_ms,
            'dayStart': day_start,
            'dayEnd': day_end,
            'isCurrentDay': is_current_day,
            'developing': {
                'POC': d_poc,
                'VAH': d_vah,
                'VAL': d_val,
                'high': max(today_profile.keys()) if today_profile else None,
                'low': min(today_profile.keys()) if today_profile else None,
                'totalVolume': float(sum(today_profile.values())) if today_profile else 0.0,
                'priceLevels': len(today_profile),
                'coverage': cov_today,
            },
            'previousDay': {
                'POC': pd_poc,
                'VAH': pd_vah,
                'VAL': pd_val,
                'high': max(yesterday_profile.keys()) if yesterday_profile else None,
                'low': min(yesterday_profile.keys()) if yesterday_profile else None,
                'totalVolume': float(sum(yesterday_profile.values())) if yesterday_profile else 0.0,
                'priceLevels': len(yesterday_profile),
                'coverage': cov_yesterday,
                'complete': pd_complete,
            },
            'unit': 'USDT',
        }

    def reset_day(self, symbol: str) -> None:
        """Reset daily profile (called at day rollover)."""
        symbol = symbol.upper()
        self._profiles[symbol] = defaultdict(float)
        self._profile_day[symbol] = get_day_start_ms(timestamp_ms())
        self.logger.info("profile_reset", symbol=symbol)


def calculate_volume_profile(
    trades: list[dict[str, Any]],
    tick_size: float,
) -> dict[float, dict[str, float]]:
    """Calculate volume profile from trades.
    
    Args:
        trades: List of trade dicts with 'price', 'volume', 'buy_volume', 'sell_volume'
        tick_size: Price tick size for aggregation
    
    Returns:
        Dict mapping price levels to volume breakdown
    """
    profile: dict[float, dict[str, float]] = defaultdict(
        lambda: {"volume": 0, "buy_volume": 0, "sell_volume": 0, "delta": 0}
    )
    
    for trade in trades:
        price_level = round_to_tick(trade["price"], tick_size)
        buy_vol = trade.get("buy_volume", 0)
        sell_vol = trade.get("sell_volume", 0)
        
        profile[price_level]["volume"] += trade["volume"]
        profile[price_level]["buy_volume"] += buy_vol
        profile[price_level]["sell_volume"] += sell_vol
        profile[price_level]["delta"] += buy_vol - sell_vol
    
    return dict(profile)
