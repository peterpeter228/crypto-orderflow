"""Volume Profile calculator (POC, VAH, VAL)."""

from typing import Any, Iterable, Tuple
from collections import defaultdict

from src.data.storage import DataStorage
from src.binance.rest_client import BinanceRestClient
from src.binance.types import Kline
from src.config import get_settings
from src.utils import get_logger, timestamp_ms, round_to_tick
from src.utils.helpers import get_day_start_ms

# Milliseconds in one day (UTC)
MS_IN_DAY = 86_400_000


class VolumeProfileCalculator:
    """Calculate Volume Profile with POC, VAH, VAL."""
    
    def __init__(self, storage: DataStorage, rest_client: BinanceRestClient | None = None):
        self.storage = storage
        self.rest_client = rest_client
        self.settings = get_settings()
        self.logger = get_logger("indicators.volume_profile")
        
        # In-memory *quote* volume (synthetic: price*qty, USDT) by price level for current day.
        #
        # IMPORTANT: this must be tied to a specific UTC day. Without tracking the
        # day, a long-running server would keep accumulating volume across day
        # boundaries, causing dPOC/dVAH/dVAL to drift.
        self._profiles: dict[str, dict[float, float]] = {}  # symbol -> {price_level -> quote_volume_usdt}
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
        
        # Use synthetic/quote volume (USDT) so POC/VA matches most orderflow platforms
        notional = price * volume

        self._profiles[symbol][price_level] += notional
        
        # Persist to storage
        await self.storage.upsert_daily_trade(
            symbol=symbol,
            date=date,
            price_level=price_level,
            volume=volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            notional=notional,
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
        """Calculate POC/VAH/VAL for a volume profile.

        Note: some call-sites (older versions) used the kwarg name `percentage`.
        We accept both `value_area_percent` and `percentage` for backward
        compatibility.
        """
        if percentage is not None:
            value_area_percent = float(percentage)
        if not profile:
            return None, None, None

        sorted_prices = sorted(profile.keys())
        if len(sorted_prices) == 1:
            price = sorted_prices[0]
            return price, price, price

        total_volume = float(sum(profile.values()))
        if total_volume <= 0:
            return None, None, None

        target_volume = total_volume * (value_area_percent / 100.0)

        # POC (tie-break towards midpoint of the full range)
        mid_price = (sorted_prices[0] + sorted_prices[-1]) / 2.0
        poc = self.calculate_poc(profile, mid_price)
        if poc is None:
            return None, None, None

        # Index of POC within sorted prices
        try:
            poc_idx = sorted_prices.index(poc)
        except ValueError:
            # If rounding differences occur, pick nearest price level.
            poc = min(sorted_prices, key=lambda p: abs(p - poc))
            poc_idx = sorted_prices.index(poc)

        vah_idx = poc_idx
        val_idx = poc_idx
        current_volume = float(profile[poc])

        while current_volume < target_volume and (vah_idx + 1 < len(sorted_prices) or val_idx - 1 >= 0):
            up_idx = vah_idx + 1
            down_idx = val_idx - 1

            up_vol = float(profile.get(sorted_prices[up_idx], 0.0)) if up_idx < len(sorted_prices) else -1.0
            down_vol = float(profile.get(sorted_prices[down_idx], 0.0)) if down_idx >= 0 else -1.0

            # No more volume to add
            if up_vol <= 0 and down_vol <= 0:
                break

            if up_vol > down_vol:
                vah_idx = up_idx
                current_volume += up_vol
            elif down_vol > up_vol:
                val_idx = down_idx
                current_volume += down_vol
            else:
                # Equal: include both sides when possible (avoid double counting same index)
                added = False
                if up_idx < len(sorted_prices):
                    vah_idx = up_idx
                    current_volume += up_vol
                    added = True
                if down_idx >= 0 and down_idx != up_idx:
                    val_idx = down_idx
                    current_volume += down_vol
                    added = True
                if not added:
                    break

        vah = sorted_prices[vah_idx]
        val = sorted_prices[val_idx]
        return poc, vah, val

    def _build_quote_profile_with_vwap(
        self, rows: Iterable[dict[str, Any]], *, include_total_volume: bool = False
    ) -> Tuple[dict[float, float], float | None]:
        """Build a quote-denominated profile and VWAP from aggregated rows.

        Returns a tuple of (profile, vwap). VWAP is computed from the sum of base
        volume and quote notional when available.
        """
        profile: dict[float, float] = {}
        total_quote = 0.0
        total_base = 0.0

        for r in rows:
            try:
                price = float(r.get("price_level"))
            except Exception:
                continue

            buy_v = float(r.get("buy_volume") or 0.0)
            sell_v = float(r.get("sell_volume") or 0.0)
            total_v = float(r.get("total_volume") or 0.0) if include_total_volume else 0.0

            # Prefer buy+sell if provided, otherwise fall back to total_volume.
            base_v = buy_v + sell_v
            if base_v <= 0 and total_v > 0:
                base_v = total_v

            quote_v = base_v * price
            if quote_v == 0:
                continue

            profile[price] = float(profile.get(price, 0.0) + quote_v)
            total_quote += quote_v
            total_base += base_v

        vwap = (total_quote / total_base) if total_base > 0 else None
        return profile, vwap

    def build_quote_profile(self, rows: list[dict[str, Any]]) -> dict[float, float]:
        """Backward-compatible wrapper to build a quote profile from rows."""
        profile, _ = self._build_quote_profile_with_vwap(rows, include_total_volume=True)
        return profile

    def build_kline_profile(self, symbol: str, klines: list[Kline]) -> tuple[dict[float, float], float | None]:
        """Build a coarse quote-volume profile and VWAP from klines."""
        tick_size = self.settings.get_tick_size(symbol)
        profile: dict[float, float] = defaultdict(float)
        total_quote = 0.0
        total_base = 0.0

        for k in klines:
            try:
                typical_price = (float(k.high) + float(k.low) + float(k.close)) / 3.0
            except Exception:
                continue

            base_v = float(k.volume or 0.0)
            quote_v = float(k.quote_volume or 0.0)
            if quote_v <= 0 and base_v > 0:
                quote_v = base_v * typical_price

            if quote_v <= 0:
                continue

            level = round_to_tick(typical_price, tick_size) if tick_size and tick_size > 0 else typical_price
            profile[level] += quote_v
            total_quote += quote_v
            if base_v > 0:
                total_base += base_v

        vwap = (total_quote / total_base) if total_base > 0 else None
        return dict(profile), vwap
    
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
                    profile[row["price_level"]] = row["notional"]

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

        # 1) Preferred: footprint-derived profile (synthetic/quote volume)
        try:
            rows = await self.storage.get_profile_range(symbol, yesterday_start, yesterday_end)
            if rows:
                return self.build_quote_profile(rows)
        except Exception:
            # Fall back to daily_trades below.
            pass

        # 2) Fallback: daily_trades-derived profile (also quote/notional)
        rows = await self.storage.get_daily_trades(symbol, yesterday_start)
        profile: dict[float, float] = {}
        for row in rows:
            profile[row["price_level"]] = float(row.get("notional", 0.0))
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

        async def _load_profile(start_ms: int, end_ms: int) -> tuple[dict[float, float], str | None, float | None]:
            """Load profile with multi-level fallback.

            Order:
                1) footprint_1m (highest fidelity)
                2) daily_trades
                3) Binance klines (coarse backup)
            """
            try:
                rows = await self.storage.get_profile_range(symbol, start_ms, end_ms)
                profile, vwap = self._build_quote_profile_with_vwap(rows, include_total_volume=True)
                if profile:
                    return profile, "footprint_1m", vwap
            except Exception as e:
                self.logger.warning(
                    "volume_profile_storage_read_failed",
                    symbol=symbol,
                    error=str(e),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )

            try:
                rows = await self.storage.get_daily_trades(symbol, start_ms)
                if rows:
                    profile: dict[float, float] = defaultdict(float)
                    total_quote = 0.0
                    total_base = 0.0
                    for row in rows:
                        price = float(row.get("price_level"))
                        notional = float(row.get("notional") or 0.0)
                        base_v = float(row.get("volume") or 0.0)
                        if notional <= 0 and base_v > 0:
                            notional = base_v * price
                        if notional <= 0:
                            continue
                        profile[price] += notional
                        total_quote += notional
                        if base_v > 0:
                            total_base += base_v
                    vwap = (total_quote / total_base) if total_base > 0 else None
                    return dict(profile), "daily_trades", vwap
            except Exception as e:
                self.logger.warning(
                    "volume_profile_daily_trades_failed",
                    symbol=symbol,
                    error=str(e),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )

            if self.rest_client:
                try:
                    klines = await self.rest_client.get_klines(
                        symbol=symbol,
                        interval="1m",
                        start_time=start_ms,
                        end_time=end_ms,
                        limit=1500,
                    )
                    profile, vwap = self.build_kline_profile(symbol, klines)
                    if profile:
                        return profile, "binance_klines", vwap
                except Exception as e:
                    self.logger.warning(
                        "volume_profile_kline_fallback_failed",
                        symbol=symbol,
                        error=str(e),
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )

            return {}, None, None

        # Profiles (quote-denominated)
        today_profile, today_source, today_vwap = await _load_profile(day_start, dev_end)
        d_poc, d_vah, d_val = self.calculate_value_area(today_profile)

        yesterday_profile, yesterday_source, yesterday_vwap = await _load_profile(prev_day_start, day_start)
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
                'VWAP': today_vwap,
                'high': max(today_profile.keys()) if today_profile else None,
                'low': min(today_profile.keys()) if today_profile else None,
                'totalVolume': float(sum(today_profile.values())) if today_profile else 0.0,
                'priceLevels': len(today_profile),
                'coverage': cov_today,
                'source': today_source,
                'usingFallback': today_source not in (None, "footprint_1m"),
            },
            'previousDay': {
                'POC': pd_poc,
                'VAH': pd_vah,
                'VAL': pd_val,
                'VWAP': yesterday_vwap,
                'high': max(yesterday_profile.keys()) if yesterday_profile else None,
                'low': min(yesterday_profile.keys()) if yesterday_profile else None,
                'totalVolume': float(sum(yesterday_profile.values())) if yesterday_profile else 0.0,
                'priceLevels': len(yesterday_profile),
                'coverage': cov_yesterday,
                'complete': pd_complete,
                'source': yesterday_source,
                'usingFallback': yesterday_source not in (None, "footprint_1m"),
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
