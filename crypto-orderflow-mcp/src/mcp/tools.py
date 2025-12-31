"""MCP Tools definitions for Crypto Orderflow."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any

from src.config import get_settings

from src.data.cache import MemoryCache
from src.data.storage import DataStorage
from src.data.backfill import AggTradesBackfiller
from src.data.orderbook import OrderbookManager
from src.binance.rest_client import BinanceRestClient
from src.indicators import (
    VWAPCalculator,
    VolumeProfileCalculator,
    VolumeProfileEngine,
    ValueAreaCalculator,
    SessionLevelsCalculator,
    FootprintCalculator,
    DeltaCVDCalculator,
    ImbalanceDetector,
    DepthDeltaCalculator,
    OrderbookHeatmapSampler,
    TPOProfileCalculator,
)
from src.utils import get_logger, timestamp_ms
from src.utils.helpers import get_day_start_ms, get_timeframe_ms


def build_quote_profile(
    rows: list[dict[str, Any]] | list[Any],
    bin_size: float,
    *,
    value_area_percent: float = 70.0,
    normalize_seconds: int = 3,
    mode: str = "raw",
) -> dict[str, Any]:
    """Aggregate quote notional into price bins and compute key levels.

    Args:
        rows: Footprint-style rows (preferred) or kline-derived rows. Expected keys include
            price_level/price/close, buy_volume & sell_volume (base) or buy_quote/sell_quote,
            optional quote_volume + taker_buy_quote_volume for kline fallback.
        bin_size: Price bucket size.
        value_area_percent: Value area percentage (e.g., 70 for 70%).
        normalize_seconds: Synthetic mode scaler (aligned with VolumeProfileEngine).
        mode: "raw" or "synthetic"; synthetic scales totals by normalize_seconds/60.

    Returns:
        Dict with histogram, vPOC/VAH/VAL, totals, delta, and quality flags.
    """

    def _get(row: dict[str, Any] | Any, key: str) -> Any:
        if isinstance(row, dict):
            return row.get(key)
        return getattr(row, key, None)

    if bin_size is None or bin_size <= 0:
        raise ValueError("bin_size must be positive for quote profile aggregation")

    histogram: dict[float, float] = {}
    buy_total = 0.0
    sell_total = 0.0
    base_total = 0.0
    trade_count = 0

    for row in rows or []:
        price = None
        for key in ("price_level", "price", "close", "open"):
            val = _get(row, key)
            if val is not None:
                price = float(val)
                break
        if price is None:
            continue

        buy_quote = None
        sell_quote = None

        buy_base = _get(row, "buy_volume") or _get(row, "buyVolume") or _get(row, "buy_qty") or _get(row, "buyQty")
        sell_base = _get(row, "sell_volume") or _get(row, "sellVolume") or _get(row, "sell_qty") or _get(row, "sellQty")

        for key in ("buy_quote", "buyNotional", "buy_notional"):
            val = _get(row, key)
            if val is not None:
                buy_quote = float(val)
                break

        for key in ("sell_quote", "sellNotional", "sell_notional"):
            val = _get(row, key)
            if val is not None:
                sell_quote = float(val)
                break

        if buy_quote is None and buy_base is not None:
            buy_quote = float(buy_base) * price
        if sell_quote is None and sell_base is not None:
            sell_quote = float(sell_base) * price

        quote_volume = _get(row, "quote_volume") or _get(row, "quoteVolume") or _get(row, "volumeQuote") or _get(row, "volume_quote")
        taker_buy_quote = _get(row, "taker_buy_quote_volume") or _get(row, "takerBuyQuoteVolume")
        if buy_quote is None and quote_volume is not None:
            buy_quote = float(taker_buy_quote or 0.0)
        if sell_quote is None and quote_volume is not None:
            sell_quote = max(0.0, float(quote_volume) - float(buy_quote or 0.0))

        bq = float(buy_quote or 0.0)
        sq = float(sell_quote or 0.0)
        if bq <= 0 and sq <= 0:
            continue

        base_total += float(buy_base or 0.0) + float(sell_base or 0.0)
        buy_total += bq
        sell_total += sq

        binned = math.floor(price / bin_size) * bin_size
        histogram[binned] = histogram.get(binned, 0.0) + bq + sq

        trade_count += int(_get(row, "trade_count") or _get(row, "tradeCount") or 0)

    if mode == "synthetic":
        factor = max(0.01, float(normalize_seconds) / 60.0)
        histogram = {p: v * factor for p, v in histogram.items()}

    v_poc = v_vah = v_val = None
    quality_flags: list[str] = []
    if histogram:
        v_poc, v_vah, v_val = ValueAreaCalculator.compute(histogram, value_area_percent)
    else:
        quality_flags.append("insufficient_data")

    total_quote = float(sum(histogram.values())) if histogram else 0.0
    delta_quote = float(buy_total - sell_total)
    high = max(histogram.keys()) if histogram else None
    low = min(histogram.keys()) if histogram else None

    return {
        "success": True,
        "quality_flags": quality_flags,
        "histogram": histogram,
        "vPOC": v_poc,
        "vVAH": v_vah,
        "vVAL": v_val,
        "volumeQuote": total_quote,
        "buyQuote": float(buy_total),
        "sellQuote": float(sell_total),
        "deltaQuote": delta_quote,
        "tradeCount": int(trade_count),
        "binCount": len(histogram),
        "high": high,
        "low": low,
        "baseVolume": float(base_total),
        "mode": mode,
        "valueAreaPercent": float(value_area_percent),
        "degraded": bool(quality_flags),
    }


class MCPTools:
    """MCP Tools for Crypto Orderflow indicators."""
    
    def __init__(
        self,
        cache: MemoryCache,
        storage: DataStorage,
        orderbook: OrderbookManager,
        rest_client: BinanceRestClient,
        vwap: VWAPCalculator,
        volume_profile: VolumeProfileCalculator,
        tpo_profile: TPOProfileCalculator,
        session_levels: SessionLevelsCalculator,
        footprint: FootprintCalculator,
        delta_cvd: DeltaCVDCalculator,
        imbalance: ImbalanceDetector,
        depth_delta: DepthDeltaCalculator,
        heatmap: OrderbookHeatmapSampler | None = None,
        profile_engine: VolumeProfileEngine | None = None,
    ):
        self.settings = get_settings()
        self.cache = cache
        self.storage = storage
        self.orderbook = orderbook
        self.rest_client = rest_client
        self.vwap = vwap
        self.volume_profile = volume_profile
        self.tpo_profile = tpo_profile
        self.session_levels = session_levels
        self.footprint = footprint
        self.delta_cvd = delta_cvd
        self.imbalance = imbalance
        self.depth_delta = depth_delta
        self.heatmap = heatmap
        self.profile_engine = profile_engine
        self.logger = get_logger("mcp.tools")
        # Data quality thresholds (tunable)
        self.coverage_threshold = 0.8
        self.min_bars = 20
        self.min_levels = 20
        # On-demand backfill can be toggled in tests to avoid network calls.
        self.on_demand_backfill_enabled = True

        if self.profile_engine is None:
            self.profile_engine = VolumeProfileEngine(
                storage=self.storage,
                cache=self.cache,
                backfill_callback=self._maybe_backfill_footprint,
                coverage_threshold=self.coverage_threshold,
            )

    def _build_data_quality(
        self,
        *,
        window_start_ms: int,
        window_end_ms: int,
        coverage_raw: dict[str, Any] | None,
        bars_returned: int | None,
        levels_count: int | None = None,
        source: str = "local",
        reasons_prefix: list[str] | None = None,
    ) -> dict[str, Any]:
        """Construct a normalized data quality block."""
        expected_minutes = max(1, int((window_end_ms - window_start_ms) / 60_000))
        raw_buckets = coverage_raw or {}
        minute_buckets = int(
            raw_buckets.get("minute_buckets")
            or raw_buckets.get("minuteBuckets")
            or 0
        )
        min_ts = raw_buckets.get("min_ts") or raw_buckets.get("minTs")
        max_ts = raw_buckets.get("max_ts") or raw_buckets.get("maxTs")
        coverage_pct = float(minute_buckets / expected_minutes) if expected_minutes else 0.0

        reasons: list[str] = []
        if reasons_prefix:
            reasons.extend(reasons_prefix)
        if coverage_pct < self.coverage_threshold:
            reasons.append("low_coverage")
        if bars_returned is not None and bars_returned < self.min_bars:
            reasons.append("insufficient_bars")
        if levels_count is not None and levels_count < self.min_levels:
            reasons.append("insufficient_levels")

        degraded = bool(reasons)
        missing_minutes = max(0, expected_minutes - minute_buckets)
        remediation = None
        if degraded and coverage_pct < self.coverage_threshold:
            remediation = (
                f"Backfill ~{round(missing_minutes / 60, 2)}h of footprint_1m "
                f"between {window_start_ms} and {window_end_ms}"
            )

        return {
            "windowStartMs": window_start_ms,
            "windowEndMs": window_end_ms,
            "expectedMinutes": expected_minutes,
            "barsReturned": bars_returned,
            "coveragePct": round(coverage_pct, 4),
            "minuteBuckets": minute_buckets,
            "levelsCount": levels_count,
            "minTimestamp": min_ts,
            "maxTimestamp": max_ts,
            "source": source,
            "degraded": degraded,
            "reasons": reasons,
            "remediation": remediation,
        }

    async def _maybe_backfill_footprint(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        *,
        coverage_pct: float,
    ) -> bool:
        """Attempt a limited on-demand backfill when coverage is poor."""
        if (
            not self.on_demand_backfill_enabled
            or not getattr(self.settings, "backfill_enabled", False)
        ):
            return False
        if coverage_pct >= self.coverage_threshold:
            return False

        span_ms = max(0, end_ms - start_ms)
        # Avoid overly large backfills from interactive calls.
        max_span_ms = int(12 * 3_600_000)
        if span_ms > max_span_ms:
            start_ms = end_ms - max_span_ms

        try:
            backfiller = AggTradesBackfiller(self.rest_client, self.storage)
            max_requests = (
                self.settings.backfill_max_requests_per_symbol
                if getattr(self.settings, "backfill_max_requests_per_symbol", 0) > 0
                else None
            )
            await backfiller.backfill_symbol_range(
                symbol=symbol,
                start_time=start_ms,
                end_time=end_ms,
                clear_days=False,
                pause_ms=getattr(self.settings, "backfill_request_pause_ms", 250),
                max_requests=max_requests,
            )
            try:
                await backfiller.close()
            except Exception:
                pass
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "on_demand_backfill_failed",
                symbol=symbol,
                error=str(exc),
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return False

    def _get_tpo_tick_size_safe(self, symbol: str) -> float:
        """Get TPO tick size with fallbacks for missing settings fields."""
        symbol = symbol.upper()

        has_helper = hasattr(self.settings, "get_tpo_tick_size")
        has_btc_tick = hasattr(self.settings, "tpo_tick_size_btc")
        has_eth_tick = hasattr(self.settings, "tpo_tick_size_eth")

        # If the specialized tick sizes are missing, fall back to conservative defaults.
        fallback_map = {
            "BTC": 70.0,
            "ETH": 5.0,
        }

        # Prefer the configured helper when available and relevant fields exist.
        if has_helper and ((has_btc_tick and "BTC" in symbol) or (has_eth_tick and "ETH" in symbol)):
            try:
                return float(self.settings.get_tpo_tick_size(symbol))
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "tpo_tick_size_helper_failed",
                    symbol=symbol,
                    error=str(exc),
                )

        # Use hardcoded defaults when symbol matches and fields are missing.
        for key, default_value in fallback_map.items():
            if key in symbol and not (has_btc_tick if key == "BTC" else has_eth_tick):
                self.logger.warning(
                    "tpo_tick_size_fallback_used",
                    symbol=symbol,
                    fallback=default_value,
                    reason="missing_settings_field",
                )
                return default_value

        # As a last resort, try the helper even if the expected fields are absent; it guards internally.
        if has_helper:
            return float(self.settings.get_tpo_tick_size(symbol))

        # Ultimate safety net: choose ETH default for alts and BTC for everything else.
        default_value = fallback_map["BTC"] if "BTC" in symbol else fallback_map["ETH"]
        self.logger.warning(
            "tpo_tick_size_fallback_used",
            symbol=symbol,
            fallback=default_value,
            reason="missing_helper",
        )
        return default_value

    @staticmethod
    def _get_day_start_ms_in_tz(timestamp_ms: int, tzinfo: ZoneInfo) -> int:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).astimezone(tzinfo)
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(day_start.timestamp() * 1000)

    def parse_date_or_today(
        self,
        date_str: str | None,
        session_tz: str = "UTC",
        *,
        now_ms: int | None = None,
    ) -> tuple[int, bool]:
        """Parse a YYYY-MM-DD date string or fall back to the current day.

        Returns a tuple of (day_start_ms, is_current_day).
        """

        tz_name = session_tz or "UTC"
        try:
            tzinfo = ZoneInfo(tz_name)
        except Exception as exc:
            raise ValueError(f"Invalid session timezone: {tz_name}") from exc

        now_ms = now_ms or timestamp_ms()
        today_start_ms = self._get_day_start_ms_in_tz(now_ms, tzinfo)

        if date_str is None or str(date_str).strip() == "":
            return today_start_ms, True

        try:
            parsed = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").replace(tzinfo=tzinfo)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc

        day_start_ms = parsed.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
        day_start_ms = int(day_start_ms)
        return day_start_ms, day_start_ms == today_start_ms
    
    async def get_market_snapshot(self, symbol: str) -> dict[str, Any]:
        """Get market snapshot including price, funding, OI.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
        Returns:
            Market snapshot with latest price, mark price, 24h stats, funding, OI
        """
        symbol = symbol.upper()
        self.logger.info("get_market_snapshot", symbol=symbol)
        
        # Prefer fresh REST data when the local cache hasn't warmed up yet.
        # (CherryStudio often queries immediately after connecting; relying only on
        # websocket-updated cache can show stale / mismatched values vs TradingView.)
        cache_obj = self.cache.get_cache(symbol)
        now = timestamp_ms()
        needs_refresh = (cache_obj.last_trade_time == 0) or (now - cache_obj.last_trade_time > 3_000)

        if needs_refresh:
            try:
                ticker = await self.rest_client.get_ticker_24h(symbol)
                mark = await self.rest_client.get_mark_price(symbol)
                oi = await self.rest_client.get_open_interest(symbol)

                cache_obj.last_price = ticker.last_price
                cache_obj.high_24h = ticker.high_price
                cache_obj.low_24h = ticker.low_price
                cache_obj.volume_24h = ticker.volume
                cache_obj.quote_volume_24h = ticker.quote_volume

                cache_obj.mark_price = mark.mark_price
                cache_obj.index_price = mark.index_price
                cache_obj.funding_rate = mark.funding_rate
                cache_obj.next_funding_time = mark.next_funding_time

                cache_obj.open_interest = oi.open_interest
                cache_obj.open_interest_notional = oi.open_interest_notional

                # Not a real trade time, but indicates "fresh as of" for snapshot.
                cache_obj.last_trade_time = now
            except Exception as e:
                # Fall back to whatever we have in memory.
                self.logger.warning("snapshot_refresh_failed", symbol=symbol, error=str(e))

        snapshot = self.cache.get_snapshot(symbol)
        
        # Enhance with additional data if available
        depth_summary = self.depth_delta.get_depth_summary(symbol)
        if depth_summary:
            snapshot["depthBidVolume"] = depth_summary["bidVolume"]
            snapshot["depthAskVolume"] = depth_summary["askVolume"]
            snapshot["depthNetVolume"] = depth_summary["netVolume"]
            snapshot["depthBidAskRatio"] = depth_summary["bidAskRatio"]
        
        return snapshot
    
    async def get_key_levels(
        self,
        symbol: str,
        date: str | None = None,
        session_tz: str = "UTC",
        *,
        bin_size: float | None = None,
        value_area_pct: float = 70.0,
        mode: str = "raw",
        normalize_seconds: int = 3,
    ) -> dict[str, Any]:
        """Get key price levels for trading analysis.

        Combines:
        - VWAP: dVWAP, pdVWAP
        - Volume Profile: dPOC/dVAH/dVAL + previous day equivalents
        - Session High/Low: configurable sessions from `SESSIONS`

        Notes:
        - `session_tz` is currently informational; session windows are tracked in UTC.
        """
        symbol = symbol.upper()
        now = timestamp_ms()

        try:
            day_start_ms, is_current_day = self.parse_date_or_today(date, session_tz, now_ms=now)
        except ValueError as exc:
            return {
                "success": False,
                "error": {
                    "message": str(exc),
                    "code": "invalid_date",
                    "input": date,
                },
            }

        quality_flags: list[str] = []
        day_end_ms = day_start_ms + 86_400_000 - 1
        dev_window_end_ms = now if is_current_day else day_end_ms
        if dev_window_end_ms < day_end_ms:
            quality_flags.append("using_partial_session_data")

        # Get VWAP levels
        vwap_levels = await self.vwap.get_key_levels(symbol, date=day_start_ms)

        # Volume Profile (aligned with profile engine / Exocharts binning)
        bin_sz = float(bin_size or self.settings.get_tpo_tick_size(symbol))
        va_pct = float(value_area_pct)
        prev_day_start = day_start_ms - 86_400_000
        prev_day_end_ms = day_start_ms - 1

        dev_profile = await self.profile_engine.build_profile(
            symbol,
            window_start_ms=day_start_ms,
            window_end_ms=dev_window_end_ms,
            bin_size=bin_sz,
            value_area_pct=va_pct,
            mode=mode,
            normalize_seconds=normalize_seconds,
            include_levels=False,
        )
        prev_profile = await self.profile_engine.build_profile(
            symbol,
            window_start_ms=prev_day_start,
            window_end_ms=prev_day_end_ms,
            bin_size=bin_sz,
            value_area_pct=va_pct,
            mode=mode,
            normalize_seconds=normalize_seconds,
            include_levels=False,
        )

        def _profile_to_dict(p: dict[str, Any]) -> dict[str, Any]:
            return {
                "POC": (p.get("valueArea") or {}).get("vPOC"),
                "VAH": (p.get("valueArea") or {}).get("VAH"),
                "VAL": (p.get("valueArea") or {}).get("VAL"),
                "high": None,
                "low": None,
                "totalVolume": (p.get("totals") or {}).get("totalVolume"),
                "priceLevels": (p.get("totals") or {}).get("binCount"),
                "coverage": p.get("dataQuality") or {},
                "window": p.get("window"),
                "binSize": (p.get("window") or {}).get("binSize"),
                "valueAreaPct": p.get("valueAreaPct"),
            }

        vp_levels = {
            "developing": _profile_to_dict(dev_profile),
            "previousDay": _profile_to_dict(prev_profile),
        }

        # Get Session levels
        # NOTE: SessionLevelsCalculator.get_key_levels() accepts `date` (ms day_start), not `date_ms`.
        # `session_tz` is informational for now; sessions are tracked in UTC.
        session_levels = await self.session_levels.get_key_levels(symbol, date=day_start_ms)

        # Flatten the most commonly used labels (matches many charting packages)
        developing = (vp_levels or {}).get("developing", {})
        previous = (vp_levels or {}).get("previousDay", {})
        today_sess = (session_levels or {}).get("today", {})
        y_sess = (session_levels or {}).get("yesterday", {})

        def _latest_session_value(key: str) -> Any:
            """Prefer today's session value, fall back to yesterday.

            This matches common TradingView scripts (e.g. Leviathan Key Levels) where, before a
            session starts today, the most recent completed session levels are still plotted.
            """
            val = today_sess.get(key)
            return val if val is not None else y_sess.get(key)

        flat: dict[str, Any] = {
            # VWAP
            "dVWAP": (vwap_levels or {}).get("dVWAP"),
            "pdVWAP": (vwap_levels or {}).get("pdVWAP"),
            # Developing volume profile
            "dPOC": developing.get("POC"),
            "dVAH": developing.get("VAH"),
            "dVAL": developing.get("VAL"),
            "dH": developing.get("high"),
            "dL": developing.get("low"),
            # Previous day volume profile
            "pdPOC": previous.get("POC"),
            "pdVAH": previous.get("VAH"),
            "pdVAL": previous.get("VAL"),
            "pdH": previous.get("high"),
            "pdL": previous.get("low"),
        }

        # Session levels (latest known for each configured session)
        for s in self.settings.session_defs:
            h_key = f"{s.name}H"
            l_key = f"{s.name}L"
            flat[h_key] = _latest_session_value(h_key)
            flat[l_key] = _latest_session_value(l_key)
            flat[f"{h_key}_y"] = y_sess.get(h_key)
            flat[f"{l_key}_y"] = y_sess.get(l_key)

        # Data quality gating for VP-based levels
        dev_cov = developing.get("coverage") or {}
        prev_cov = previous.get("coverage") or {}

        dev_degraded = (dev_cov.get("coveragePct", 0.0) < self.coverage_threshold) and not dev_cov.get("dayIncomplete")
        prev_degraded = (prev_cov.get("coveragePct", 0.0) < self.coverage_threshold)

        if dev_degraded:
            for key in ("dPOC", "dVAH", "dVAL", "dVWAP"):
                flat[key] = None
        if prev_degraded:
            for key in ("pdPOC", "pdVAH", "pdVAL", "pdVWAP"):
                flat[key] = None

        session_windows = []
        for sdef in self.settings.session_defs:
            start_ms = day_start_ms + int(sdef.start_minutes) * 60_000
            end_ms = day_start_ms + int(sdef.end_minutes) * 60_000
            if sdef.end_minutes <= sdef.start_minutes:
                end_ms += 86_400_000
            session_windows.append(
                {
                    "name": sdef.name,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "timezone": session_tz or "UTC",
                }
            )

        return {
            "symbol": symbol,
            "timestamp": now,
            "date": date,
            "flat": flat,
            "vwap": vwap_levels,
            "volumeProfile": vp_levels,
            "sessions": session_levels,
            "unit": "USDT",
            "dataQuality": {
                "developing": dev_cov,
                "previousDay": prev_cov,
            },
            "sessionWindows": session_windows,
            "sessionTZ": session_tz or "UTC",
            "qualityFlags": quality_flags,
            "profileParams": {
                "binSize": bin_sz,
                "valueAreaPct": float(va_pct if va_pct <= 1 else va_pct / 100.0),
                "mode": mode,
                "normalizeSeconds": int(normalize_seconds),
            },
        }

    async def get_session_profile(
        self,
        symbol: str,
        date: str | None = None,
        session: str = "all",
        interval: str = "15m",
        value_area_percent: float = 70.0,
        include_profile_levels: bool = False,
        max_profile_levels: int = 400,
        bin_size: float | None = None,
        mode: str = "raw",
        normalize_seconds: int = 3,
        window_start_ms: int | None = None,
        window_end_ms: int | None = None,
    ) -> dict[str, Any]:
        """Session profile(s): OHLC, Vol/Delta (quote), vPOC/vVAH/vVAL, and session VWAP.

        Key behaviors (requested):
        - When session="all": return **previous day** sessions (A/L/N/E) + **current day** sessions
          that have already started. Sessions that haven't started yet today are omitted.
        - Each session's metrics are calculated strictly within that session time window.
        - Adds `vwap` (session VWAP) per session.

        Data sources:
        - Profile (vPOC/VAH/VAL, buy/sell notional): local `footprint_1m` (best accuracy; needs backfill)
        - OHLC + fallback Vol/Delta/VWAP: Binance klines
        """
        symbol = symbol.upper()
        session = (session or "all").lower().strip()

        # Day start (UTC)
        if date:
            try:
                day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                day_start = int(day_dt.timestamp() * 1000)
            except ValueError as e:
                raise ValueError("date must be YYYY-MM-DD") from e
        else:
            day_start = get_day_start_ms(timestamp_ms())

        now = timestamp_ms()
        today_start = get_day_start_ms(now)
        prev_day_start = day_start - 86_400_000

        session_defs = self.settings.session_defs
        session_map = {s.name.lower(): s for s in session_defs}

        if session != "all" and session not in session_map:
            known = ", ".join([s.name for s in session_defs]) or "(none)"
            raise ValueError(f"Unknown session '{session}'. Known sessions: {known}, all")

        bin_sz = float(bin_size or self.settings.get_tpo_tick_size(symbol))
        # Visible Range override
        if window_start_ms is not None and window_end_ms is not None:
            start_ms = int(window_start_ms)
            end_ms = int(window_end_ms)
            if end_ms <= start_ms:
                raise ValueError("window_end_ms must be greater than window_start_ms")
            effective_end = min(end_ms, now)
            kline_data = []
            try:
                kline_data = await self.rest_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_ms,
                    end_time=effective_end,
                    limit=1500,
                )
            except Exception:
                kline_data = []
            k_quote = float(sum(k.quote_volume for k in kline_data)) if kline_data else 0.0
            k_base = float(sum(k.volume for k in kline_data)) if kline_data else 0.0
            k_delta = (
                float(sum((2.0 * k.taker_buy_quote_volume - k.quote_volume) for k in kline_data))
                if kline_data
                else 0.0
            )
            k_vwap = (k_quote / k_base) if k_base > 0 else None
            profile_res = await self.profile_engine.build_profile(
                symbol,
                window_start_ms=start_ms,
                window_end_ms=effective_end,
                bin_size=bin_sz,
                value_area_pct=value_area_percent,
                mode=mode,
                normalize_seconds=normalize_seconds,
                include_levels=include_profile_levels,
                top_n_levels=max_profile_levels if include_profile_levels else None,
                price_level_limit=max_profile_levels if include_profile_levels else None,
            )
            name = session if session != "all" else "custom"
            return {
                "symbol": symbol,
                "timestamp": now,
                "sessionTZ": "UTC",
                "date": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "sessions": {
                    name: {
                        "name": name,
                        "status": "window",
                        "range": {"startTime": start_ms, "endTime": end_ms, "effectiveEndTime": effective_end},
                        "ohlc": {
                            "open": kline_data[0].open if kline_data else None,
                            "high": max((k.high for k in kline_data), default=None),
                            "low": min((k.low for k in kline_data), default=None),
                            "close": kline_data[-1].close if kline_data else None,
                        },
                        "volQuote": (profile_res.get("totals") or {}).get("totalVolume") or k_quote,
                        "deltaQuote": k_delta,
                        "deltaPercent": (k_delta / k_quote * 100.0) if k_quote else 0.0,
                        "vwap": (k_quote / k_base) if k_base > 0 else k_vwap,
                        "sources": {
                            "totals": "footprint_1m" if (profile_res.get("totals") or {}).get("totalVolume") else "binance_klines",
                            "profile": "footprint_1m" if (profile_res.get("totals") or {}).get("totalVolume") else None,
                        },
                        "profile": {
                            "mode": mode,
                            "available": True,
                            "valueAreaPercent": value_area_percent,
                            "vPOC": (profile_res.get("valueArea") or {}).get("vPOC"),
                            "vVAH": (profile_res.get("valueArea") or {}).get("VAH"),
                            "vVAL": (profile_res.get("valueArea") or {}).get("VAL"),
                            "volumeQuote": (profile_res.get("totals") or {}).get("totalVolume"),
                            "levels": profile_res.get("levels") if include_profile_levels else None,
                            "binSize": bin_sz,
                            "degraded": (profile_res.get("dataQuality") or {}).get("coveragePct", 0) < self.coverage_threshold
                            and not (profile_res.get("dataQuality") or {}).get("dayIncomplete"),
                        },
                        "dataQuality": profile_res.get("dataQuality"),
                    }
                },
                "unit": "USDT",
                "sessionWindows": [{"name": name, "startTime": start_ms, "endTime": end_ms, "timezone": "UTC"}],
                "profileParams": {
                    "binSize": bin_sz,
                    "valueAreaPct": float(value_area_percent if value_area_percent <= 1 else value_area_percent / 100.0),
                    "mode": mode,
                    "normalizeSeconds": int(normalize_seconds),
                },
            }

        requested_defs = list(session_defs) if session == "all" else [session_map[session]]

        async def _build_one_day(day_start_ms: int) -> dict[str, Any]:
            """Build session metrics for one day_start.

            If day_start_ms is today's day start, omit sessions that have not started yet.
            """
            is_today_day = day_start_ms == today_start
            out: dict[str, Any] = {}

            for sdef in requested_defs:
                name = sdef.name
                s = sdef.time
                start_ms = day_start_ms + int(s.start_minutes) * 60_000
                end_ms = day_start_ms + int(s.end_minutes) * 60_000
                if s.end_minutes <= s.start_minutes:
                    # Session crosses midnight
                    end_ms += 86_400_000

                # Determine session status/effective end
                status = "completed"
                effective_end = end_ms
                if is_today_day:
                    if now < start_ms:
                        # Requested: don't return sessions that haven't started yet.
                        continue
                    if start_ms <= now < end_ms:
                        status = "developing"
                        effective_end = now
                    else:
                        status = "completed"
                        effective_end = end_ms

                if effective_end <= start_ms:
                    continue

                # -----------------------------
                # OHLC + kline-based totals
                # -----------------------------
                totals_error: str | None = None
                klines = []
                try:
                    klines = await self.rest_client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_ms,
                        end_time=effective_end,
                        limit=1500,
                    )
                except Exception as e:
                    totals_error = str(e)
                    klines = []

                k_quote = float(sum(k.quote_volume for k in klines)) if klines else 0.0
                k_base = float(sum(k.volume for k in klines)) if klines else 0.0
                k_delta = (
                    float(sum((2.0 * k.taker_buy_quote_volume - k.quote_volume) for k in klines))
                    if klines
                    else 0.0
                )
                k_vwap = (k_quote / k_base) if k_base > 0 else None

                o = klines[0].open if klines else None
                c = klines[-1].close if klines else None
                hi = max((k.high for k in klines), default=None)
                lo = min((k.low for k in klines), default=None)

                # ACR + RF are kept for compatibility, but they are not core to Exocharts session profile.
                acr = None
                rf = None
                if klines:
                    ranges = [(k.high - k.low) for k in klines]
                    acr = (sum(ranges) / len(ranges)) if ranges else None

                    # Rotation factor: count direction flips between up/down candles
                    dirs: list[int] = []
                    for k in klines:
                        if k.close > k.open:
                            dirs.append(1)
                        elif k.close < k.open:
                            dirs.append(-1)
                        else:
                            dirs.append(0)
                    prev_dir = None
                    flips = 0
                    for d in dirs:
                        if d == 0:
                            continue
                        if prev_dir is None:
                            prev_dir = d
                            continue
                        if d != prev_dir:
                            flips += 1
                            prev_dir = d
                    rf = flips

                # -----------------------------
                # Unified profile engine (VR/Session-aligned)
                # -----------------------------
                profile_result = await self.profile_engine.build_profile(
                    symbol,
                    window_start_ms=start_ms,
                    window_end_ms=effective_end,
                    bin_size=bin_sz,
                    value_area_pct=value_area_percent,
                    mode=mode,
                    normalize_seconds=normalize_seconds,
                    include_levels=include_profile_levels,
                    top_n_levels=max_profile_levels if include_profile_levels else None,
                    price_level_limit=max_profile_levels if include_profile_levels else None,
                )

                profile_levels = profile_result.get("levels") if include_profile_levels else None
                dq = profile_result.get("dataQuality") or {}

                # -----------------------------
                # Quote-based profile (footprint preferred, klines fallback)
                # -----------------------------
                fp_rows = await self.storage.get_profile_range(symbol, start_ms, effective_end)
                quote_rows = fp_rows
                quote_rows_source = "footprint_1m"
                if not quote_rows and klines:
                    quote_rows = [
                        {
                            "price": float(k.close if k.close is not None else k.high),
                            "quote_volume": float(k.quote_volume),
                            "taker_buy_quote_volume": float(k.taker_buy_quote_volume),
                            "trade_count": int(getattr(k, "trade_count", 0) or 0),
                        }
                        for k in klines
                    ]
                    quote_rows_source = "binance_klines"

                quote_profile = build_quote_profile(
                    quote_rows,
                    bin_size=bin_sz,
                    value_area_percent=value_area_percent,
                    normalize_seconds=normalize_seconds,
                    mode=mode,
                )

                quality_flags = list(quote_profile.get("quality_flags") or [])
                dq_degraded = dq.get("coveragePct", 0) < self.coverage_threshold and not dq.get("dayIncomplete")
                if dq_degraded and "low_coverage" not in quality_flags:
                    quality_flags.append("low_coverage")

                vol_quote = quote_profile.get("volumeQuote") or k_quote
                base_volume = float(quote_profile.get("baseVolume") or 0.0)
                buy_quote = quote_profile.get("buyQuote") or 0.0
                sell_quote = quote_profile.get("sellQuote") or 0.0
                delta_quote = quote_profile.get("deltaQuote", k_delta)
                profile_available = bool(quote_profile.get("histogram"))
                vwap = None
                if profile_available and base_volume > 0:
                    vwap = vol_quote / base_volume
                elif k_base > 0:
                    vwap = k_quote / k_base
                else:
                    vwap = k_vwap
                totals_source = quote_rows_source if profile_available else "binance_klines"

                out[name] = {
                    "name": name,
                    "status": status,
                    "range": {
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "effectiveEndTime": effective_end,
                    },
                    "ohlc": {"open": o, "high": hi, "low": lo, "close": c},
                    "volQuote": vol_quote,
                    "deltaQuote": delta_quote,
                    "deltaPercent": (delta_quote / vol_quote * 100.0) if vol_quote else 0.0,
                    "vwap": vwap,
                    "sources": {
                        "totals": totals_source,
                        "ohlc": "binance_klines" if klines else None,
                        "profile": "footprint_1m" if profile_available else None,
                        "errors": {"klines": totals_error},
                    },
                    "profile": {
                        "mode": "quote_usdt",
                        "available": profile_available,
                        "valueAreaPercent": float(value_area_percent),
                        "vPOC": quote_profile.get("vPOC"),
                        "vVAH": quote_profile.get("vVAH"),
                        "vVAL": quote_profile.get("vVAL"),
                        "high": quote_profile.get("high"),
                        "low": quote_profile.get("low"),
                        "volumeQuote": quote_profile.get("volumeQuote"),
                        "buyQuote": buy_quote,
                        "sellQuote": sell_quote,
                        "deltaQuote": delta_quote,
                        "tradeCount": quote_profile.get("tradeCount"),
                        "levels": profile_levels,
                        "degraded": bool(quality_flags),
                        "qualityFlags": quality_flags,
                        "binSize": bin_sz,
                    },
                    "extra": {"acr": acr, "rf": rf, "klineVWAP": k_vwap},
                    "dataQuality": dq,
                }

            return out

        prev_sessions = await _build_one_day(prev_day_start)
        cur_sessions = await _build_one_day(day_start)

        def _fmt_date(day_start_ms: int) -> str:
            return datetime.fromtimestamp(day_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        cur_date = _fmt_date(day_start)
        pd_date = _fmt_date(prev_day_start)
        session_windows = []
        for sdef in session_defs:
            start_ms = day_start + int(sdef.start_minutes) * 60_000
            end_ms = day_start + int(sdef.end_minutes) * 60_000
            if sdef.end_minutes <= sdef.start_minutes:
                end_ms += 86_400_000
            session_windows.append(
                {
                    "name": sdef.name,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "timezone": "UTC",
                }
            )

        return {
            "symbol": symbol,
            "timestamp": now,
            "sessionTZ": "UTC",
            # Convenience aliases (flat dictionaries) – common in Exocharts naming:
            "date": cur_date,
            "sessions": cur_sessions,
            "pdDate": pd_date,
            "pdSessions": prev_sessions,
            # Rich metadata:
            "currentDay": {"date": cur_date, "dayStart": day_start, "sessions": cur_sessions},
            "previousDay": {"date": pd_date, "dayStart": prev_day_start, "sessions": prev_sessions},
            "unit": "USDT",
            "sessionWindows": session_windows,
            "notes": [
                "When session='all', currentDay omits sessions that have not started yet (requested).",
                "Profile(vPOC/vVAH/vVAL, buy/sell notional) uses local footprint_1m when available; otherwise totals fall back to Binance klines.",
                "Session VWAP is computed as (sum quote_volume) / (sum base_volume) over the same window (or from footprint when available).",
            ],
        }

    async def get_swing_liquidity(
        self,
        symbol: str,
        interval: str = "15m",
        lookback_bars: int = 300,
        pivot_left: int = 10,
        pivot_right: int = 15,
        active_only: bool = False,
        max_levels: int = 150,
    ) -> dict[str, Any]:
        """Detect swing highs/lows and output liquidity levels.

        This approximates TradingView-style "Swing Liquidity":
        - Pivot swing highs => buy-side liquidity (stops above highs)
        - Pivot swing lows  => sell-side liquidity (stops below lows)

        Data source: Binance klines (high/low + quote volume + taker buy quote -> delta)

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (default 15m)
            lookback_bars: Number of recent bars to scan (max 1500)
            pivot_left: Pivot left strength (bars)
            pivot_right: Pivot right strength (bars)
            active_only: Return only unswept levels
            max_levels: Cap returned levels
        """
        symbol = symbol.upper()
        lookback = int(max(50, min(1500, lookback_bars)))
        pivot_left = int(max(1, pivot_left))
        pivot_right = int(max(1, pivot_right))
        max_levels = int(max(10, max_levels))

        tf_ms = get_timeframe_ms(interval)

        end_time = timestamp_ms()
        start_time = end_time - lookback * tf_ms

        klines = await self.rest_client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=lookback,
        )

        # Ensure we only keep the last `lookback` bars
        if len(klines) > lookback:
            klines = klines[-lookback:]

        n = len(klines)
        if n < pivot_left + pivot_right + 5:
            return {
                "symbol": symbol,
                "interval": interval,
                "timestamp": timestamp_ms(),
                "levels": [],
                "error": "Not enough kline bars for the requested pivot settings",
            }

        highs = [k.high for k in klines]
        lows = [k.low for k in klines]

        pivots: list[dict[str, Any]] = []

        for i in range(pivot_left, n - pivot_right):
            win_h = highs[i - pivot_left : i + pivot_right + 1]
            win_l = lows[i - pivot_left : i + pivot_right + 1]

            h = highs[i]
            l = lows[i]

            is_pivot_high = (h == max(win_h)) and (win_h.count(h) == 1)
            is_pivot_low = (l == min(win_l)) and (win_l.count(l) == 1)

            if not (is_pivot_high or is_pivot_low):
                continue

            k = klines[i]
            vol_q = float(k.quote_volume)
            delta_q = float(2.0 * k.taker_buy_quote_volume - k.quote_volume)
            delta_pct = (delta_q / vol_q * 100.0) if vol_q else 0.0

            if is_pivot_high:
                side = "buy_side"
                level_price = float(h)
                level_type = "swing_high"
            else:
                side = "sell_side"
                level_price = float(l)
                level_type = "swing_low"

            # Determine if / when the level got swept
            swept = False
            swept_time = None

            for j in range(i + 1, n):
                if side == "buy_side":
                    if klines[j].high >= level_price:
                        swept = True
                        swept_time = klines[j].open_time
                        break
                else:
                    if klines[j].low <= level_price:
                        swept = True
                        swept_time = klines[j].open_time
                        break

            pivots.append(
                {
                    "type": level_type,
                    "side": side,
                    "price": level_price,
                    "pivotTime": k.open_time,
                    "pivotIndex": i,
                    "volumeQuote": vol_q,
                    "deltaQuote": delta_q,
                    "deltaPercent": delta_pct,
                    "swept": swept,
                    "sweptTime": swept_time,
                }
            )

        if active_only:
            pivots = [p for p in pivots if not p["swept"]]

        # Sort newest first and cap
        pivots.sort(key=lambda x: x["pivotTime"], reverse=True)
        pivots = pivots[:max_levels]

        # Basic nearest levels around current price
        current_price = float(klines[-1].close)
        above = sorted([p for p in pivots if p["price"] > current_price], key=lambda x: x["price"])
        below = sorted([p for p in pivots if p["price"] < current_price], key=lambda x: x["price"], reverse=True)

        return {
            "symbol": symbol,
            "interval": interval,
            "timestamp": timestamp_ms(),
            "range": {"startTime": start_time, "endTime": end_time},
            "params": {
                "lookbackBars": lookback,
                "pivotLeft": pivot_left,
                "pivotRight": pivot_right,
                "activeOnly": active_only,
                "maxLevels": max_levels,
            },
            "currentPrice": current_price,
            "nearestAbove": above[0] if above else None,
            "nearestBelow": below[0] if below else None,
            "levels": pivots,
            "unit": "USDT",
            "deltaDefinition": "deltaQuote = takerBuyQuote - takerSellQuote = 2*takerBuyQuote - totalQuote",
        }

    async def get_footprint(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        timeframe: str = "30m",
        view: str = "statistics",
        max_levels_per_bar: int = 200,
        *,
        bin_size: float | None = None,
        tick_size: float | None = None,
        price_level_limit: int | None = None,
        top_n_levels: int | None = None,
        compress: bool = True,
        cursor: int | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get footprint information.

        By default this returns **compact statistics** (like Exocharts "Footprint bar statistics")
        to avoid blowing up LLM contexts.

        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp in ms (optional)
            end_time: End timestamp in ms (optional)
            timeframe: Timeframe (default '30m')
            view: 'statistics' (default) or 'levels'
            max_levels_per_bar: When view='levels', caps the number of price levels returned per bar.
        """
        end_time = end_time or timestamp_ms()

        # Default window: last 24h for 30m stats, otherwise last 2h
        if start_time is None:
            if timeframe == "30m" and view == "statistics":
                start_time = end_time - 24 * 60 * 60 * 1000
            else:
                start_time = end_time - 2 * 60 * 60 * 1000

        if view.lower() != "levels":
            return await self.get_footprint_statistics(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                timeframe=timeframe,
            )

        # Full footprint levels (can be large)
        cov_raw = await self.storage.get_footprint_coverage(symbol, start_time, end_time)
        cov_pct = float(
            (cov_raw.get("minute_buckets") or 0)
            / max(1, int((end_time - start_time) / 60_000))
        )
        if cov_pct < self.coverage_threshold:
            await self._maybe_backfill_footprint(symbol, start_time, end_time, coverage_pct=cov_pct)
            cov_raw = await self.storage.get_footprint_coverage(symbol, start_time, end_time)

        data = await self.footprint.get_footprint_range(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )

        bin_sz = None
        if bin_size is not None:
            bin_sz = float(bin_size)
        elif tick_size is not None:
            bin_sz = float(tick_size)

        price_cap = price_level_limit or max_levels_per_bar
        bars_out: list[dict[str, Any]] = []
        truncated_any = False

        for bar in data:
            levels = list(bar.get("levels") or [])
            if bin_sz and bin_sz > 0:
                agg: dict[float, dict[str, float]] = {}
                for lvl in levels:
                    price = float(lvl.get("price"))
                    bucket = (int(price // bin_sz)) * bin_sz
                    if bucket not in agg:
                        agg[bucket] = {
                            "buy": 0.0,
                            "sell": 0.0,
                            "trades": 0,
                        }
                    agg[bucket]["buy"] += float(lvl.get("buyVolume", 0.0))
                    agg[bucket]["sell"] += float(lvl.get("sellVolume", 0.0))
                    agg[bucket]["trades"] += int(lvl.get("tradeCount", 0))
                levels = [
                    {
                        "price": p,
                        "buyVolume": v["buy"],
                        "sellVolume": v["sell"],
                        "tradeCount": v["trades"],
                        "delta": v["buy"] - v["sell"],
                        "totalVolume": v["buy"] + v["sell"],
                    }
                    for p, v in sorted(agg.items())
                ]

            if compress:
                total_levels = len(levels)
                if top_n_levels is not None and total_levels > top_n_levels:
                    levels = sorted(levels, key=lambda l: l.get("totalVolume", 0.0), reverse=True)[: int(top_n_levels)]
                if price_cap and len(levels) > price_cap:
                    levels = levels[:price_cap]
                dropped = total_levels - len(levels)
                bar["levelsDropped"] = max(0, dropped)
                truncated_any = truncated_any or dropped > 0
            else:
                if price_cap and len(levels) > price_cap:
                    levels = levels[:price_cap]
                    bar["levels_truncated"] = True
                    truncated_any = True

            bar["levels"] = levels
            bars_out.append(bar)

        start_idx = int(cursor or 0)
        if start_idx < 0:
            start_idx = 0
        end_idx = start_idx + limit if limit is not None else None
        paged = bars_out[start_idx:end_idx]
        next_cursor = None
        if end_idx is not None and end_idx < len(bars_out):
            next_cursor = end_idx

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "startTime": start_time,
            "endTime": end_time,
            "view": "levels",
            "levelsTruncated": truncated_any,
            "cursor": start_idx,
            "nextCursor": next_cursor,
            "bars": paged,
            "binSize": bin_sz,
            "topNLevels": top_n_levels,
            "levelLimit": price_cap,
            "dataQuality": self._build_data_quality(
                window_start_ms=start_time,
                window_end_ms=end_time,
                coverage_raw=cov_raw,
                bars_returned=len(bars_out),
                levels_count=sum(len((b.get("levels") or [])) for b in bars_out),
                source="local",
            ),
        }

    async def get_footprint_statistics(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        timeframe: str = "30m",
    ) -> dict[str, Any]:
        """Return compact footprint statistics (quote-denominated).

        Output is designed to match the *shape* of Exocharts' "Footprint bar statistics":
        - Vol (quote)
        - Delta (quote)
        - Delta Max / Min (max/min per-price delta inside the bar, quote)
        - Trades (count)
        """
        end_time = end_time or timestamp_ms()
        start_time = start_time or (end_time - 24 * 60 * 60 * 1000)

        tf_ms_map = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        bucket_ms = tf_ms_map.get(timeframe)
        if bucket_ms is None:
            raise ValueError(f"Unsupported timeframe for statistics: {timeframe}")

        # Coverage check & optional on-demand backfill
        cov_raw = await self.storage.get_footprint_coverage(symbol, start_time, end_time)
        cov_pct = float(
            (cov_raw.get("minute_buckets") or 0)
            / max(1, int((end_time - start_time) / 60_000))
        )
        if cov_pct < self.coverage_threshold:
            await self._maybe_backfill_footprint(symbol, start_time, end_time, coverage_pct=cov_pct)
            cov_raw = await self.storage.get_footprint_coverage(symbol, start_time, end_time)

        rows = await self.storage.get_footprint_statistics(symbol, start_time, end_time, bucket_ms=bucket_ms)

        bars: list[dict[str, Any]] = []
        for r in rows:
            bs = int(r["bucket_start"])
            bars.append(
                {
                    "startTime": bs,
                    "endTime": bs + bucket_ms,
                    "volQuote": float(r["vol_quote"] or 0.0),
                    "deltaQuote": float(r["delta_quote"] or 0.0),
                    "deltaMaxQuote": float(r["delta_max_quote"] or 0.0),
                    "deltaMinQuote": float(r["delta_min_quote"] or 0.0),
                    "buyQuote": float(r["buy_quote"] or 0.0),
                    "sellQuote": float(r["sell_quote"] or 0.0),
                    "trades": int(r["trades"] or 0),
                    "rektLongQuote": None,
                    "rektShortQuote": None,
                    "rektTotalQuote": None,
                }
            )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "view": "statistics",
            "startTime": start_time,
            "endTime": end_time,
            "bars": bars,
            "dataQuality": self._build_data_quality(
                window_start_ms=start_time,
                window_end_ms=end_time,
                coverage_raw=cov_raw,
                bars_returned=len(bars),
                levels_count=None,
                source="local",
            ),
        }

    async def get_orderflow_metrics(
        self,
        symbol: str,
        timeframe: str,
        start_time: int,
        end_time: int,
    ) -> dict[str, Any]:
        """Get orderflow metrics including delta, CVD, imbalances.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            Orderflow metrics with delta sequence, CVD, imbalances. The
            ``currentCVD`` field reflects the last value of ``cvdSequence``.
        """
        symbol = symbol.upper()
        self.logger.info("get_orderflow_metrics", symbol=symbol, timeframe=timeframe)
        
        # Get delta/CVD data
        delta_data = await self.delta_cvd.get_delta_range(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )
        cvd_sequence = delta_data.get("cvdSequence", [])
        current_cvd = cvd_sequence[-1]["cvd"] if cvd_sequence else delta_data.get("currentCVD", 0)

        cov_raw = await self.storage.get_footprint_coverage(symbol, start_time, end_time)
        dq = self._build_data_quality(
            window_start_ms=start_time,
            window_end_ms=end_time,
            coverage_raw=cov_raw,
            bars_returned=delta_data.get("summary", {}).get("barCount"),
            levels_count=len(cvd_sequence) if cvd_sequence else None,
            source="local",
        )

        # Get footprint bars for imbalance analysis
        footprint_bars = await self.footprint.get_footprint_range(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )
        
        # Analyze imbalances in latest bar
        imbalance_analysis = None
        if footprint_bars:
            # Create footprint bar object from dict
            from src.indicators.footprint import FootprintBar, FootprintLevel
            
            latest = footprint_bars[-1]
            bar = FootprintBar(
                symbol=latest["symbol"],
                timeframe=latest["timeframe"],
                timestamp=latest["timestamp"],
            )
            for level_data in latest.get("levels", []):
                bar.levels[level_data["price"]] = FootprintLevel(
                    price=level_data["price"],
                    buy_volume=level_data["buyVolume"],
                    sell_volume=level_data["sellVolume"],
                    trade_count=level_data["tradeCount"],
                )
            
            imbalance_analysis = self.imbalance.analyze_footprint(bar)
        
        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timeframe": timeframe,
            "startTime": start_time,
            "endTime": end_time,
            "timestamp": timestamp_ms(),
            "delta": delta_data.get("summary", {}),
            "deltaSequence": delta_data.get("deltaSequence", []),
            "cvdSequence": cvd_sequence,
            "currentCVD": current_cvd,
            "imbalances": imbalance_analysis,
            "volumeUnit": symbol.replace("USDT", ""),
            "dataQuality": dq,
        }
    
    async def get_orderbook_depth_delta(
        self,
        symbol: str,
        percent: float = 1.0,
        window_sec: int = 60,
        lookback: int = 3600,
        max_points: int = 30,
    ) -> dict[str, Any]:
        """Get orderbook depth delta over time.
        
        Args:
            symbol: Trading pair symbol
            percent: Price range percentage from mid (default 1%)
            window_sec: Snapshot interval in seconds
            lookback: Lookback period in seconds
        
        Returns:
            Depth delta time series with bid/ask volumes
        """
        symbol = symbol.upper()
        self.logger.info("get_orderbook_depth_delta", symbol=symbol, percent=percent)
        
        # NOTE: window_sec is used as an output aggregation interval (granularity).
        # The snapshot collection interval is controlled by ORDERBOOK_UPDATE_INTERVAL_SEC.
        depth_data = await self.depth_delta.get_depth_delta_series(
            symbol=symbol,
            percent_range=percent,
            lookback_seconds=lookback,
            max_points=max_points,
            granularity_sec=max(0, int(window_sec)),
        )
        
        # Add current summary
        current_summary = self.depth_delta.get_depth_summary(symbol)
        if current_summary:
            depth_data["current"] = current_summary
        
        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "percentRange": percent,
            "windowSec": window_sec,
            "lookbackSec": lookback,
            "maxPoints": max_points,
            **depth_data,
            "volumeUnit": symbol.replace("USDT", ""),
        }

    async def get_orderbook_heatmap(
        self,
        symbol: str,
        lookback_minutes: int = 180,
        max_levels: int = 15,
    ) -> dict[str, Any]:
        """Get a compact summary of the stored orderbook heatmap.

        This tool is intentionally *summary-first* to avoid blowing up MCP / LLM
        context windows. For a full matrix, an external client can query the
        database directly or you can extend this tool with a dedicated export
        format.

        Args:
            symbol: Trading pair symbol
            lookback_minutes: Lookback window for coverage metadata
            max_levels: How many top bid/ask bins to return

        Returns:
            Summary (top bid/ask liquidity bins) + coverage metadata
        """
        symbol = symbol.upper()
        lookback_minutes = int(
            lookback_minutes
            if lookback_minutes is not None
            else getattr(self.settings, "heatmap_lookback_minutes", 180)
        )

        if not getattr(self.settings, "heatmap_enabled", False) or self.heatmap is None:
            return {
                "symbol": symbol,
                "enabled": False,
                "message": "Heatmap is disabled. Set HEATMAP_ENABLED=true to enable sampling.",
                "howToEnable": "Set HEATMAP_ENABLED=true and restart the MCP server.",
                "remediation": "Enable HEATMAP_ENABLED and allow sampler to warm up for the requested lookback window.",
                "dataQuality": {
                    "isEnabled": False,
                    "degraded": True,
                    "reasons": ["heatmap_disabled"],
                },
            }

        now_ms = timestamp_ms()
        start_ms = now_ms - int(lookback_minutes * 60_000)

        latest_snapshot = await self.storage.get_latest_orderbook_heatmap_snapshot(symbol)
        if isinstance(latest_snapshot, tuple) and len(latest_snapshot) == 2:
            latest_ts, latest_rows = latest_snapshot
        else:
            latest_ts, latest_rows = None, latest_snapshot or []
        if not latest_rows:
            warmup = {
                "expectedMinutes": lookback_minutes,
                "note": "Allow one full lookback window for the sampler to accumulate snapshots.",
            }
            coverage = {
                "lookbackMinutes": lookback_minutes,
                "startTime": start_ms,
                "endTime": now_ms,
                "uniqueSnapshots": 0,
                "latestTimestamp": latest_ts,
                "binSize": float(getattr(self.settings, "heatmap_bin_ticks", 10.0)),
                "percentRange": float(getattr(self.settings, "heatmap_depth_percent", 1.0)),
            }
            return {
                "symbol": symbol,
                "enabled": True,
                "message": "No heatmap data yet. Wait for snapshots to accumulate.",
                "warmup": warmup,
                "coverage": coverage,
                "dataQuality": {
                    "isEnabled": True,
                    "degraded": True,
                    "reasons": ["no_snapshots"],
                    "coverageMinutes": 0,
                },
            }

        # Current mid price from in-memory book (best-effort)
        mid = None
        book = self.orderbook.get_orderbook(symbol)
        if book:
            mid = book.mid_price

        # Rank bins by bid/ask liquidity
        top_bids = sorted(latest_rows, key=lambda r: float(r.get("bid_volume", 0.0)), reverse=True)[: max_levels]
        top_asks = sorted(latest_rows, key=lambda r: float(r.get("ask_volume", 0.0)), reverse=True)[: max_levels]

        # Basic coverage info (how many unique timestamps in lookback)
        cached_meta = self.cache.get_heatmap_metadata(symbol)
        meta_matches = cached_meta and int(cached_meta.get("lookbackMinutes", 0)) == lookback_minutes
        coverage_meta = cached_meta if meta_matches else None
        if coverage_meta is None:
            coverage_meta = await self.storage.get_orderbook_heatmap_coverage(symbol, start_ms, now_ms)
            coverage_meta = {
                "uniqueSnapshots": coverage_meta.get("uniqueSnapshots", 0),
                "latestTimestamp": coverage_meta.get("latestTimestamp"),
                "lookbackMinutes": lookback_minutes,
                "sampledAtMs": now_ms,
            }
            try:
                self.cache.set_heatmap_metadata(symbol, coverage_meta)
            except Exception:
                pass

        latest_timestamp = latest_ts if latest_ts is not None else coverage_meta.get("latestTimestamp")
        coverage = {
            "lookbackMinutes": lookback_minutes,
            "startTime": start_ms,
            "endTime": now_ms,
            "uniqueSnapshots": int(coverage_meta.get("uniqueSnapshots", 0)),
            "latestTimestamp": latest_timestamp,
            "binSize": float(getattr(self.settings, "heatmap_bin_ticks", 10.0)),
            "percentRange": float(getattr(self.settings, "heatmap_depth_percent", 1.0)),
        }
        is_stale = False
        if coverage["latestTimestamp"] is not None:
            is_stale = (now_ms - coverage["latestTimestamp"]) > max(2, int(self.settings.heatmap_interval_sec * 3)) * 1000
        dq = {
            "isEnabled": True,
            "degraded": coverage["uniqueSnapshots"] == 0 or is_stale,
            "reasons": (
                ["stale_snapshots"] if is_stale else []
            ) + (["no_snapshots"] if coverage["uniqueSnapshots"] == 0 else []),
            "lastUpdateMs": coverage["latestTimestamp"],
            "coverageMinutes": lookback_minutes,
        }

        warmup = {
            "expectedMinutes": lookback_minutes,
            "note": "Heatmap snapshots accumulate continuously; allow one full lookback window after enabling.",
        }

        return {
            "symbol": symbol,
            "enabled": True,
            "midPrice": mid,
            "coverage": coverage,
            "dataQuality": dq,
            "warmup": warmup,
            "remediation": (
                "Wait for new snapshots or reduce lookbackMinutes to fit collected window."
                if dq["degraded"]
                else None
            ),
            "topBidBins": [
                {
                    "priceBin": float(r["price_bin"]),
                    "bidVolume": float(r.get("bid_volume", 0.0)),
                }
                for r in top_bids
            ],
            "topAskBins": [
                {
                    "priceBin": float(r["price_bin"]),
                    "askVolume": float(r.get("ask_volume", 0.0)),
                }
                for r in top_asks
            ],
        }
    
    async def stream_liquidations(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get recent liquidation events.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of liquidations to return
        
        Returns:
            Recent liquidation events
        """
        symbol = symbol.upper()
        self.logger.info("stream_liquidations", symbol=symbol, limit=limit)
        
        liquidations = self.cache.get_liquidations(symbol, limit)
        
        # Convert to dict format
        liq_list = [
            {
                "timestamp": liq.timestamp,
                "symbol": liq.symbol,
                "side": liq.side,
                "price": liq.price,
                "avgPrice": liq.avg_price,
                "originalQty": liq.original_qty,
                "filledQty": liq.filled_qty,
                "notional": liq.notional,
                "isLongLiquidation": liq.is_long_liquidation,
                "orderStatus": liq.order_status,
            }
            for liq in liquidations
        ]
        
        # Calculate statistics
        long_liqs = [l for l in liq_list if l["isLongLiquidation"]]
        short_liqs = [l for l in liq_list if not l["isLongLiquidation"]]
        
        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timestamp": timestamp_ms(),
            "count": len(liq_list),
            "statistics": {
                "longLiquidations": len(long_liqs),
                "shortLiquidations": len(short_liqs),
                "totalLongNotional": sum(l["notional"] for l in long_liqs),
                "totalShortNotional": sum(l["notional"] for l in short_liqs),
            },
            "liquidations": liq_list,
            "notionalUnit": "USDT",
            "volumeUnit": symbol.replace("USDT", ""),
        }
    
    async def get_open_interest(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get open interest data including history.
        
        Args:
            symbol: Trading pair symbol
            period: Historical period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of historical records
        
        Returns:
            Current OI and historical data
        """
        symbol = symbol.upper()
        self.logger.info("get_open_interest", symbol=symbol, period=period)
        
        # Get current OI from cache
        cache = self.cache.get_cache(symbol)
        current_oi = cache.open_interest
        current_oi_notional = cache.open_interest_notional
        
        # Get historical OI
        try:
            oi_history = await self.rest_client.get_open_interest_hist(
                symbol=symbol,
                period=period,
                limit=limit,
            )
            
            history_list = [
                {
                    "timestamp": h.timestamp,
                    "openInterest": h.sum_open_interest,
                    "openInterestNotional": h.sum_open_interest_value,
                }
                for h in oi_history
            ]
            
            # Calculate OI delta
            if len(history_list) >= 2:
                oi_delta = history_list[-1]["openInterest"] - history_list[-2]["openInterest"]
                oi_delta_notional = history_list[-1]["openInterestNotional"] - history_list[-2]["openInterestNotional"]
            else:
                oi_delta = 0
                oi_delta_notional = 0
            
        except Exception as e:
            self.logger.error("get_oi_history_failed", error=str(e))
            history_list = []
            oi_delta = 0
            oi_delta_notional = 0
        
        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timestamp": timestamp_ms(),
            "current": {
                "openInterest": current_oi,
                "openInterestNotional": current_oi_notional,
            },
            "delta": {
                "period": period,
                "openInterestDelta": oi_delta,
                "openInterestDeltaNotional": oi_delta_notional,
            },
            "history": history_list,
            "oiUnit": symbol.replace("USDT", ""),
            "notionalUnit": "USDT",
        }
    
    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Get current and historical funding rate.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Current funding rate and next funding time
        """
        symbol = symbol.upper()
        self.logger.info("get_funding_rate", symbol=symbol)
        
        # Get from cache
        cache = self.cache.get_cache(symbol)
        
        # Get historical funding rates
        try:
            funding_history = await self.rest_client.get_funding_rate(symbol, limit=10)
            history_list = [
                {
                    "fundingTime": f.funding_time,
                    "fundingRate": f.funding_rate,
                }
                for f in funding_history
            ]
        except Exception as e:
            self.logger.error("get_funding_history_failed", error=str(e))
            history_list = []
        
        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timestamp": timestamp_ms(),
            "current": {
                "fundingRate": cache.funding_rate,
                "fundingRatePercent": cache.funding_rate * 100,
                "nextFundingTime": cache.next_funding_time,
            },
            "history": history_list,
        }


    async def get_tpo_profile(
        self,
        symbol: str,
        date: str | None = None,
        session: str = "all",
        period_minutes: int = 30,
        tick_size: float | None = None,
        value_area_percent: float = 70.0,
        use_volume_for_va: bool | None = None,
        include_levels: bool = False,
        max_levels: int = 240,
        ib_periods: int = 2,
        include_period_profiles: bool = True,
        include_single_prints: bool = True,
        single_prints_mode: str = "compact",
        tail_min_len: int = 2,
    ) -> dict[str, Any]:
        """Get a TPO (Time-Price-Opportunity) profile for a UTC day/session.

        Implementation notes:
        - Distribution source: local `footprint_1m` (accurate, but requires backfill for history)
        - Period stats (open/high/low/close, rotation factor, IB range): Binance klines

        Args:
            symbol: e.g. BTCUSDT
            date: UTC date YYYY-MM-DD (defaults to today)
            session: session name (case-insensitive) defined by `SESSIONS`, or 'all'
            period_minutes: TPO period (letters). Exocharts default is typically 30m.
            tick_size: optional price bucket size (e.g. 50 for daily BTC TPO)
            value_area_percent: VA% for POC/VAH/VAL (default 70)
            use_volume_for_va: compute VA/POC from volume instead of time (Exocharts option)
            include_levels: include per-price distribution (capped by max_levels)
            max_levels: cap returned levels (only when include_levels=True)
            ib_periods: number of periods used for Initial Balance (default 2 periods)
            single_prints_mode:
                - "compact" (default): return single prints as compressed price ranges + small tail lists
                - "full": return full single print level arrays (can be very large)

        Returns:
            Dict with TPO profile + stats.
        """
        symbol = symbol.upper().strip()
        session = (session or "all").lower().strip()

        # Day start (UTC)
        if date:
            try:
                day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                day_start = int(day_dt.timestamp() * 1000)
            except ValueError as e:
                raise ValueError("date must be YYYY-MM-DD") from e
        else:
            day_start = get_day_start_ms(timestamp_ms())

        now = timestamp_ms()
        is_today = day_start == get_day_start_ms(now)

        # Sessions are configurable in env (`SESSIONS`).
        session_defs = self.settings.session_defs
        session_map = {s.name.lower(): s for s in session_defs}
        if session != "all" and session not in session_map:
            choices = ", ".join([s.name for s in session_defs] + ["all"])
            raise ValueError(f"Unknown session '{session}'. Use one of: {choices}")

        if session == "all":
            start_ms = day_start
            end_ms = day_start + 86_400_000
        else:
            s = session_map[session].time
            start_ms = day_start + int(s.start_minutes) * 60_000
            end_ms = day_start + int(s.end_minutes) * 60_000
            if s.end_minutes <= s.start_minutes:
                end_ms += 86_400_000  # crosses midnight

        status = "completed"
        effective_end = end_ms
        if is_today:
            if now < start_ms:
                status = "not_started"
                effective_end = start_ms
            elif start_ms <= now < end_ms:
                status = "developing"
                effective_end = now
            else:
                status = "completed"
                effective_end = end_ms

        # If caller didn't specify, follow env default (matches Exocharts option)
        if use_volume_for_va is None:
            use_volume_for_va = bool(self.settings.tpo_use_volume_for_va_default)

        # Default tick size to the configured symbol tick size if not provided
        if tick_size is None:
            # TPO profiles typically use a much coarser price step than footprints.
            # Use the dedicated TPO tick size defaults from env.
            tick_size = self._get_tpo_tick_size_safe(symbol)

        cov_raw = await self.storage.get_footprint_coverage(symbol, start_ms, effective_end)
        cov_pct = float(
            (cov_raw.get("minute_buckets") or 0)
            / max(1, int((effective_end - start_ms) / 60_000))
        )
        if cov_pct < self.coverage_threshold:
            await self._maybe_backfill_footprint(symbol, start_ms, effective_end, coverage_pct=cov_pct)
            cov_raw = await self.storage.get_footprint_coverage(symbol, start_ms, effective_end)

        # Build TPO profile from footprint (up to effective_end for developing sessions)
        tpo_profile = await self.tpo_profile.build_profile(
            symbol=symbol,
            start_time=start_ms,
            end_time=effective_end,
            period_minutes=int(period_minutes),
            tick_size=float(tick_size) if tick_size is not None else None,
            value_area_percent=float(value_area_percent),
            use_volume_for_va=bool(use_volume_for_va),
            include_levels=bool(include_levels),
            max_levels=int(max_levels),
            include_period_profiles=bool(include_period_profiles),
            include_single_prints=bool(include_single_prints),
            tail_min_len=int(tail_min_len),
        )

        # Period stats from klines
        def _minutes_to_interval(m: int) -> str | None:
            # Binance supported intervals for klines.
            mapping = {
                1: "1m",
                3: "3m",
                5: "5m",
                15: "15m",
                30: "30m",
                60: "1h",
                120: "2h",
                240: "4h",
                360: "6h",
                480: "8h",
                720: "12h",
                1440: "1d",
            }
            return mapping.get(int(m))

        interval = _minutes_to_interval(int(period_minutes))
        period_klines = []
        period_stats_error: str | None = None

        try:
            if effective_end > start_ms:
                if interval:
                    period_klines = await self.rest_client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_ms,
                        end_time=effective_end,
                        limit=1500,
                    )
                else:
                    # Fallback: use 1m klines and aggregate into custom periods
                    one = await self.rest_client.get_klines(
                        symbol=symbol,
                        interval="1m",
                        start_time=start_ms,
                        end_time=effective_end,
                        limit=1500,
                    )
                    if one:
                        bucket_ms = int(period_minutes) * 60_000
                        buckets: dict[int, list[Any]] = {}
                        for k in one:
                            b = ((k.open_time - start_ms) // bucket_ms) * bucket_ms + start_ms
                            buckets.setdefault(b, []).append(k)
                        for b in sorted(buckets.keys()):
                            ks = buckets[b]
                            o = ks[0].open
                            c = ks[-1].close
                            h = max(x.high for x in ks)
                            lo = min(x.low for x in ks)
                            # Build a minimal object with needed attrs
                            period_klines.append(type("K", (), {"open_time": b, "open": o, "close": c, "high": h, "low": lo}))
        except Exception as e:
            period_stats_error = str(e)
            period_klines = []

        open_p = period_klines[0].open if period_klines else None
        close_p = period_klines[-1].close if period_klines else None
        high_p = max((k.high for k in period_klines), default=None)
        low_p = min((k.low for k in period_klines), default=None)

        # Rotation factor (Exocharts definition from MP statistics settings):
        # If current period high > previous period high => +1 else -1; same for lows.
        rotation_factor: int | None = None
        if len(period_klines) >= 2:
            rf = 0
            for i in range(1, len(period_klines)):
                rf += 1 if period_klines[i].high > period_klines[i - 1].high else -1
                rf += 1 if period_klines[i].low > period_klines[i - 1].low else -1
            rotation_factor = rf

        # Initial Balance (default: first 2 periods when period_minutes=30 => first hour)
        ib_high = None
        ib_low = None
        if period_klines and int(ib_periods) > 0:
            n = min(len(period_klines), int(ib_periods))
            ib_slice = period_klines[:n]
            ib_high = max(k.high for k in ib_slice)
            ib_low = min(k.low for k in ib_slice)

        warnings: list[str] = []
        if tpo_profile.get("totals", {}).get("tpoTotalCount", 0) == 0:
            warnings.append(
                "No local footprint data found for this window. Run backfill or wait for live collection."
            )

        if period_stats_error:
            warnings.append(f"periodStatsError: {period_stats_error}")

        # ------------------------------------------------------------
        # Compact single prints/tails output
        # ------------------------------------------------------------
        # Single print arrays can easily be thousands of levels on crypto.
        # Exocharts draws them visually, but for an MCP tool response we
        # keep the payload small by default.
        if include_single_prints and str(single_prints_mode).lower() != "full":
            sp = (tpo_profile or {}).get("singlePrints")
            if isinstance(sp, dict):
                levels = list(sp.get("levels") or [])
                low_tail = list(sp.get("lowTailLevels") or [])
                high_tail = list(sp.get("highTailLevels") or [])
                body = list(sp.get("bodyLevels") or [])
                ts = float(tick_size) if tick_size is not None else None

                def _compress(vals: list[float]) -> list[dict[str, Any]]:
                    if not vals or not ts or ts <= 0:
                        return []
                    # Convert to integer ticks to avoid float adjacency issues
                    ticks = sorted({int(round(float(v) / ts)) for v in vals})
                    if not ticks:
                        return []
                    out_ranges: list[dict[str, Any]] = []
                    start_t = ticks[0]
                    end_t = ticks[0]
                    for t in ticks[1:]:
                        if t == end_t + 1:
                            end_t = t
                        else:
                            out_ranges.append(
                                {
                                    "start": round(start_t * ts, 10),
                                    "end": round(end_t * ts, 10),
                                    "count": int(end_t - start_t + 1),
                                }
                            )
                            start_t = end_t = t
                    out_ranges.append(
                        {
                            "start": round(start_t * ts, 10),
                            "end": round(end_t * ts, 10),
                            "count": int(end_t - start_t + 1),
                        }
                    )
                    return out_ranges

                # Keep tails as explicit levels (usually short), compress body + total.
                body_sample: list[float]
            if len(body) > 30:
                body_sample = body[:15] + body[-15:]
            else:
                body_sample = body

            tpo_profile["singlePrints"] = {
                "tailMinLength": int(sp.get("tailMinLength") or tail_min_len),
                "tickSize": ts,
                "counts": {
                    "levels": int(len(levels)),
                    "lowTail": int(len(low_tail)),
                    "highTail": int(len(high_tail)),
                    "body": int(len(body)),
                },
                "lowTailLevels": low_tail,
                "highTailLevels": high_tail,
                "levelRanges": _compress(levels),
                "bodyRanges": _compress(body),
                "bodySample": body_sample,
            }

        dq = self._build_data_quality(
            window_start_ms=start_ms,
            window_end_ms=effective_end,
            coverage_raw=cov_raw,
            bars_returned=None,
            levels_count=tpo_profile.get("priceLevelCount"),
            source="footprint_1m",
        )
        if dq["degraded"]:
            for key in ("poc", "vah", "val"):
                if key in tpo_profile.get("timeBased", {}):
                    tpo_profile["timeBased"][key] = None
                if key in tpo_profile.get("active", {}):
                    tpo_profile["active"][key] = None
            for key in ("vpoc", "vvah", "vval"):
                if key in tpo_profile.get("volumeBased", {}):
                    tpo_profile["volumeBased"][key] = None
            warnings.append("Low coverage: profile levels may be unreliable; key levels nulled.")

        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timestamp": timestamp_ms(),
            "date": datetime.fromtimestamp(day_start / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "session": session,
            "status": status,
            "range": {"startTime": start_ms, "endTime": end_ms, "effectiveEnd": effective_end},
            "tpo": tpo_profile,
            "periodStats": {
                "periodMinutes": int(period_minutes),
                "periodCount": len(period_klines),
                "open": open_p,
                "close": close_p,
                "high": high_p,
                "low": low_p,
                "rotationFactor": rotation_factor,
                "ibPeriods": int(ib_periods),
                "ibHigh": ib_high,
                "ibLow": ib_low,
            },
            "dataQuality": dq,
            "warnings": warnings,
        }


    async def get_market_structure(
        self,
        symbol: str,
        interval: str = "15m",
        lookback_bars: int = 400,
        pivot_left: int = 10,
        pivot_right: int = 10,
        zigzag_leg_min_percent: float | None = None,
        max_points: int = 80,
        end_time: int | None = None,
    ) -> dict[str, Any]:
        """Market Structure (swings, trend, BOS/CHoCH, optional ZigZag).

        This tool is designed to be *deterministic* and easy to consume by an LLM or UI.

        Args:
            symbol: e.g. BTCUSDT
            interval: kline interval (default 15m)
            lookback_bars: number of bars to analyze
            pivot_left/pivot_right: swing pivot detection window
            zigzag_leg_min_percent: if set (>0), also compute ZigZag pivots
            max_points: cap swing/zigzag points returned
            end_time: optional ms timestamp (UTC). If provided, analyze the window ending here.

        Returns:
            Dict with swings, trend, and break signals.
        """
        symbol = symbol.upper().strip()
        interval = interval.strip()

        tf_ms = get_timeframe_ms(interval)
        if lookback_bars <= 20:
            lookback_bars = 20
        if lookback_bars > 1500:
            lookback_bars = 1500

        # Resolve window
        if end_time is None:
            end_ms = None
            start_ms = None
        else:
            end_ms = int(end_time)
            start_ms = end_ms - (lookback_bars * tf_ms)

        klines = await self.rest_client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_ms,
            end_time=end_ms,
            limit=int(lookback_bars),
        )

        if not klines:
            return {
                "symbol": symbol,
                "interval": interval,
                "timestamp": timestamp_ms(),
                "error": "no_klines",
            }

        # Arrays
        times = [k.open_time for k in klines]
        highs = [float(k.high) for k in klines]
        lows = [float(k.low) for k in klines]
        closes = [float(k.close) for k in klines]

        last_price = closes[-1]
        last_time = times[-1]

        n = len(klines)
        L = max(1, int(pivot_left))
        R = max(1, int(pivot_right))

        # Pivot detection
        pivots: list[dict[str, Any]] = []
        for i in range(L, n - R):
            hi = highs[i]
            lo = lows[i]

            left_hi = max(highs[i - L : i + R + 1])
            left_lo = min(lows[i - L : i + R + 1])

            if hi == left_hi:
                pivots.append({"type": "high", "index": i, "time": times[i], "price": hi})
            if lo == left_lo:
                pivots.append({"type": "low", "index": i, "time": times[i], "price": lo})

        pivots.sort(key=lambda x: x["index"])

        # Enforce alternation (replace consecutive same-type pivots with the more extreme one)
        swings: list[dict[str, Any]] = []
        for p in pivots:
            if not swings:
                swings.append(p)
                continue
            last = swings[-1]
            if p["type"] == last["type"]:
                if p["type"] == "high" and p["price"] >= last["price"]:
                    swings[-1] = p
                elif p["type"] == "low" and p["price"] <= last["price"]:
                    swings[-1] = p
            else:
                swings.append(p)

        # Forward max/min for swept detection
        fwd_max_high = [0.0] * n
        fwd_min_low = [0.0] * n
        cur_max = float("-inf")
        cur_min = float("inf")
        for i in range(n - 1, -1, -1):
            cur_max = max(cur_max, highs[i])
            cur_min = min(cur_min, lows[i])
            fwd_max_high[i] = cur_max
            fwd_min_low[i] = cur_min

        # Label swings (HH/LH/HL/LL)
        last_high = None
        last_low = None
        highs_list = []
        lows_list = []

        labeled: list[dict[str, Any]] = []
        for s in swings:
            typ = s["type"]
            idx = int(s["index"])
            price = float(s["price"])
            swept = False
            if typ == "high":
                swept = (idx + 1 < n) and (fwd_max_high[idx + 1] > price)
                if last_high is None:
                    label = "SH"
                else:
                    if price > last_high:
                        label = "HH"
                    elif price < last_high:
                        label = "LH"
                    else:
                        label = "EH"
                last_high = price
                highs_list.append((s["time"], price))
            else:
                swept = (idx + 1 < n) and (fwd_min_low[idx + 1] < price)
                if last_low is None:
                    label = "SL"
                else:
                    if price > last_low:
                        label = "HL"
                    elif price < last_low:
                        label = "LL"
                    else:
                        label = "EL"
                last_low = price
                lows_list.append((s["time"], price))

            labeled.append(
                {
                    "type": typ,
                    "time": s["time"],
                    "price": price,
                    "label": label,
                    "swept": bool(swept),
                }
            )

        # Trend inference (requires 2 highs + 2 lows)
        trend = "unknown"
        if len(highs_list) >= 2 and len(lows_list) >= 2:
            prev_high = highs_list[-2][1]
            cur_high = highs_list[-1][1]
            prev_low = lows_list[-2][1]
            cur_low = lows_list[-1][1]
            if cur_high > prev_high and cur_low > prev_low:
                trend = "up"
            elif cur_high < prev_high and cur_low < prev_low:
                trend = "down"
            else:
                trend = "range"

        # Last swing high/low levels (most recent of each)
        last_swing_high = None
        last_swing_low = None
        for s in reversed(labeled):
            if last_swing_high is None and s["type"] == "high":
                last_swing_high = {"time": s["time"], "price": s["price"], "label": s["label"], "swept": s["swept"]}
            if last_swing_low is None and s["type"] == "low":
                last_swing_low = {"time": s["time"], "price": s["price"], "label": s["label"], "swept": s["swept"]}
            if last_swing_high and last_swing_low:
                break

        # BOS / CHoCH detection based on last close
        bos = None
        choch = None
        if last_swing_high and last_swing_low:
            sh = float(last_swing_high["price"])
            sl = float(last_swing_low["price"])
            if trend == "up":
                if last_price > sh:
                    bos = {"type": "bullish", "level": sh, "time": last_time, "price": last_price}
                elif last_price < sl:
                    choch = {"type": "bearish", "level": sl, "time": last_time, "price": last_price}
            elif trend == "down":
                if last_price < sl:
                    bos = {"type": "bearish", "level": sl, "time": last_time, "price": last_price}
                elif last_price > sh:
                    choch = {"type": "bullish", "level": sh, "time": last_time, "price": last_price}
            else:
                # range/unknown: report both potential breaks if crossed
                if last_price > sh:
                    bos = {"type": "bullish", "level": sh, "time": last_time, "price": last_price}
                if last_price < sl:
                    bos = bos or {"type": "bearish", "level": sl, "time": last_time, "price": last_price}

        # Optional ZigZag (percentage reversal)
        zigzag = None
        if zigzag_leg_min_percent is not None and float(zigzag_leg_min_percent) > 0:
            thr = float(zigzag_leg_min_percent) / 100.0
            zz_points: list[dict[str, Any]] = []
            pivot_price = closes[0]
            direction = 0  # 0 unknown, +1 up, -1 down
            extreme_price = pivot_price
            extreme_idx = 0

            for i in range(1, n):
                price = closes[i]
                if direction == 0:
                    if price >= pivot_price * (1 + thr):
                        direction = 1
                        extreme_price = price
                        extreme_idx = i
                    elif price <= pivot_price * (1 - thr):
                        direction = -1
                        extreme_price = price
                        extreme_idx = i
                elif direction == 1:
                    if price > extreme_price:
                        extreme_price = price
                        extreme_idx = i
                    elif price <= extreme_price * (1 - thr):
                        # reversal down: record high
                        zz_points.append({"type": "high", "time": times[extreme_idx], "price": float(extreme_price)})
                        pivot_price = extreme_price
                        direction = -1
                        extreme_price = price
                        extreme_idx = i
                else:  # -1
                    if price < extreme_price:
                        extreme_price = price
                        extreme_idx = i
                    elif price >= extreme_price * (1 + thr):
                        zz_points.append({"type": "low", "time": times[extreme_idx], "price": float(extreme_price)})
                        pivot_price = extreme_price
                        direction = 1
                        extreme_price = price
                        extreme_idx = i

            # Add last extreme as the current leg endpoint
            if direction == 1:
                zz_points.append({"type": "high", "time": times[extreme_idx], "price": float(extreme_price)})
            elif direction == -1:
                zz_points.append({"type": "low", "time": times[extreme_idx], "price": float(extreme_price)})

            # De-duplicate consecutive same type
            clean = []
            for p in zz_points:
                if not clean:
                    clean.append(p)
                    continue
                if p["type"] == clean[-1]["type"]:
                    # replace with more extreme
                    if p["type"] == "high" and p["price"] >= clean[-1]["price"]:
                        clean[-1] = p
                    elif p["type"] == "low" and p["price"] <= clean[-1]["price"]:
                        clean[-1] = p
                else:
                    clean.append(p)

            zigzag = {
                "legMinPercent": float(zigzag_leg_min_percent),
                "points": clean[-max(1, int(max_points)) :],
            }

        # Cap swings returned
        labeled = labeled[-max(1, int(max_points)) :]

        return {
            "symbol": symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timestamp": timestamp_ms(),
            "interval": interval,
            "lookbackBars": int(lookback_bars),
            "pivotLeft": int(pivot_left),
            "pivotRight": int(pivot_right),
            "lastPrice": last_price,
            "lastBarTime": last_time,
            "trend": trend,
            "lastSwingHigh": last_swing_high,
            "lastSwingLow": last_swing_low,
            "bos": bos,
            "choch": choch,
            "swings": labeled,
            "zigzag": zigzag,
        }
