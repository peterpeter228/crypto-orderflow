"""Historical backfill utilities.

The real-time websocket stream only provides data from *now* onward. Many of the
orderflow-style indicators shown in your screenshots (previous-day VWAP / POC /
VAH-VAL / session highs-lows / etc.) require historical prints.

This module backfills Binance Futures `aggTrades` into our local SQLite storage:

- daily_trades (for Volume Profile)
- footprint_1m (for Footprint + Delta/CVD)
- vwap_data (for VWAP)
- session_levels (configurable session highs/lows)

It is designed to be:
- **Precise** (uses Binance's official aggTrades endpoint)
- **Deterministic** (optional clear_days=True to rebuild cleanly)
- **Rate-limit aware** (pause_ms + optional max_requests cap)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import asyncio
import math
from datetime import datetime, timezone


from src.binance.rest_client import BinanceRestClient
from src.binance.vision import BinanceVisionAggTrades
from src.data.storage import DataStorage
from src.config import get_settings
from src.utils import get_logger, timestamp_ms, round_to_tick
from src.utils.helpers import get_day_start_ms


@dataclass
class BackfillResult:
    symbol: str
    start_time: int
    end_time: int
    requests_made: int
    trades_fetched: int
    days_cleared: int
    elapsed_ms: int
    cvd_end_day: float | None = None


def get_required_backfill_range() -> tuple[int, int]:
    """Return the default startup backfill range.

    We align the start to the UTC day boundary so "previous day" indicators are
    immediately available after backfill.
    """
    settings = get_settings()
    end_ms = timestamp_ms()
    start_ms = end_ms - int(settings.backfill_lookback_hours) * 3_600_000
    start_ms = get_day_start_ms(start_ms)
    return start_ms, end_ms


def _parse_hhmm(s: str) -> int:
    """Return minutes-from-midnight for an 'HH:MM' string."""
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def _parse_session_range(rng: str) -> tuple[int, int]:
    """Parse 'HH:MM-HH:MM' to (start_minute, end_minute)."""
    a, b = rng.split("-")
    return _parse_hhmm(a), _parse_hhmm(b)


def _sessions_for_timestamp(
    ts_ms: int,
    day_start_ms: int,
    sessions: list[tuple[str, int, int]],
) -> list[tuple[str, int]]:
    """Return a list of active sessions for a timestamp.

    This matches the real-time `SessionLevelsCalculator` behavior where overlapping sessions
    (e.g. London & NY) can both be updated by the same trade.

    For sessions that span midnight (end < start), timestamps after midnight belong to the
    previous day's session.
    """
    ms_in_day = 86_400_000
    minutes = int((ts_ms - day_start_ms) // 60_000)

    active: list[tuple[str, int]] = []

    for name, start_min, end_min in sessions:
        if end_min > start_min:
            if start_min <= minutes < end_min:
                active.append((name, day_start_ms))
        else:
            # Spans midnight, e.g. 22:00-02:00
            if minutes >= start_min:
                active.append((name, day_start_ms))
            elif minutes < end_min:
                active.append((name, day_start_ms - ms_in_day))

    return active


class AggTradesBackfiller:
    """Backfill Binance Futures aggTrades into local storage."""

    def __init__(self, rest_client: BinanceRestClient, storage: DataStorage):
        self.rest = rest_client
        self.storage = storage
        self.settings = get_settings()
        self.logger = get_logger("data.backfill")

        # Optional Binance Vision client (public daily aggTrades zips).
        # Using this avoids REST 429 rate limiting when backfilling full past days.
        self.vision: BinanceVisionAggTrades | None = None
        if self.settings.backfill_source in ("auto", "vision"):
            try:
                self.vision = BinanceVisionAggTrades(self.settings, self.logger)
            except Exception as e:
                # Never fail startup due to Vision init; we can always fall back to REST.
                self.logger.warning("vision_client_init_failed", error=str(e))

        # Session definitions from settings (UTC)
        # Format: [(name, start_minute, end_minute), ...]
        self._sessions: list[tuple[str, int, int]] = [
            (s.name, s.time.start_minutes, s.time.end_minutes)
            for s in self.settings.session_defs
        ]

    async def close(self) -> None:
        """Close any network clients held by the backfiller."""
        if self.vision is not None:
            await self.vision.close()

    async def backfill_symbol_range(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        max_requests: int | None = None,
        pause_ms: int = 50,
        clear_days: bool = True,
    ) -> BackfillResult:
        """Backfill a symbol for a specific time range.

        Notes on Binance aggTrades pagination:
        - The endpoint supports *either* fromId pagination *or* time filtering.
        - If `startTime` and `endTime` are both supplied, Binance requires the window <= 1 hour.
        - Some Binance edge cases return 400 for invalid parameter combos; we therefore never send
          `fromId` together with `startTime/endTime`.

        To stay compliant and robust, we:
        1) Split the requested range into day windows.
        2) Further split each day into <=60 minute chunks for the initial request.
        3) If a chunk contains >1000 aggTrades, we continue paging with `fromId` only (no time params)
           and stop as soon as trade.timestamp reaches the chunk end.
        """
        symbol = symbol.upper()

        # Binance aggTrades only supports querying trade history up to ~1 year back.
        # Clamp overly-old ranges so startup doesn't get stuck on 400 responses.
        one_year_ms = 365 * 86_400_000
        min_start = int(end_time) - one_year_ms
        if start_time < min_start:
            self.logger.warning(
                "backfill_start_clamped_one_year",
                symbol=symbol,
                requested_start=start_time,
                clamped_start=min_start,
                end_time=end_time,
            )
            start_time = min_start

        if end_time <= start_time:
            return BackfillResult(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                requests_made=0,
                trades_fetched=0,
                days_cleared=0,
                elapsed_ms=0,
                cvd_end_day=None,
            )

        t0 = timestamp_ms()
        ms_in_day = 86_400_000

        # Used to decide whether a day is a fully completed UTC day (in the past).
        # For completed days, we can prefer Binance Vision daily aggTrades zips.
        today_start_ms = get_day_start_ms(timestamp_ms())

        # Determine covered days. Treat the range as [start_time, end_time) for day selection.
        last_inclusive = max(start_time, end_time - 1)
        d0 = get_day_start_ms(start_time)
        d1 = get_day_start_ms(last_inclusive)

        days: list[int] = []
        day = d0
        while day <= d1:
            days.append(day)
            day += ms_in_day

        # Optional cleanup / or smart backfill only when missing.
        #
        # IMPORTANT: our storage upserts are additive ("+=") to support high-throughput
        # streaming ingestion. That means backfilling a day that already has partial data
        # would double-count unless we clear it first. Therefore, any day we decide to
        # backfill is cleared before ingestion.
        days_cleared = 0
        days_to_backfill: list[int] = []
        if clear_days:
            # Force rebuild of every day in the requested range.
            for d in days:
                await self.storage.clear_day(symbol, d)
                days_cleared += 1
            days_to_backfill = days
        else:
            # Smart mode: only rebuild days that are missing or incomplete in storage.
            for d in days:
                d_end = min(d + ms_in_day, end_time)
                if not await self.storage.has_day_data(symbol, d, d_end):
                    days_to_backfill.append(d)

            for d in days_to_backfill:
                await self.storage.clear_day(symbol, d)
                days_cleared += 1

        tick_size = self.settings.get_tick_size(symbol)
        requests_made = 0
        trades_fetched = 0

        # Aggregations (flush periodically to keep memory bounded)
        fp: dict[tuple[str, int, float], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        dt: dict[tuple[str, int, float], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])  # vol, buy, sell, notional, trade_count
        vwap: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        day_cvd: dict[int, float] = defaultdict(float)
        session_map: dict[tuple[str, int, str], dict[str, float]] = {}

        async def flush() -> None:
            nonlocal fp, dt, vwap, session_map

            if fp:
                fp_rows = [
                    (sym, ts, price_level, buy, sell, int(cnt))
                    for (sym, ts, price_level), (buy, sell, cnt) in fp.items()
                ]
                await self.storage.bulk_upsert_footprint(fp_rows)
                fp.clear()

            if dt:
                dt_rows = [
                    (sym, date, price_level, vol, buy, sell, notional, int(cnt))
                    for (sym, date, price_level), (vol, buy, sell, notional, cnt) in dt.items()
                ]
                await self.storage.bulk_upsert_daily_trades(dt_rows)
                dt.clear()

            if vwap:
                vwap_rows = [
                    (sym, date, pv, vol, timestamp_ms())
                    for (sym, date), (pv, vol) in vwap.items()
                ]
                await self.storage.bulk_set_vwap_data(vwap_rows)
                # Keep vwap cumulative around; it is cheap and used for final result
                # (we do not clear it here)

            if session_map:
                sess_rows = []
                for (sym, date, session), m in session_map.items():
                    sess_rows.append(
                        (
                            sym,
                            date,
                            session,
                            m["high"],
                            m["low"],
                            int(m["high_time"]),
                            int(m["low_time"]),
                            m["volume"],
                        )
                    )
                await self.storage.bulk_set_session_levels(sess_rows)
                # Do NOT clear session_map: we maintain cumulative high/low/volume for the whole session/day.
                # Clearing here would overwrite rows with partial data on subsequent flushes.

        # Flush threshold: when we have ~50k unique footprint keys, flush.
        FLUSH_KEYS = 50_000

        # Chunk size for startTime+endTime requests (Binance requires <=60 minutes).
        try:
            chunk_minutes = int(self.settings.backfill_chunk_minutes)
        except Exception:
            chunk_minutes = 60
        chunk_minutes = max(1, min(60, chunk_minutes))
        chunk_ms = chunk_minutes * 60_000

        last_processed_id: int | None = None

        def _process_page(
            trades: list[AggTrade],
            stop_ts: int,
        ) -> tuple[bool, bool]:
            """Process trades in a page.

            Returns:
                processed_any: whether we processed at least one trade within [start_time, stop_ts)
                boundary_reached: whether we encountered a trade at/after stop_ts (and therefore
                                  should stop paging for this chunk)
            """
            nonlocal trades_fetched, last_processed_id

            processed_any = False
            boundary_reached = False

            for tr in trades:
                # De-duplicate across chunk boundaries
                if last_processed_id is not None and int(tr.agg_trade_id) <= last_processed_id:
                    continue

                if tr.timestamp < start_time:
                    continue

                # Stop at chunk boundary (stop_ts is exclusive)
                if tr.timestamp >= stop_ts:
                    boundary_reached = True
                    break

                processed_any = True
                trades_fetched += 1

                day_start = get_day_start_ms(tr.timestamp)
                minute_start = (tr.timestamp // 60_000) * 60_000

                price_level = round_to_tick(tr.price, tick_size)
                vol = float(tr.quantity)
                notional = float(tr.price) * vol

                is_sell = bool(tr.is_buyer_maker)
                buy = 0.0 if is_sell else vol
                sell = vol if is_sell else 0.0

                # Footprint (1m) aggregation
                kfp = (symbol, minute_start, price_level)
                fp[kfp][0] += buy
                fp[kfp][1] += sell
                fp[kfp][2] += 1.0

                # Daily trades for volume profile
                kdt = (symbol, day_start, price_level)
                dt[kdt][0] += vol
                dt[kdt][1] += buy
                dt[kdt][2] += sell
                dt[kdt][3] += notional
                dt[kdt][4] += 1.0

                # VWAP
                kv = (symbol, day_start)
                vwap[kv][0] += notional
                vwap[kv][1] += vol

                # Day CVD
                day_cvd[day_start] += (buy - sell)

                # Session levels (can overlap, e.g. London & NY)
                for sess_name, sess_day in _sessions_for_timestamp(tr.timestamp, day_start, self._sessions):
                    ks = (symbol, sess_day, sess_name)
                    m = session_map.get(ks)
                    if m is None:
                        session_map[ks] = {
                            "high": float(tr.price),
                            "high_time": float(tr.timestamp),
                            "low": float(tr.price),
                            "low_time": float(tr.timestamp),
                            "volume": vol,
                        }
                    else:
                        m["volume"] += vol
                        if tr.price > m["high"]:
                            m["high"] = float(tr.price)
                            m["high_time"] = float(tr.timestamp)
                        if tr.price < m["low"]:
                            m["low"] = float(tr.price)
                            m["low_time"] = float(tr.timestamp)

                last_processed_id = int(tr.agg_trade_id)

            return processed_any, boundary_reached

        if not days_to_backfill:
            # Nothing to do.
            elapsed = timestamp_ms() - t0
            cvd_end = None
            try:
                cvd_end = float(await self.storage.get_day_cvd(symbol, get_day_start_ms(end_time)))
            except Exception:
                cvd_end = None

            return BackfillResult(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                requests_made=0,
                trades_fetched=0,
                days_cleared=days_cleared,
                elapsed_ms=elapsed,
                cvd_end_day=cvd_end,
            )

        # Backfill day by day (keeps ranges small and makes "missing days" logic easy)
        capped = False
        for day_start in days_to_backfill:
            if capped:
                break

            day_range_start = max(start_time, day_start)
            day_range_end = min(end_time, day_start + ms_in_day)
            if day_range_end <= day_range_start:
                continue

            # Prefer Binance Vision for completed (past) UTC days.
            # This avoids hammering `/fapi/v1/aggTrades` and triggering 429 rate limits.
            use_vision = (
                self.vision is not None
                and self.settings.backfill_source in ("auto", "vision")
                and (day_start + ms_in_day) <= today_start_ms
            )
            if use_vision:
                if max_requests is not None and requests_made >= max_requests:
                    self.logger.warning(
                        "backfill_request_cap_hit",
                        symbol=symbol,
                        requests_made=requests_made,
                        max_requests=max_requests,
                    )
                    capped = True
                    break

                day_date = datetime.fromtimestamp(day_start / 1000.0, tz=timezone.utc).date()
                self.logger.info(
                    "backfill_day_vision_start",
                    symbol=symbol,
                    day=str(day_date),
                    range_start=day_range_start,
                    range_end=day_range_end,
                )
                vision_ok = False
                try:
                    batch: list[AggTrade] = []
                    async for tr in self.vision.iter_daily_aggtrades(
                        symbol=symbol,
                        day=day_date,
                        start_ms=day_range_start,
                        end_ms=day_range_end,
                    ):
                        batch.append(tr)
                        if len(batch) >= 5000:
                            _process_page(batch, day_range_end)
                            batch.clear()

                            if len(fp) >= FLUSH_KEYS or len(dt) >= FLUSH_KEYS:
                                await flush()

                            # Yield control so the event loop stays responsive
                            await asyncio.sleep(0)

                    if batch:
                        _process_page(batch, day_range_end)
                        batch.clear()

                    # Final flush for this day
                    await flush()
                    vision_ok = True
                except Exception as e:
                    self.logger.warning(
                        "backfill_day_vision_failed",
                        symbol=symbol,
                        day=str(day_date),
                        error=str(e),
                    )

                # Count the Vision fetch attempt as a single request for stats/capping purposes.
                requests_made += 1

                if vision_ok:
                    self.logger.info(
                        "backfill_day_vision_done",
                        symbol=symbol,
                        day=str(day_date),
                    )
                    continue

                # If Vision is configured but unavailable for this day, fall back to REST.
                self.logger.info(
                    "backfill_day_vision_fallback_to_rest",
                    symbol=symbol,
                    day=str(day_date),
                )

            chunk_start = day_range_start
            while chunk_start < day_range_end:
                if max_requests is not None and requests_made >= max_requests:
                    self.logger.warning(
                        "backfill_request_cap_hit",
                        symbol=symbol,
                        requests_made=requests_made,
                        max_requests=max_requests,
                    )
                    capped = True
                    break

                chunk_end = min(day_range_end, chunk_start + chunk_ms)

                # First page for this chunk uses startTime+endTime (<=60 min)
                trades = await self.rest.get_agg_trades(
                    symbol=symbol,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    limit=1000,
                )
                requests_made += 1

                processed_any, boundary_reached = _process_page(trades, chunk_end)

                # Periodic flush
                if len(fp) >= FLUSH_KEYS or len(dt) >= FLUSH_KEYS:
                    await flush()

                if pause_ms > 0:
                    await asyncio.sleep(pause_ms / 1000.0)

                # If the initial page already crossed the boundary (or had no trades), stop paging.
                if not trades or boundary_reached or not processed_any:
                    chunk_start = chunk_end
                    continue

                # If less than the limit returned, we're done for this chunk.
                if len(trades) < 1000:
                    chunk_start = chunk_end
                    continue

                # There may be more trades within the chunk; continue paging via fromId only.
                while True:
                    if max_requests is not None and requests_made >= max_requests:
                        self.logger.warning(
                            "backfill_request_cap_hit",
                            symbol=symbol,
                            requests_made=requests_made,
                            max_requests=max_requests,
                        )
                        capped = True
                        break

                    if last_processed_id is None:
                        break

                    next_from_id = last_processed_id + 1
                    page = await self.rest.get_agg_trades(
                        symbol=symbol,
                        from_id=next_from_id,
                        limit=1000,
                    )
                    requests_made += 1

                    processed_any2, boundary_reached2 = _process_page(page, chunk_end)

                    if len(fp) >= FLUSH_KEYS or len(dt) >= FLUSH_KEYS:
                        await flush()

                    if pause_ms > 0:
                        await asyncio.sleep(pause_ms / 1000.0)

                    if not page or boundary_reached2 or not processed_any2:
                        break

                    if len(page) < 1000:
                        break

                chunk_start = chunk_end

        # Final flush
        await flush()

        elapsed = timestamp_ms() - t0
        cvd_end = None
        end_day = get_day_start_ms(end_time)
        if end_day in day_cvd:
            cvd_end = float(day_cvd[end_day])

        return BackfillResult(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            requests_made=requests_made,
            trades_fetched=trades_fetched,
            days_cleared=days_cleared,
            elapsed_ms=elapsed,
            cvd_end_day=cvd_end,
        )
