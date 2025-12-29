"""SQLite storage for trade data and aggregations."""

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from src.config import get_settings
from src.utils import get_logger, timestamp_ms, get_day_start_ms


class DataStorage:
    """SQLite storage for historical trade data and aggregations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_path = self.settings.ensure_data_dir()
        self.logger = get_logger("data.storage")
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize database and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        
        await self._db.executescript("""
            -- Aggregated trades (1-minute footprint data)
            CREATE TABLE IF NOT EXISTS footprint_1m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,  -- Start of minute (ms)
                price_level REAL NOT NULL,   -- Price tick level
                buy_volume REAL NOT NULL DEFAULT 0,
                sell_volume REAL NOT NULL DEFAULT 0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(symbol, timestamp, price_level)
            );
            
            CREATE INDEX IF NOT EXISTS idx_footprint_1m_symbol_ts 
                ON footprint_1m(symbol, timestamp);
            
            -- Daily aggregated data for VWAP/Volume Profile
            CREATE TABLE IF NOT EXISTS daily_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date INTEGER NOT NULL,       -- Day start timestamp (ms)
                price_level REAL NOT NULL,   -- Price tick level
                volume REAL NOT NULL DEFAULT 0,
                buy_volume REAL NOT NULL DEFAULT 0,
                sell_volume REAL NOT NULL DEFAULT 0,
                notional REAL NOT NULL DEFAULT 0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(symbol, date, price_level)
            );
            
            CREATE INDEX IF NOT EXISTS idx_daily_trades_symbol_date 
                ON daily_trades(symbol, date);
            
            -- Session high/low tracking
            CREATE TABLE IF NOT EXISTS session_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date INTEGER NOT NULL,       -- Day start timestamp (ms)
                session TEXT NOT NULL,       -- session id/name (e.g. 'A','L','N','E' or 'tokyo'...)
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                high_time INTEGER NOT NULL,
                low_time INTEGER NOT NULL,
                volume REAL NOT NULL DEFAULT 0,
                UNIQUE(symbol, date, session)
            );
            
            CREATE INDEX IF NOT EXISTS idx_session_levels_symbol_date 
                ON session_levels(symbol, date);
            
            -- VWAP tracking
            CREATE TABLE IF NOT EXISTS vwap_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date INTEGER NOT NULL,       -- Day start timestamp (ms)
                cumulative_pv REAL NOT NULL DEFAULT 0,  -- Price * Volume
                cumulative_volume REAL NOT NULL DEFAULT 0,
                last_update INTEGER NOT NULL,
                UNIQUE(symbol, date)
            );
            
            CREATE INDEX IF NOT EXISTS idx_vwap_data_symbol_date 
                ON vwap_data(symbol, date);
            
            -- Open Interest snapshots
            CREATE TABLE IF NOT EXISTS oi_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open_interest REAL NOT NULL,
                open_interest_notional REAL NOT NULL,
                UNIQUE(symbol, timestamp)
            );
            
            CREATE INDEX IF NOT EXISTS idx_oi_snapshots_symbol_ts 
                ON oi_snapshots(symbol, timestamp);
            
            -- Depth delta snapshots
            CREATE TABLE IF NOT EXISTS depth_delta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bid_volume REAL NOT NULL,
                ask_volume REAL NOT NULL,
                net_volume REAL NOT NULL,
                percent_range REAL NOT NULL,
                mid_price REAL NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_depth_delta_symbol_ts 
                ON depth_delta(symbol, timestamp);

            -- Orderbook heatmap snapshots (binned depth ladder)
            CREATE TABLE IF NOT EXISTS orderbook_heatmap (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price_bin REAL NOT NULL,
                bid_volume REAL NOT NULL,
                ask_volume REAL NOT NULL,
                PRIMARY KEY (symbol, timestamp, price_bin)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS idx_orderbook_heatmap_symbol_ts
                ON orderbook_heatmap(symbol, timestamp);
        """)
        
        await self._db.commit()

        # --------------------------------------------------------------
        # Lightweight data migrations
        # --------------------------------------------------------------
        # v7 bug: backfill inserted session_levels with a wrong column order:
        #   (high_price, high_time, low_price, low_time, volume)
        # instead of:
        #   (high_price, low_price, high_time, low_time, volume)
        # This caused low_price to contain a millisecond timestamp (1.7e12...)
        # and high_time to contain a price (~80k).
        #
        # We fix it in-place so users can upgrade without deleting the DB.
        try:
            cursor = await self._db.execute(
                """
                UPDATE session_levels
                SET
                    low_price = high_time,
                    high_time = CAST(low_price AS INTEGER)
                WHERE
                    low_price > 1000000000
                    AND high_time < 1000000000
                """
            )
            await self._db.commit()
            rows = getattr(cursor, "rowcount", None)
            if rows and rows > 0:
                self.logger.warning("session_levels_migrated_v7_bug", rows=rows)
            await cursor.close()
        except Exception as e:
            # Migration failure should never prevent startup.
            self.logger.warning("session_levels_migration_failed", error=str(e))

        self.logger.info("database_initialized", path=str(self.db_path))
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def clear_day(self, symbol: str, day_start: int) -> None:
        """Clear all cached aggregates for a specific UTC day.

        This is used by the historical backfill to avoid double-counting.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        day_end = day_start + 86400000
        symbol = symbol.upper()

        async with self._lock:
            await self._db.execute(
                "DELETE FROM footprint_1m WHERE symbol = ? AND timestamp >= ? AND timestamp < ?",
                (symbol, day_start, day_end),
            )
            await self._db.execute(
                "DELETE FROM daily_trades WHERE symbol = ? AND date = ?",
                (symbol, day_start),
            )
            await self._db.execute(
                "DELETE FROM vwap_data WHERE symbol = ? AND date = ?",
                (symbol, day_start),
            )
            await self._db.execute(
                "DELETE FROM session_levels WHERE symbol = ? AND date = ?",
                (symbol, day_start),
            )
            await self._db.execute(
                "DELETE FROM orderbook_heatmap WHERE symbol = ? AND timestamp >= ? AND timestamp < ?",
                (symbol, day_start, day_end),
            )
            await self._db.commit()

    async def has_day_data(self, symbol: str, day_start: int, day_end: int | None = None) -> bool:
        """Return True if we have *sufficient* cached data for the given time range.

        Why this exists:
        - In v7 we treated a day as "present" if there was *any* row in vwap_data/daily_trades.
          That caused a common failure mode: a partial day (e.g. after a restart) would be
          incorrectly considered "complete", so startup backfill would skip it.

        New logic (v8):
        - We validate coverage using the 1m footprint table.
        - We require that the earliest and latest footprint timestamps are close to the requested
          range boundaries.

        This is intentionally conservative: if we are unsure, we prefer to backfill.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        symbol = symbol.upper()
        if day_end is None:
            day_end = int(day_start) + 86_400_000

        start_ms = int(day_start)
        end_ms = int(day_end)
        if end_ms <= start_ms:
            return False

        # Tolerance: allow a few minutes of boundary slack.
        tol_ms = 5 * 60_000

        async with self._lock:
            # Earliest footprint minute in range
            cur = await self._db.execute(
                "SELECT timestamp FROM footprint_1m WHERE symbol = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp ASC LIMIT 1",
                (symbol, start_ms, end_ms),
            )
            row = await cur.fetchone()
            await cur.close()
            if not row or row[0] is None:
                return False
            min_ts = int(row[0])

            # Latest footprint minute in range
            cur = await self._db.execute(
                "SELECT timestamp FROM footprint_1m WHERE symbol = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
                (symbol, start_ms, end_ms),
            )
            row = await cur.fetchone()
            await cur.close()
            if not row or row[0] is None:
                return False
            max_ts = int(row[0])

        if min_ts > start_ms + tol_ms:
            return False
        if max_ts < end_ms - tol_ms:
            return False

        return True

    async def get_day_row_counts(self, symbol: str, day_start: int) -> dict[str, int]:
        """Return row counts for key aggregates for a given day."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        symbol = symbol.upper()
        day_end = int(day_start) + 86_400_000

        async with self._lock:
            cur = await self._db.execute(
                """
                SELECT COUNT(*) FROM footprint_1m
                WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
                """,
                (symbol, int(day_start), day_end),
            )
            fp_count = int((await cur.fetchone() or (0,))[0])
            await cur.close()

            cur = await self._db.execute(
                """
                SELECT COUNT(*) FROM daily_trades
                WHERE symbol = ? AND date = ?
                """,
                (symbol, int(day_start)),
            )
            dt_count = int((await cur.fetchone() or (0,))[0])
            await cur.close()

            cur = await self._db.execute(
                """
                SELECT COUNT(*) FROM vwap_data
                WHERE symbol = ? AND date = ?
                """,
                (symbol, int(day_start)),
            )
            vwap_count = int((await cur.fetchone() or (0,))[0])
            await cur.close()

            cur = await self._db.execute(
                """
                SELECT COUNT(*) FROM session_levels
                WHERE symbol = ? AND date = ?
                """,
                (symbol, int(day_start)),
            )
            session_count = int((await cur.fetchone() or (0,))[0])
            await cur.close()

        return {
            "footprint_1m": fp_count,
            "daily_trades": dt_count,
            "vwap_data": vwap_count,
            "session_levels": session_count,
        }

    async def get_day_cvd(self, symbol: str, day_start: int) -> float:
        """Get cumulative volume delta (CVD) for a given day from persisted aggregates.

        We try the cheapest available source first:
        - `daily_trades` (already aggregated per price level)
        - fall back to `footprint_1m` if `daily_trades` is empty
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        symbol = symbol.upper()
        ms_in_day = 86_400_000
        day_end = int(day_start) + ms_in_day

        async with self._lock:
            cur = await self._db.execute(
                "SELECT SUM(buy_volume - sell_volume) FROM daily_trades WHERE symbol = ? AND date = ?",
                (symbol, int(day_start)),
            )
            row = await cur.fetchone()
            await cur.close()

            if row and row[0] is not None:
                return float(row[0])

            # Fallback (slower): sum footprint range for the day
            cur = await self._db.execute(
                """
                SELECT SUM(buy_volume - sell_volume)
                FROM footprint_1m
                WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
                """,
                (symbol, int(day_start), int(day_end)),
            )
            row = await cur.fetchone()
            await cur.close()

        if row and row[0] is not None:
            return float(row[0])
        return 0.0

    async def bulk_upsert_footprint(self, rows: list[tuple]) -> None:
        """Bulk upsert footprint rows.

        rows: [(symbol, timestamp, price_level, buy_volume, sell_volume, trade_count), ...]
        """
        if not rows:
            return
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._db.executemany(
                """
                INSERT INTO footprint_1m
                    (symbol, timestamp, price_level, buy_volume, sell_volume, trade_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timestamp, price_level) DO UPDATE SET
                    buy_volume = buy_volume + excluded.buy_volume,
                    sell_volume = sell_volume + excluded.sell_volume,
                    trade_count = trade_count + excluded.trade_count
                """,
                rows,
            )
            await self._db.commit()

    async def bulk_upsert_daily_trades(self, rows: list[tuple]) -> None:
        """Bulk upsert daily trade aggregates.

        rows: [(symbol, date, price_level, volume, buy_volume, sell_volume, notional, trade_count), ...]
        """
        if not rows:
            return
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._db.executemany(
                """
                INSERT INTO daily_trades
                    (symbol, date, price_level, volume, buy_volume, sell_volume, notional, trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date, price_level) DO UPDATE SET
                    volume = volume + excluded.volume,
                    buy_volume = buy_volume + excluded.buy_volume,
                    sell_volume = sell_volume + excluded.sell_volume,
                    notional = notional + excluded.notional,
                    trade_count = trade_count + excluded.trade_count
                """,
                rows,
            )
            await self._db.commit()

    async def bulk_set_vwap_data(self, rows: list[tuple]) -> None:
        """Set (replace) vwap_data rows.

        rows: [(symbol, date, cumulative_pv, cumulative_volume, last_update), ...]
        """
        if not rows:
            return
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._db.executemany(
                """
                INSERT INTO vwap_data
                    (symbol, date, cumulative_pv, cumulative_volume, last_update)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    cumulative_pv = excluded.cumulative_pv,
                    cumulative_volume = excluded.cumulative_volume,
                    last_update = excluded.last_update
                """,
                rows,
            )
            await self._db.commit()

    async def bulk_set_session_levels(self, rows: list[tuple]) -> None:
        """Set (replace) session level rows.

        rows: [(symbol, date, session, high_price, low_price, high_time, low_time, volume), ...]
        """
        if not rows:
            return
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._db.executemany(
                """
                INSERT INTO session_levels
                    (symbol, date, session, high_price, low_price, high_time, low_time, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date, session) DO UPDATE SET
                    high_price = excluded.high_price,
                    low_price = excluded.low_price,
                    high_time = excluded.high_time,
                    low_time = excluded.low_time,
                    volume = excluded.volume
                """,
                rows,
            )
            await self._db.commit()
    
    async def cleanup_old_data(self) -> int:
        """Remove data older than retention period."""
        if not self._db:
            return 0
        
        cutoff = timestamp_ms() - (self.settings.data_retention_days * 86_400_000)
        
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM footprint_1m WHERE timestamp < ?", (cutoff,)
            )
            deleted_footprint = cursor.rowcount
            
            cursor = await self._db.execute(
                "DELETE FROM daily_trades WHERE date < ?", (cutoff,)
            )
            deleted_daily = cursor.rowcount
            
            cursor = await self._db.execute(
                "DELETE FROM session_levels WHERE date < ?", (cutoff,)
            )
            deleted_sessions = cursor.rowcount
            
            cursor = await self._db.execute(
                "DELETE FROM vwap_data WHERE date < ?", (cutoff,)
            )
            deleted_vwap = cursor.rowcount
            
            cursor = await self._db.execute(
                "DELETE FROM oi_snapshots WHERE timestamp < ?", (cutoff,)
            )
            deleted_oi = cursor.rowcount
            
            cursor = await self._db.execute(
                "DELETE FROM depth_delta WHERE timestamp < ?", (cutoff,)
            )
            deleted_depth = cursor.rowcount

            cursor = await self._db.execute(
                "DELETE FROM orderbook_heatmap WHERE timestamp < ?", (cutoff,)
            )
            deleted_heatmap = cursor.rowcount
            
            await self._db.commit()
        
        total_deleted = (
            deleted_footprint
            + deleted_daily
            + deleted_sessions
            + deleted_vwap
            + deleted_oi
            + deleted_depth
            + deleted_heatmap
        )
        self.logger.info("cleanup_complete", deleted=total_deleted, cutoff_days=self.settings.data_retention_days)
        return total_deleted
    
    # Footprint operations
    async def upsert_footprint(
        self,
        symbol: str,
        timestamp: int,
        price_level: float,
        buy_volume: float,
        sell_volume: float,
        trade_count: int = 1,
    ) -> None:
        """Insert or update footprint data."""
        if not self._db:
            return
        
        async with self._lock:
            await self._db.execute("""
                INSERT INTO footprint_1m (symbol, timestamp, price_level, buy_volume, sell_volume, trade_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timestamp, price_level) DO UPDATE SET
                    buy_volume = buy_volume + excluded.buy_volume,
                    sell_volume = sell_volume + excluded.sell_volume,
                    trade_count = trade_count + excluded.trade_count
            """, (symbol, timestamp, price_level, buy_volume, sell_volume, trade_count))
            await self._db.commit()
    
    async def get_footprint_range(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> list[dict[str, Any]]:
        """Get footprint data for a time range."""
        if not self._db:
            return []

        cursor = await self._db.execute(
            """
            SELECT timestamp, price_level, buy_volume, sell_volume, trade_count
            FROM footprint_1m
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp, price_level
            """,
            (symbol, start_time, end_time),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


    async def get_footprint_coverage(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> dict[str, Any]:
        """Return basic coverage stats for footprint_1m in a range.

        - minute_buckets: number of distinct 1m buckets that have any data
        - min_ts/max_ts: first/last bucket timestamp seen (ms)

        Note: footprint_1m stores data in 1-minute buckets, so COUNT(DISTINCT timestamp)
        is a good proxy for completeness.
        """
        if not self._db:
            return {"minute_buckets": 0, "min_ts": None, "max_ts": None}

        cursor = await self._db.execute(
            """
            SELECT
                COUNT(DISTINCT timestamp) AS minute_buckets,
                MIN(timestamp) AS min_ts,
                MAX(timestamp) AS max_ts
            FROM footprint_1m
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            """,
            (symbol, start_time, end_time),
        )
        row = await cursor.fetchone()
        if not row:
            return {"minute_buckets": 0, "min_ts": None, "max_ts": None}

        return {
            "minute_buckets": int(row["minute_buckets"] or 0),
            "min_ts": int(row["min_ts"]) if row["min_ts"] is not None else None,
            "max_ts": int(row["max_ts"]) if row["max_ts"] is not None else None,
        }


    async def get_footprint_statistics(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        bucket_ms: int | None = None,
        timeframe_minutes: int = 30,
    ) -> list[dict[str, Any]]:
        """Get bucketed footprint statistics for a time range.

        Returns quote-denominated (USDT) values by multiplying base volume by price_level.
        This is intentionally compact for LLM/tool usage (e.g., "Footprint bar statistics").
        """
        if not self._db:
            return []

        if bucket_ms is None:
            bucket_ms = int(timeframe_minutes * 60_000)
        cursor = await self._db.execute(
            """
            SELECT
                (CAST(timestamp / ? AS INTEGER) * ?) AS bucket_start,
                SUM((buy_volume + sell_volume) * price_level) AS vol_quote,
                SUM((buy_volume - sell_volume) * price_level) AS delta_quote,
                MAX((buy_volume - sell_volume) * price_level) AS delta_max_quote,
                MIN((buy_volume - sell_volume) * price_level) AS delta_min_quote,
                SUM(buy_volume * price_level) AS buy_quote,
                SUM(sell_volume * price_level) AS sell_quote,
                SUM(trade_count) AS trades
            FROM footprint_1m
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY bucket_start
            ORDER BY bucket_start
            """,
            (bucket_ms, bucket_ms, symbol, start_time, end_time),
        )

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


    # Daily trades operations
    async def upsert_daily_trade(
        self,
        symbol: str,
        date: int,
        price_level: float,
        volume: float,
        buy_volume: float,
        sell_volume: float,
        notional: float,
        trade_count: int = 1,
    ) -> None:
        """Insert or update daily trade aggregation."""
        if not self._db:
            return
        
        async with self._lock:
            await self._db.execute("""
                INSERT INTO daily_trades (symbol, date, price_level, volume, buy_volume, sell_volume, notional, trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date, price_level) DO UPDATE SET
                    volume = volume + excluded.volume,
                    buy_volume = buy_volume + excluded.buy_volume,
                    sell_volume = sell_volume + excluded.sell_volume,
                    notional = notional + excluded.notional,
                    trade_count = trade_count + excluded.trade_count
            """, (symbol, date, price_level, volume, buy_volume, sell_volume, notional, trade_count))
            await self._db.commit()
    
    async def get_daily_trades(self, symbol: str, date: int) -> list[dict[str, Any]]:
        """Get daily trade aggregation for volume profile."""
        if not self._db:
            return []
        
        cursor = await self._db.execute("""
            SELECT price_level, volume, buy_volume, sell_volume, notional, trade_count
            FROM daily_trades
            WHERE symbol = ? AND date = ?
            ORDER BY price_level
        """, (symbol, date))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    

    async def get_profile_range(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> list[dict[str, Any]]:
        """Aggregate footprint data into a volume profile for an arbitrary time range.

        Uses `footprint_1m` as the base table so we can slice by sessions (Tokyo/London/NY)
        or any custom window.

        Returns rows of:
        - price_level
        - buy_volume
        - sell_volume
        - trade_count
        """
        if not self._db:
            return []

        cursor = await self._db.execute(
            """
            SELECT
                price_level,
                SUM(buy_volume) AS buy_volume,
                SUM(sell_volume) AS sell_volume,
                SUM(trade_count) AS trade_count
            FROM footprint_1m
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY price_level
            ORDER BY price_level
            """,
            (symbol, start_time, end_time),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]



    async def get_tpo_aggregates(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        period_ms: int,
        tick_size: float | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate footprint into TPO counts for a time range.

        TPO (Time-Price-Opportunity) counts how many *time periods* traded at a given price.
        We compute this from `footprint_1m` by counting distinct period-buckets.

        If `tick_size` is provided, we bucket price levels down to that tick size first,
        and then count distinct periods per bucket (deduplicates multiple prints inside
        the same bucket within the same time period).

        Args:
            symbol: Trading pair symbol
            start_time: range start (ms)
            end_time: range end (ms, exclusive)
            period_ms: period size in milliseconds (e.g. 30m = 1_800_000)
            tick_size: optional price bucket size (e.g. 50 for daily BTC TPO)

        Returns:
            List of rows with keys:
              - price_level (bucketed if tick_size provided)
              - tpo_count
              - buy_volume
              - sell_volume
              - trade_count
              - notional (SUM(volume * price))
              - delta_notional (SUM(delta * price))
        """
        if not self._db:
            return []

        if period_ms <= 0:
            raise ValueError(f"period_ms must be positive, got {period_ms}")

        use_bucket = tick_size is not None and float(tick_size) > 0
        if use_bucket:
            ts = float(tick_size)
            cursor = await self._db.execute(
                """
                SELECT
                    (CAST(price_level / ? AS INT) * ?) AS price_level,
                    COUNT(DISTINCT CAST((timestamp - ?) / ? AS INT)) AS tpo_count,
                    SUM(buy_volume) AS buy_volume,
                    SUM(sell_volume) AS sell_volume,
                    SUM(trade_count) AS trade_count,
                    SUM((buy_volume + sell_volume) * price_level) AS notional,
                    SUM((buy_volume - sell_volume) * price_level) AS delta_notional
                FROM footprint_1m
                WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
                GROUP BY (CAST(price_level / ? AS INT) * ?)
                ORDER BY price_level
                """,
                (ts, ts, start_time, period_ms, symbol, start_time, end_time, ts, ts),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT
                    price_level,
                    COUNT(DISTINCT CAST((timestamp - ?) / ? AS INT)) AS tpo_count,
                    SUM(buy_volume) AS buy_volume,
                    SUM(sell_volume) AS sell_volume,
                    SUM(trade_count) AS trade_count,
                    SUM((buy_volume + sell_volume) * price_level) AS notional,
                    SUM((buy_volume - sell_volume) * price_level) AS delta_notional
                FROM footprint_1m
                WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
                GROUP BY price_level
                ORDER BY price_level
                """,
                (start_time, period_ms, symbol, start_time, end_time),
            )

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_tpo_period_matrix(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        period_ms: int,
        tick_size: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get per-period per-price aggregates for TPO/Market Profile analysis.

        Returns rows with:
          - period_idx: 0-based index from start_time
          - price_level: bucketed price level (if tick_size is provided)
          - buy_volume, sell_volume, trade_count (base units)
          - notional, delta_notional (quote units)

        Notes:
        - footprint_1m stores per-minute price-level aggregates. This query
          folds them into larger period buckets (e.g., 30m).
        """
        if not self._db:
            return []

        symbol = symbol.upper()

        # Build price expression and bind params (tick_size needs to be bound multiple times)
        if tick_size is not None and tick_size > 0:
            price_expr = "(CAST(price_level / ? AS INT) * ?)"
            # params order matches placeholders in SQL below
            params = (
                start_time,
                period_ms,
                tick_size,
                tick_size,
                tick_size,
                tick_size,
                tick_size,
                tick_size,
                symbol,
                start_time,
                end_time,
            )
        else:
            price_expr = "price_level"
            params = (
                start_time,
                period_ms,
                symbol,
                start_time,
                end_time,
            )

        sql = f"""
            SELECT
                CAST((timestamp - ?) / ? AS INT) AS period_idx,
                {price_expr} AS price_level,
                SUM(buy_volume) AS buy_volume,
                SUM(sell_volume) AS sell_volume,
                SUM(trade_count) AS trade_count,
                (SUM(buy_volume) + SUM(sell_volume)) * {price_expr} AS notional,
                (SUM(buy_volume) - SUM(sell_volume)) * {price_expr} AS delta_notional
            FROM footprint_1m
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY period_idx, price_level
            ORDER BY period_idx, price_level
        """

        async with self._lock:
            cursor = await self._db.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # VWAP operations
    async def update_vwap(
        self,
        symbol: str,
        date: int,
        price: float,
        volume: float,
    ) -> None:
        """Update VWAP cumulative values."""
        if not self._db:
            return
        
        pv = price * volume
        now = timestamp_ms()
        
        async with self._lock:
            await self._db.execute("""
                INSERT INTO vwap_data (symbol, date, cumulative_pv, cumulative_volume, last_update)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    cumulative_pv = cumulative_pv + excluded.cumulative_pv,
                    cumulative_volume = cumulative_volume + excluded.cumulative_volume,
                    last_update = excluded.last_update
            """, (symbol, date, pv, volume, now))
            await self._db.commit()
    
    async def get_vwap(self, symbol: str, date: int) -> dict[str, float] | None:
        """Get VWAP data for a specific day.

        This returns both the raw cumulative fields and the derived VWAP.
        """
        if not self._db:
            return None
        
        cursor = await self._db.execute("""
            SELECT cumulative_pv, cumulative_volume, last_update
            FROM vwap_data
            WHERE symbol = ? AND date = ?
        """, (symbol, date))
        
        row = await cursor.fetchone()
        if not row:
            return None
        
        cumulative_pv = float(row["cumulative_pv"])
        cumulative_volume = float(row["cumulative_volume"])
        vwap = (cumulative_pv / cumulative_volume) if cumulative_volume > 0 else 0.0
        return {
            "vwap": vwap,
            "volume": cumulative_volume,
            "notional": cumulative_pv,
            "cumulative_pv": cumulative_pv,
            "cumulative_volume": cumulative_volume,
            "last_update": float(row["last_update"]),
        }
    
    # Session levels operations

    async def update_session_levels(
        self,
        symbol: str,
        date: int,
        session: str,
        price: float,
        timestamp: int,
        volume: float,
    ) -> None:
        """Update session high/low.

        Notes:
            * Always increments `volume` exactly once per trade.
            * Updates `high_time`/`low_time` only when a *new* extreme is made.
        """
        if not self._db:
            return

        async with self._lock:
            cursor = await self._db.execute(
                """
                SELECT high_price, low_price, high_time, low_time
                FROM session_levels
                WHERE symbol = ? AND date = ? AND session = ?
                """,
                (symbol, date, session),
            )
            row = await cursor.fetchone()

            if row:
                prev_high = float(row["high_price"])
                prev_low = float(row["low_price"])
                prev_high_time = int(row["high_time"]) if row["high_time"] is not None else None
                prev_low_time = int(row["low_time"]) if row["low_time"] is not None else None

                new_high = price > prev_high
                new_low = price < prev_low

                high_price = price if new_high else prev_high
                low_price = price if new_low else prev_low
                high_time = timestamp if new_high else prev_high_time
                low_time = timestamp if new_low else prev_low_time

                await self._db.execute(
                    """
                    UPDATE session_levels
                    SET high_price = ?, high_time = ?,
                        low_price = ?, low_time = ?,
                        volume = volume + ?
                    WHERE symbol = ? AND date = ? AND session = ?
                    """,
                    (
                        high_price,
                        high_time,
                        low_price,
                        low_time,
                        volume,
                        symbol,
                        date,
                        session,
                    ),
                )
            else:
                await self._db.execute(
                    """
                    INSERT INTO session_levels (
                        symbol, date, session,
                        high_price, low_price,
                        high_time, low_time,
                        volume
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (symbol, date, session, price, price, timestamp, timestamp, volume),
                )

            await self._db.commit()

    async def get_session_levels(self, symbol: str, date: int) -> dict[str, dict[str, Any]]:
        """Get all session levels for a day."""
        if not self._db:
            return {}
        
        cursor = await self._db.execute("""
            SELECT session, high_price, low_price, high_time, low_time, volume
            FROM session_levels
            WHERE symbol = ? AND date = ?
        """, (symbol, date))
        
        rows = await cursor.fetchall()
        return {
            row["session"]: {
                "high": row["high_price"],
                "low": row["low_price"],
                "high_time": row["high_time"],
                "low_time": row["low_time"],
                "volume": row["volume"],
            }
            for row in rows
        }
    
    # Open Interest operations
    async def save_oi_snapshot(
        self,
        symbol: str,
        timestamp: int,
        open_interest: float,
        open_interest_notional: float,
    ) -> None:
        """Save OI snapshot."""
        if not self._db:
            return
        
        async with self._lock:
            await self._db.execute("""
                INSERT OR REPLACE INTO oi_snapshots (symbol, timestamp, open_interest, open_interest_notional)
                VALUES (?, ?, ?, ?)
            """, (symbol, timestamp, open_interest, open_interest_notional))
            await self._db.commit()
    
    async def get_oi_history(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> list[dict[str, Any]]:
        """Get OI history for a time range."""
        if not self._db:
            return []
        
        cursor = await self._db.execute("""
            SELECT timestamp, open_interest, open_interest_notional
            FROM oi_snapshots
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """, (symbol, start_time, end_time))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # Depth delta operations
    async def save_depth_delta(
        self,
        symbol: str,
        timestamp: int,
        bid_volume: float,
        ask_volume: float,
        percent_range: float,
        mid_price: float,
    ) -> None:
        """Save depth delta snapshot."""
        if not self._db:
            return
        
        net_volume = bid_volume - ask_volume
        
        async with self._lock:
            await self._db.execute("""
                INSERT INTO depth_delta (symbol, timestamp, bid_volume, ask_volume, net_volume, percent_range, mid_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, bid_volume, ask_volume, net_volume, percent_range, mid_price))
            await self._db.commit()
    
    async def get_depth_delta_history(
        self,
        symbol: str,
        lookback_seconds: int,
    ) -> list[dict[str, Any]]:
        """Get depth delta history."""
        if not self._db:
            return []
        
        start_time = timestamp_ms() - (lookback_seconds * 1000)
        
        cursor = await self._db.execute("""
            SELECT timestamp, bid_volume, ask_volume, net_volume, percent_range, mid_price
            FROM depth_delta
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp
        """, (symbol, start_time))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------
    # Orderbook heatmap snapshots
    # ------------------------------
    async def save_orderbook_heatmap_snapshot(
        self,
        symbol: str,
        timestamp: int,
        bins: list[tuple[float, float, float]],
    ) -> None:
        """Save a single heatmap snapshot.

        Args:
            bins: List of (price_bin, bid_volume, ask_volume).
        """
        if not self._db or not bins:
            return

        async with self._lock:
            await self._db.executemany(
                """
                INSERT OR REPLACE INTO orderbook_heatmap (symbol, timestamp, price_bin, bid_volume, ask_volume)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(symbol, timestamp, float(p), float(bid), float(ask)) for (p, bid, ask) in bins],
            )
            await self._db.commit()

    async def get_latest_orderbook_heatmap_snapshot(
        self,
        symbol: str,
    ) -> tuple[int | None, list[dict[str, Any]]]:
        """Return latest heatmap snapshot for symbol.

        Returns:
            (timestamp_ms, rows) where rows contain price_bin, bid_volume, ask_volume.
        """
        if not self._db:
            return None, []

        cursor = await self._db.execute(
            "SELECT MAX(timestamp) AS ts FROM orderbook_heatmap WHERE symbol = ?",
            (symbol,),
        )
        row = await cursor.fetchone()
        if not row or row["ts"] is None:
            return None, []
        ts = int(row["ts"])

        cursor = await self._db.execute(
            """
            SELECT price_bin, bid_volume, ask_volume
            FROM orderbook_heatmap
            WHERE symbol = ? AND timestamp = ?
            ORDER BY price_bin
            """,
            (symbol, ts),
        )
        rows = await cursor.fetchall()
        return ts, [dict(r) for r in rows]

    async def get_orderbook_heatmap_rows(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> list[dict[str, Any]]:
        """Get heatmap rows for time range (raw rows)."""
        if not self._db:
            return []

        cursor = await self._db.execute(
            """
            SELECT timestamp, price_bin, bid_volume, ask_volume
            FROM orderbook_heatmap
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp, price_bin
            """,
            (symbol, start_time, end_time),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
