"""Orderbook Depth Delta calculator."""

from typing import Any
from collections import deque
from dataclasses import dataclass

from src.data.storage import DataStorage
from src.data.orderbook import OrderbookManager
from src.config import get_settings
from src.utils import get_logger, timestamp_ms


@dataclass
class DepthSnapshot:
    """Snapshot of orderbook depth within a price range."""
    timestamp: int
    mid_price: float
    bid_volume: float
    ask_volume: float
    net_volume: float  # bid - ask
    percent_range: float


class DepthDeltaCalculator:
    """Calculate orderbook depth delta over time."""
    
    def __init__(self, storage: DataStorage, orderbook_manager: OrderbookManager):
        self.storage = storage
        self.orderbook = orderbook_manager
        self.settings = get_settings()
        self.logger = get_logger("indicators.depth_delta")
        
        # Recent snapshots for real-time tracking
        self._snapshots: dict[str, deque[DepthSnapshot]] = {}
        self._last_snapshot_time: dict[str, int] = {}
    
    def _get_snapshots(self, symbol: str) -> deque[DepthSnapshot]:
        """Get or create snapshot deque for symbol."""
        symbol = symbol.upper()
        if symbol not in self._snapshots:
            self._snapshots[symbol] = deque(maxlen=1000)
        return self._snapshots[symbol]
    
    async def take_snapshot(
        self,
        symbol: str,
        percent_range: float | None = None,
    ) -> DepthSnapshot | None:
        """Take a depth snapshot for symbol.
        
        Args:
            symbol: Trading pair symbol
            percent_range: Price range percentage (default from config)
        
        Returns:
            DepthSnapshot or None if orderbook not available
        """
        symbol = symbol.upper()
        
        if percent_range is None:
            percent_range = self.settings.orderbook_depth_percent
        
        depth = self.orderbook.get_depth_within_percent(symbol, percent_range)
        
        if depth is None:
            return None
        
        snapshot = DepthSnapshot(
            timestamp=depth["timestamp"],
            mid_price=depth["midPrice"],
            bid_volume=depth["bidVolume"],
            ask_volume=depth["askVolume"],
            net_volume=depth["netVolume"],
            percent_range=percent_range,
        )
        
        # Store in memory
        snapshots = self._get_snapshots(symbol)
        snapshots.append(snapshot)
        self._last_snapshot_time[symbol] = snapshot.timestamp
        
        # Persist to storage
        await self.storage.save_depth_delta(
            symbol=symbol,
            timestamp=snapshot.timestamp,
            bid_volume=snapshot.bid_volume,
            ask_volume=snapshot.ask_volume,
            percent_range=snapshot.percent_range,
            mid_price=snapshot.mid_price,
        )
        
        return snapshot
    
    async def should_take_snapshot(self, symbol: str) -> bool:
        """Check if it's time to take a new snapshot.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            True if should take snapshot
        """
        symbol = symbol.upper()
        
        last_time = self._last_snapshot_time.get(symbol, 0)
        interval_ms = self.settings.orderbook_update_interval_sec * 1000
        
        return timestamp_ms() - last_time >= interval_ms
    
    def get_latest_snapshot(self, symbol: str) -> DepthSnapshot | None:
        """Get most recent depth snapshot."""
        symbol = symbol.upper()
        snapshots = self._get_snapshots(symbol)
        
        if not snapshots:
            return None
        
        return snapshots[-1]
    
    def get_recent_snapshots(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[DepthSnapshot]:
        """Get recent depth snapshots from memory.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of snapshots
        
        Returns:
            List of DepthSnapshots
        """
        symbol = symbol.upper()
        snapshots = self._get_snapshots(symbol)
        return list(snapshots)[-limit:]
    
    async def get_depth_delta_series(
        self,
        symbol: str,
        percent_range: float | None = None,
        lookback_seconds: int = 3600,
        *,
        max_points: int = 30,
        granularity_sec: int = 60,
    ) -> dict[str, Any]:
        """Get depth delta time series.
        
        Args:
            symbol: Trading pair symbol
            percent_range: Filter by percent range
            lookback_seconds: How far back to look
        
        Returns:
            Depth delta analysis
        """
        symbol = symbol.upper()
        
        # Get from storage
        rows = await self.storage.get_depth_delta_history(symbol, lookback_seconds)
        
        if not rows:
            return {
                "symbol": symbol,
                "timestamp": timestamp_ms(),
                "lookbackSeconds": lookback_seconds,
                "snapshots": [],
                "analysis": None,
            }
        
        # Filter by percent range if specified
        if percent_range is not None:
            rows = [r for r in rows if abs(r["percent_range"] - percent_range) < 0.01]
        
        # Convert DB rows to raw points
        points = [
            {
                "timestamp": r["timestamp"],
                "bidVolume": r["bid_volume"],
                "askVolume": r["ask_volume"],
                "netVolume": r["net_volume"],
                "midPrice": r["mid_price"],
            }
            for r in rows
        ]

        # Downsample to avoid huge payloads / context overflow.
        if granularity_sec and granularity_sec > 0:
            bucket_ms = granularity_sec * 1000
            downsampled: list[dict[str, Any]] = []
            last_bucket = None
            last_point = None
            for p in points:
                b = p["timestamp"] // bucket_ms
                if last_bucket is None:
                    last_bucket = b
                if b != last_bucket:
                    if last_point is not None:
                        downsampled.append(last_point)
                    last_bucket = b
                last_point = p
            if last_point is not None:
                downsampled.append(last_point)
            points = downsampled

        # Calculate deltas between points
        deltas: list[dict[str, Any]] = []
        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            deltas.append(
                {
                    "timestamp": curr["timestamp"],
                    "bidVolume": curr["bidVolume"],
                    "askVolume": curr["askVolume"],
                    "netVolume": curr["netVolume"],
                    "midPrice": curr["midPrice"],
                    "bidDelta": curr["bidVolume"] - prev["bidVolume"],
                    "askDelta": curr["askVolume"] - prev["askVolume"],
                    "netDelta": curr["netVolume"] - prev["netVolume"],
                    "priceDelta": curr["midPrice"] - prev["midPrice"],
                }
            )
        
        # Analysis
        if deltas:
            avg_net = sum(d["netVolume"] for d in deltas) / len(deltas)
            max_net = max(d["netVolume"] for d in deltas)
            min_net = min(d["netVolume"] for d in deltas)
            
            # Trend detection
            recent_deltas = deltas[-10:] if len(deltas) >= 10 else deltas
            bid_trend = sum(1 for d in recent_deltas if d["bidDelta"] > 0) / len(recent_deltas)
            ask_trend = sum(1 for d in recent_deltas if d["askDelta"] > 0) / len(recent_deltas)
            
            analysis = {
                "avgNetVolume": avg_net,
                "maxNetVolume": max_net,
                "minNetVolume": min_net,
                "currentNetVolume": deltas[-1]["netVolume"] if deltas else 0,
                "bidTrendStrength": bid_trend,
                "askTrendStrength": ask_trend,
                "dominantSide": "bids" if avg_net > 0 else "asks" if avg_net < 0 else "neutral",
                "snapshotCount": len(rows),
                "downsampledCount": len(points),
            }
        else:
            analysis = None
        
        return {
            "symbol": symbol,
            "timestamp": timestamp_ms(),
            "lookbackSeconds": lookback_seconds,
            "percentRange": percent_range or self.settings.orderbook_depth_percent,
            "granularitySec": granularity_sec,
            "maxPoints": max_points,
            "snapshots": deltas[-max_points:],
            "analysis": analysis,
        }
    
    def calculate_depth_ratio(self, symbol: str) -> float | None:
        """Calculate current bid/ask depth ratio.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Ratio (>1 means more bids, <1 means more asks)
        """
        snapshot = self.get_latest_snapshot(symbol)
        
        if snapshot is None or snapshot.ask_volume == 0:
            return None
        
        return snapshot.bid_volume / snapshot.ask_volume
    
    def get_depth_summary(self, symbol: str) -> dict[str, Any] | None:
        """Get current depth summary.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Depth summary dict
        """
        snapshot = self.get_latest_snapshot(symbol)
        
        if snapshot is None:
            return None
        
        ratio = self.calculate_depth_ratio(symbol)
        
        return {
            "symbol": symbol,
            "timestamp": snapshot.timestamp,
            "midPrice": snapshot.mid_price,
            "bidVolume": snapshot.bid_volume,
            "askVolume": snapshot.ask_volume,
            "netVolume": snapshot.net_volume,
            "bidAskRatio": ratio,
            "percentRange": snapshot.percent_range,
            "dominantSide": "bids" if snapshot.net_volume > 0 else "asks" if snapshot.net_volume < 0 else "neutral",
        }
