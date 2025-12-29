"""Delta and Cumulative Volume Delta (CVD) calculator."""

from typing import Any
from dataclasses import dataclass, field
from collections import deque

from src.data.storage import DataStorage
from src.config import get_settings
from src.utils import (
    get_logger,
    timestamp_ms,
    get_timeframe_ms,
    align_timestamp_to_timeframe,
)


@dataclass
class DeltaBar:
    """Delta bar for a specific timeframe."""
    timestamp: int
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    
    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume
    
    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume
    
    @property
    def delta_percent(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return (self.delta / self.total_volume) * 100


class DeltaCVDCalculator:
    """Calculate delta bars and cumulative volume delta (CVD)."""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.settings = get_settings()
        self.logger = get_logger("indicators.delta")
        
        # CVD tracking
        self._cvd: dict[str, float] = {}  # symbol -> cumulative delta
        self._cvd_reset_time: dict[str, int] = {}
        
        # Recent delta bars for each timeframe
        self._delta_bars: dict[str, dict[str, deque[DeltaBar]]] = {}
        self._current_bars: dict[str, dict[str, DeltaBar]] = {}
    
    def _get_or_create_bar(
        self,
        symbol: str,
        timeframe: str,
        timestamp: int,
    ) -> DeltaBar:
        """Get or create delta bar."""
        symbol = symbol.upper()
        bar_start = align_timestamp_to_timeframe(timestamp, timeframe)
        
        if symbol not in self._current_bars:
            self._current_bars[symbol] = {}
            self._delta_bars[symbol] = {}
        
        if timeframe not in self._current_bars[symbol]:
            self._current_bars[symbol][timeframe] = DeltaBar(timestamp=bar_start)
            self._delta_bars[symbol][timeframe] = deque(maxlen=1000)
        
        bar = self._current_bars[symbol][timeframe]
        
        # Check if we need a new bar
        if bar.timestamp != bar_start:
            # Archive old bar
            self._delta_bars[symbol][timeframe].append(bar)
            
            # Create new bar
            bar = DeltaBar(timestamp=bar_start)
            self._current_bars[symbol][timeframe] = bar
        
        return bar
    
    async def update(
        self,
        symbol: str,
        volume: float,
        is_buyer_maker: bool,
        timestamp: int,
    ) -> None:
        """Update delta/CVD with new trade.
        
        Args:
            symbol: Trading pair symbol
            volume: Trade volume
            is_buyer_maker: True if taker was seller
            timestamp: Trade timestamp in milliseconds
        """
        symbol = symbol.upper()
        
        # Calculate delta for this trade
        delta = -volume if is_buyer_maker else volume
        
        # Update CVD
        if symbol not in self._cvd:
            self._cvd[symbol] = 0.0
            self._cvd_reset_time[symbol] = timestamp
        
        self._cvd[symbol] += delta
        
        # Update delta bars for all timeframes
        for timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            bar = self._get_or_create_bar(symbol, timeframe, timestamp)
            bar.trade_count += 1
            
            if is_buyer_maker:
                bar.sell_volume += volume
            else:
                bar.buy_volume += volume
    
    def get_cvd(self, symbol: str) -> float:
        """Get current CVD value."""
        return self._cvd.get(symbol.upper(), 0.0)
    
    def reset_cvd(self, symbol: str) -> None:
        """Reset CVD to zero."""
        symbol = symbol.upper()
        self._cvd[symbol] = 0.0
        self._cvd_reset_time[symbol] = timestamp_ms()
        self.logger.info("cvd_reset", symbol=symbol)
    
    def get_delta_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent delta bars.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Bar timeframe
            limit: Number of bars to return
        
        Returns:
            List of delta bar dictionaries
        """
        symbol = symbol.upper()
        
        if symbol not in self._delta_bars:
            return []
        
        if timeframe not in self._delta_bars[symbol]:
            return []
        
        bars = list(self._delta_bars[symbol][timeframe])[-limit:]
        
        # Include current bar
        if symbol in self._current_bars and timeframe in self._current_bars[symbol]:
            bars.append(self._current_bars[symbol][timeframe])
        
        return [
            {
                "timestamp": bar.timestamp,
                "buyVolume": bar.buy_volume,
                "sellVolume": bar.sell_volume,
                "delta": bar.delta,
                "deltaPercent": bar.delta_percent,
                "totalVolume": bar.total_volume,
                "tradeCount": bar.trade_count,
            }
            for bar in bars
        ]
    
    async def get_delta_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: int,
        end_time: int,
    ) -> dict[str, Any]:
        """Get delta statistics for a time range.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Bar timeframe
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            Delta statistics including delta sequence and CVD
        """
        symbol = symbol.upper()
        
        # Get footprint data from storage
        rows = await self.storage.get_footprint_range(symbol, start_time, end_time)
        
        # Aggregate into delta bars
        tf_ms = get_timeframe_ms(timeframe)
        bars: dict[int, DeltaBar] = {}
        
        for row in rows:
            bar_start = (row["timestamp"] // tf_ms) * tf_ms
            
            if bar_start not in bars:
                bars[bar_start] = DeltaBar(timestamp=bar_start)
            
            bars[bar_start].buy_volume += row["buy_volume"]
            bars[bar_start].sell_volume += row["sell_volume"]
            bars[bar_start].trade_count += row["trade_count"]
        
        # Sort bars
        sorted_bars = sorted(bars.values(), key=lambda b: b.timestamp)
        
        # Calculate CVD sequence
        cvd = 0.0
        delta_sequence = []
        cvd_sequence = []
        
        for bar in sorted_bars:
            cvd += bar.delta
            delta_sequence.append({
                "timestamp": bar.timestamp,
                "delta": bar.delta,
                "deltaPercent": bar.delta_percent,
            })
            cvd_sequence.append({
                "timestamp": bar.timestamp,
                "cvd": cvd,
            })
        
        # Summary statistics
        total_buy = sum(b.buy_volume for b in sorted_bars)
        total_sell = sum(b.sell_volume for b in sorted_bars)
        total_delta = total_buy - total_sell
        
        positive_delta_bars = sum(1 for b in sorted_bars if b.delta > 0)
        negative_delta_bars = sum(1 for b in sorted_bars if b.delta < 0)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "startTime": start_time,
            "endTime": end_time,
            "timestamp": timestamp_ms(),
            "summary": {
                "totalBuyVolume": total_buy,
                "totalSellVolume": total_sell,
                "totalDelta": total_delta,
                "totalVolume": total_buy + total_sell,
                "deltaPercent": (total_delta / (total_buy + total_sell) * 100) if (total_buy + total_sell) > 0 else 0,
                "positiveDeltaBars": positive_delta_bars,
                "negativeDeltaBars": negative_delta_bars,
                "barCount": len(sorted_bars),
            },
            "deltaSequence": delta_sequence,
            "cvdSequence": cvd_sequence,
            "currentCVD": self.get_cvd(symbol),
        }
    
    def get_current_bar(self, symbol: str, timeframe: str) -> DeltaBar | None:
        """Get current (developing) delta bar."""
        symbol = symbol.upper()
        
        if symbol not in self._current_bars:
            return None
        
        return self._current_bars[symbol].get(timeframe)
