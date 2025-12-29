"""Footprint bar calculator."""

from typing import Any
from collections import defaultdict
from dataclasses import dataclass, field

from src.data.storage import DataStorage
from src.config import get_settings
from src.utils import (
    get_logger,
    timestamp_ms,
    round_to_tick,
    get_timeframe_ms,
    align_timestamp_to_timeframe,
)


@dataclass
class FootprintLevel:
    """Single price level in a footprint bar."""
    price: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    
    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume
    
    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume


@dataclass
class FootprintBar:
    """Footprint bar containing trade data aggregated by price level."""
    symbol: str
    timeframe: str
    timestamp: int  # Bar start time
    levels: dict[float, FootprintLevel] = field(default_factory=dict)
    open_price: float | None = None
    close_price: float | None = None
    
    @property
    def total_buy_volume(self) -> float:
        return sum(level.buy_volume for level in self.levels.values())
    
    @property
    def total_sell_volume(self) -> float:
        return sum(level.sell_volume for level in self.levels.values())
    
    @property
    def total_volume(self) -> float:
        return sum(level.total_volume for level in self.levels.values())
    
    @property
    def delta(self) -> float:
        return self.total_buy_volume - self.total_sell_volume
    
    @property
    def max_delta_price(self) -> float | None:
        if not self.levels:
            return None
        return max(self.levels.keys(), key=lambda p: self.levels[p].delta)
    
    @property
    def min_delta_price(self) -> float | None:
        if not self.levels:
            return None
        return min(self.levels.keys(), key=lambda p: self.levels[p].delta)
    
    @property
    def poc_price(self) -> float | None:
        """Price level with highest volume (Point of Control)."""
        if not self.levels:
            return None
        return max(self.levels.keys(), key=lambda p: self.levels[p].total_volume)
    
    @property
    def high(self) -> float | None:
        if not self.levels:
            return None
        return max(self.levels.keys())
    
    @property
    def low(self) -> float | None:
        if not self.levels:
            return None
        return min(self.levels.keys())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        levels_list = [
            {
                "price": price,
                "buyVolume": level.buy_volume,
                "sellVolume": level.sell_volume,
                "delta": level.delta,
                "totalVolume": level.total_volume,
                "tradeCount": level.trade_count,
            }
            for price, level in sorted(self.levels.items(), reverse=True)
        ]
        
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "open": self.open_price,
            "high": self.high,
            "low": self.low,
            "close": self.close_price,
            "buyVolume": self.total_buy_volume,
            "sellVolume": self.total_sell_volume,
            "totalVolume": self.total_volume,
            "delta": self.delta,
            "maxDeltaPrice": self.max_delta_price,
            "minDeltaPrice": self.min_delta_price,
            "pocPrice": self.poc_price,
            "levels": levels_list,
            "levelCount": len(self.levels),
        }


class FootprintCalculator:
    """Calculate footprint bars from trade data."""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.settings = get_settings()
        self.logger = get_logger("indicators.footprint")
        
        # In-memory current bars by timeframe
        self._current_bars: dict[str, dict[str, FootprintBar]] = {}  # symbol -> {timeframe -> bar}
    
    def _get_tick_size(self, symbol: str) -> float:
        """Get tick size for symbol."""
        return self.settings.get_tick_size(symbol)
    
    def _get_or_create_bar(
        self,
        symbol: str,
        timeframe: str,
        timestamp: int,
    ) -> FootprintBar:
        """Get or create footprint bar for given symbol/timeframe/time."""
        symbol = symbol.upper()
        bar_start = align_timestamp_to_timeframe(timestamp, timeframe)
        
        if symbol not in self._current_bars:
            self._current_bars[symbol] = {}
        
        if timeframe not in self._current_bars[symbol]:
            self._current_bars[symbol][timeframe] = FootprintBar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=bar_start,
            )
        
        bar = self._current_bars[symbol][timeframe]
        
        # Check if we need a new bar
        if bar.timestamp != bar_start:
            # Save old bar to storage (for 1m bars)
            if timeframe == "1m":
                self._save_bar_to_storage(bar)
            
            # Create new bar
            bar = FootprintBar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=bar_start,
            )
            self._current_bars[symbol][timeframe] = bar
        
        return bar
    
    def _save_bar_to_storage(self, bar: FootprintBar) -> None:
        """Save footprint bar to storage (fire and forget)."""
        import asyncio
        
        async def _save():
            for price, level in bar.levels.items():
                await self.storage.upsert_footprint(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    price_level=price,
                    buy_volume=level.buy_volume,
                    sell_volume=level.sell_volume,
                    trade_count=level.trade_count,
                )
        
        try:
            asyncio.create_task(_save())
        except RuntimeError:
            pass  # No event loop
    
    async def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        is_buyer_maker: bool,
        timestamp: int,
    ) -> None:
        """Update footprint bars with new trade data.
        
        Args:
            symbol: Trading pair symbol
            price: Trade price
            volume: Trade volume
            is_buyer_maker: True if taker was seller
            timestamp: Trade timestamp in milliseconds
        """
        symbol = symbol.upper()
        tick_size = self._get_tick_size(symbol)
        price_level = round_to_tick(price, tick_size)
        
        # Update bars for all supported timeframes
        for timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            bar = self._get_or_create_bar(symbol, timeframe, timestamp)

            # Track open/close using the first/last trade price in this bar
            if bar.open_price is None:
                bar.open_price = price
            bar.close_price = price

            if price_level not in bar.levels:
                bar.levels[price_level] = FootprintLevel(price=price_level)
            
            level = bar.levels[price_level]
            level.trade_count += 1
            
            if is_buyer_maker:
                level.sell_volume += volume
            else:
                level.buy_volume += volume
    
    def get_current_bar(self, symbol: str, timeframe: str) -> FootprintBar | None:
        """Get current (developing) footprint bar."""
        symbol = symbol.upper()
        
        if symbol not in self._current_bars:
            return None
        
        return self._current_bars[symbol].get(timeframe)
    
    async def get_footprint_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: int,
        end_time: int,
    ) -> list[dict[str, Any]]:
        """Get footprint bars for a time range.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Bar timeframe
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            List of footprint bar dictionaries
        """
        symbol = symbol.upper()
        
        # Get from storage (1m resolution)
        rows = await self.storage.get_footprint_range(symbol, start_time, end_time)
        
        # Aggregate into bars
        bars: dict[int, FootprintBar] = {}
        tf_ms = get_timeframe_ms(timeframe)
        
        for row in rows:
            bar_start = (row["timestamp"] // tf_ms) * tf_ms
            
            if bar_start not in bars:
                bars[bar_start] = FootprintBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=bar_start,
                )
            
            bar = bars[bar_start]
            price = row["price_level"]
            
            if price not in bar.levels:
                bar.levels[price] = FootprintLevel(price=price)
            
            bar.levels[price].buy_volume += row["buy_volume"]
            bar.levels[price].sell_volume += row["sell_volume"]
            bar.levels[price].trade_count += row["trade_count"]
        
        # Include current bar if within range
        current = self.get_current_bar(symbol, timeframe)
        if current and start_time <= current.timestamp < end_time:
            bars[current.timestamp] = current
        
        # Sort and convert to list
        return [bar.to_dict() for bar in sorted(bars.values(), key=lambda b: b.timestamp)]


def aggregate_footprint_bars(
    bars: list[FootprintBar],
    target_timeframe: str,
) -> list[FootprintBar]:
    """Aggregate smaller timeframe bars into larger ones.
    
    Args:
        bars: List of footprint bars (should be 1m)
        target_timeframe: Target timeframe to aggregate to
    
    Returns:
        List of aggregated footprint bars
    """
    if not bars:
        return []
    
    tf_ms = get_timeframe_ms(target_timeframe)
    aggregated: dict[int, FootprintBar] = {}
    
    for bar in bars:
        agg_start = (bar.timestamp // tf_ms) * tf_ms
        
        if agg_start not in aggregated:
            aggregated[agg_start] = FootprintBar(
                symbol=bar.symbol,
                timeframe=target_timeframe,
                timestamp=agg_start,
            )
        
        agg_bar = aggregated[agg_start]
        
        for price, level in bar.levels.items():
            if price not in agg_bar.levels:
                agg_bar.levels[price] = FootprintLevel(price=price)
            
            agg_bar.levels[price].buy_volume += level.buy_volume
            agg_bar.levels[price].sell_volume += level.sell_volume
            agg_bar.levels[price].trade_count += level.trade_count
    
    return sorted(aggregated.values(), key=lambda b: b.timestamp)
