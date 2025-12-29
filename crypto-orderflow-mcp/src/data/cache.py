"""In-memory cache for real-time data."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.binance.types import Trade, Liquidation, MarkPrice
from src.config import get_settings
from src.utils import get_logger, timestamp_ms


@dataclass
class SymbolCache:
    """Cache for a single symbol's real-time data."""
    symbol: str
    
    # Latest data
    last_price: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: int = 0
    
    # 24h stats
    high_24h: float = 0.0
    low_24h: float = 0.0
    volume_24h: float = 0.0
    quote_volume_24h: float = 0.0
    
    # Open Interest
    open_interest: float = 0.0
    open_interest_notional: float = 0.0
    
    # Recent trades (for short-term analysis)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    # Liquidations cache
    liquidations: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # CVD tracking
    cvd: float = 0.0
    cvd_reset_time: int = 0
    
    # Last update times
    last_trade_time: int = 0
    last_mark_price_time: int = 0
    last_oi_update_time: int = 0


class MemoryCache:
    """In-memory cache manager for all symbols."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("data.cache")
        self._caches: dict[str, SymbolCache] = {}
        self._lock = asyncio.Lock()
    
    def get_cache(self, symbol: str) -> SymbolCache:
        """Get or create cache for symbol."""
        symbol = symbol.upper()
        if symbol not in self._caches:
            self._caches[symbol] = SymbolCache(symbol=symbol)
        return self._caches[symbol]
    
    async def update_trade(self, trade: Trade) -> None:
        """Update cache with new trade."""
        cache = self.get_cache(trade.symbol)
        
        cache.last_price = trade.price
        cache.last_trade_time = trade.timestamp
        cache.recent_trades.append(trade)
        
        # Update CVD
        delta = trade.buy_volume - trade.sell_volume
        cache.cvd += delta
    
    async def update_mark_price(self, data: MarkPrice) -> None:
        """Update mark price data."""
        cache = self.get_cache(data.symbol)
        
        cache.mark_price = data.mark_price
        cache.index_price = data.index_price
        cache.funding_rate = data.funding_rate
        cache.next_funding_time = data.next_funding_time
        cache.last_mark_price_time = data.timestamp
    
    async def add_liquidation(self, liq: Liquidation) -> None:
        """Add liquidation to cache."""
        cache = self.get_cache(liq.symbol)
        cache.liquidations.append(liq)
    
    async def update_ticker(
        self,
        symbol: str,
        high_24h: float,
        low_24h: float,
        volume_24h: float,
        quote_volume_24h: float,
    ) -> None:
        """Update 24h ticker stats."""
        cache = self.get_cache(symbol)
        cache.high_24h = high_24h
        cache.low_24h = low_24h
        cache.volume_24h = volume_24h
        cache.quote_volume_24h = quote_volume_24h
    
    async def update_open_interest(
        self,
        symbol: str,
        oi: float,
        oi_notional: float,
    ) -> None:
        """Update open interest."""
        cache = self.get_cache(symbol)
        cache.open_interest = oi
        cache.open_interest_notional = oi_notional
        cache.last_oi_update_time = timestamp_ms()
    
    def get_recent_trades(
        self,
        symbol: str,
        since: int | None = None,
    ) -> list[Trade]:
        """Get recent trades, optionally filtered by time."""
        cache = self.get_cache(symbol)
        
        if since is None:
            return list(cache.recent_trades)
        
        return [t for t in cache.recent_trades if t.timestamp >= since]
    
    def get_liquidations(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> list[Liquidation]:
        """Get recent liquidations."""
        cache = self.get_cache(symbol)
        
        if limit is None:
            return list(cache.liquidations)
        
        return list(cache.liquidations)[-limit:]
    
    def get_snapshot(self, symbol: str) -> dict[str, Any]:
        """Get full snapshot for a symbol."""
        cache = self.get_cache(symbol)
        
        return {
            "symbol": cache.symbol,
            "exchange": "binance",
            "marketType": "linear_perpetual",
            "timestamp": timestamp_ms(),
            "lastPrice": cache.last_price,
            "markPrice": cache.mark_price,
            "indexPrice": cache.index_price,
            "high24h": cache.high_24h,
            "low24h": cache.low_24h,
            "volume24h": cache.volume_24h,
            "quoteVolume24h": cache.quote_volume_24h,
            "fundingRate": cache.funding_rate,
            "nextFundingTime": cache.next_funding_time,
            "openInterest": cache.open_interest,
            "openInterestNotional": cache.open_interest_notional,
            "cvd": cache.cvd,
            "lastTradeTime": cache.last_trade_time,
            "priceUnit": "USDT",
            "volumeUnit": symbol.replace("USDT", ""),
        }
    
    def reset_cvd(self, symbol: str) -> None:
        """Reset CVD to zero."""
        cache = self.get_cache(symbol)
        cache.cvd = 0.0
        cache.cvd_reset_time = timestamp_ms()

    def set_cvd(self, symbol: str, value: float, reset_time: int | None = None) -> None:
        """Set CVD to a specific value.

        This is primarily used after historical backfill so that `get_market_snapshot` has a
        meaningful CVD immediately after startup.
        """
        cache = self.get_cache(symbol)
        cache.cvd = float(value)
        if reset_time is not None:
            cache.cvd_reset_time = int(reset_time)
    
    def get_all_symbols(self) -> list[str]:
        """Get all cached symbols."""
        return list(self._caches.keys())
