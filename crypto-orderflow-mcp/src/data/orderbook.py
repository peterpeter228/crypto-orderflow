"""Orderbook management with snapshot + diff consistency."""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from sortedcontainers import SortedDict

from src.binance.types import OrderbookLevel, OrderbookSnapshot, OrderbookUpdate
from src.binance.rest_client import BinanceRestClient
from src.config import get_settings
from src.utils import get_logger, timestamp_ms


@dataclass
class ManagedOrderbook:
    """Managed orderbook with consistency tracking."""
    symbol: str
    bids: SortedDict = field(default_factory=lambda: SortedDict())  # price -> qty, descending
    asks: SortedDict = field(default_factory=lambda: SortedDict())  # price -> qty, ascending
    last_update_id: int = 0
    last_update_time: int = 0
    is_valid: bool = False
    pending_updates: list = field(default_factory=list)


class OrderbookManager:
    """Manages orderbooks with snapshot + diff synchronization."""
    
    def __init__(self, rest_client: BinanceRestClient):
        self.settings = get_settings()
        self.rest_client = rest_client
        self.logger = get_logger("data.orderbook")
        
        self._orderbooks: dict[str, ManagedOrderbook] = {}
        self._lock = asyncio.Lock()
        self._rebuilding: set[str] = set()
    
    def _get_orderbook(self, symbol: str) -> ManagedOrderbook:
        """Get or create orderbook for symbol."""
        symbol = symbol.upper()
        if symbol not in self._orderbooks:
            self._orderbooks[symbol] = ManagedOrderbook(symbol=symbol)
        return self._orderbooks[symbol]
    
    async def initialize_orderbook(self, symbol: str) -> None:
        """Initialize orderbook from REST snapshot."""
        symbol = symbol.upper()
        
        async with self._lock:
            if symbol in self._rebuilding:
                return
            self._rebuilding.add(symbol)
        
        try:
            self.logger.info("initializing_orderbook", symbol=symbol)
            
            # Get snapshot from REST API
            snapshot = await self.rest_client.get_orderbook_snapshot(
                symbol, 
                limit=self.settings.orderbook_snapshot_limit
            )
            
            ob = self._get_orderbook(symbol)
            
            # Clear existing data
            ob.bids.clear()
            ob.asks.clear()
            
            # Load snapshot
            for bid in snapshot.bids:
                if bid.quantity > 0:
                    ob.bids[-bid.price] = bid.quantity  # Negative for descending sort
            
            for ask in snapshot.asks:
                if ask.quantity > 0:
                    ob.asks[ask.price] = ask.quantity
            
            ob.last_update_id = snapshot.last_update_id
            ob.last_update_time = timestamp_ms()
            ob.is_valid = True
            
            # Process any pending updates
            pending = ob.pending_updates
            ob.pending_updates = []
            
            for update in pending:
                await self._apply_update(symbol, update)
            
            self.logger.info("orderbook_initialized", 
                           symbol=symbol, 
                           last_update_id=ob.last_update_id,
                           bid_levels=len(ob.bids),
                           ask_levels=len(ob.asks))
            
        except Exception as e:
            self.logger.error("orderbook_init_failed", symbol=symbol, error=str(e))
            ob = self._get_orderbook(symbol)
            ob.is_valid = False
            raise
        
        finally:
            async with self._lock:
                self._rebuilding.discard(symbol)
    
    async def _apply_update(self, symbol: str, update: OrderbookUpdate) -> bool:
        """Apply depth update to orderbook.
        
        Returns True if update was applied successfully.
        """
        ob = self._get_orderbook(symbol)
        
        # Validate sequence
        if update.prev_last_update_id != ob.last_update_id:
            # Gap detected - need to rebuild
            if update.first_update_id > ob.last_update_id + 1:
                self.logger.warning("orderbook_gap", 
                                   symbol=symbol,
                                   expected=ob.last_update_id + 1,
                                   got=update.first_update_id)
                ob.is_valid = False
                return False
        
        # Apply bid updates
        for bid in update.bids:
            if bid.quantity == 0:
                ob.bids.pop(-bid.price, None)
            else:
                ob.bids[-bid.price] = bid.quantity
        
        # Apply ask updates
        for ask in update.asks:
            if ask.quantity == 0:
                ob.asks.pop(ask.price, None)
            else:
                ob.asks[ask.price] = ask.quantity
        
        ob.last_update_id = update.last_update_id
        ob.last_update_time = update.event_time
        
        return True
    
    async def process_update(self, update: OrderbookUpdate) -> None:
        """Process incoming orderbook update from WebSocket."""
        symbol = update.symbol.upper()
        ob = self._get_orderbook(symbol)
        
        # If not initialized, queue update
        if not ob.is_valid:
            ob.pending_updates.append(update)
            
            # Trigger rebuild if not already rebuilding
            if symbol not in self._rebuilding:
                task = asyncio.create_task(self.initialize_orderbook(symbol))
                # Suppress unhandled exception warning
                task.add_done_callback(lambda t: t.exception() if t.done() and not t.cancelled() else None)
            return
        
        # Apply update
        success = await self._apply_update(symbol, update)
        
        if not success:
            # Trigger rebuild
            task = asyncio.create_task(self.initialize_orderbook(symbol))
            # Suppress unhandled exception warning
            task.add_done_callback(lambda t: t.exception() if t.done() and not t.cancelled() else None)
    
    def get_orderbook(self, symbol: str) -> dict[str, Any] | None:
        """Get current orderbook state."""
        symbol = symbol.upper()
        
        if symbol not in self._orderbooks:
            return None
        
        ob = self._orderbooks[symbol]
        
        if not ob.is_valid:
            return None
        
        # Convert to lists
        bids = [{"price": -price, "quantity": qty} for price, qty in ob.bids.items()]
        asks = [{"price": price, "quantity": qty} for price, qty in ob.asks.items()]
        
        return {
            "symbol": symbol,
            "lastUpdateId": ob.last_update_id,
            "lastUpdateTime": ob.last_update_time,
            "bids": bids,
            "asks": asks,
            "isValid": ob.is_valid,
        }
    
    def get_best_bid_ask(self, symbol: str) -> tuple[float, float] | None:
        """Get best bid and ask prices."""
        symbol = symbol.upper()
        
        if symbol not in self._orderbooks:
            return None
        
        ob = self._orderbooks[symbol]
        
        if not ob.is_valid or not ob.bids or not ob.asks:
            return None
        
        best_bid = -ob.bids.peekitem(0)[0]  # First item (highest bid)
        best_ask = ob.asks.peekitem(0)[0]   # First item (lowest ask)
        
        return best_bid, best_ask
    
    def get_mid_price(self, symbol: str) -> float | None:
        """Get mid price."""
        result = self.get_best_bid_ask(symbol)
        if result is None:
            return None
        
        best_bid, best_ask = result
        return (best_bid + best_ask) / 2
    
    def get_depth_within_percent(
        self,
        symbol: str,
        percent: float,
    ) -> dict[str, float] | None:
        """Get total bid/ask volume within percent of mid price.
        
        Args:
            symbol: Trading pair symbol
            percent: Percentage range (e.g., 1.0 for ±1%)
        
        Returns:
            Dict with bid_volume, ask_volume, net_volume, mid_price
        """
        symbol = symbol.upper()
        
        if symbol not in self._orderbooks:
            return None
        
        ob = self._orderbooks[symbol]
        
        if not ob.is_valid or not ob.bids or not ob.asks:
            return None
        
        mid_price = self.get_mid_price(symbol)
        if mid_price is None:
            return None
        
        lower_bound = mid_price * (1 - percent / 100)
        upper_bound = mid_price * (1 + percent / 100)
        
        bid_volume = 0.0
        for neg_price, qty in ob.bids.items():
            price = -neg_price
            if price >= lower_bound:
                bid_volume += qty
            else:
                break  # Sorted descending, so no more valid prices
        
        ask_volume = 0.0
        for price, qty in ob.asks.items():
            if price <= upper_bound:
                ask_volume += qty
            else:
                break  # Sorted ascending, so no more valid prices
        
        return {
            "bidVolume": bid_volume,
            "askVolume": ask_volume,
            "netVolume": bid_volume - ask_volume,
            "midPrice": mid_price,
            "percentRange": percent,
            "timestamp": timestamp_ms(),
        }
    
    def is_valid(self, symbol: str) -> bool:
        """Check if orderbook is valid and synchronized."""
        symbol = symbol.upper()
        
        if symbol not in self._orderbooks:
            return False
        
        return self._orderbooks[symbol].is_valid
