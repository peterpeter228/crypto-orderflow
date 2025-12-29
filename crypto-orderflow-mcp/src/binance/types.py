"""Data types for Binance API responses."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Trade:
    """Individual trade from aggTrades stream."""
    id: int
    symbol: str
    price: float
    quantity: float
    timestamp: int  # milliseconds
    is_buyer_maker: bool  # True = sell, False = buy (taker side)
    
    @property
    def side(self) -> Literal["buy", "sell"]:
        """Get taker side."""
        return "sell" if self.is_buyer_maker else "buy"
    
    @property
    def buy_volume(self) -> float:
        return 0.0 if self.is_buyer_maker else self.quantity
    
    @property
    def sell_volume(self) -> float:
        return self.quantity if self.is_buyer_maker else 0.0
    
    @property
    def notional(self) -> float:
        """Trade notional value in quote currency (USDT)."""
        return self.price * self.quantity


@dataclass
class AggTrade:
    """Aggregated trade from REST API."""
    agg_trade_id: int
    symbol: str
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    timestamp: int
    is_buyer_maker: bool
    
    def to_trade(self) -> Trade:
        """Convert to Trade object."""
        return Trade(
            id=self.agg_trade_id,
            symbol=self.symbol,
            price=self.price,
            quantity=self.quantity,
            timestamp=self.timestamp,
            is_buyer_maker=self.is_buyer_maker,
        )


@dataclass
class OrderbookLevel:
    """Single orderbook price level."""
    price: float
    quantity: float


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot."""
    symbol: str
    last_update_id: int
    timestamp: int
    bids: list[OrderbookLevel]  # Sorted by price descending
    asks: list[OrderbookLevel]  # Sorted by price ascending


@dataclass
class OrderbookUpdate:
    """Orderbook depth update from WebSocket."""
    symbol: str
    event_time: int
    transaction_time: int
    first_update_id: int
    last_update_id: int
    prev_last_update_id: int
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]


@dataclass
class MarkPrice:
    """Mark price data."""
    symbol: str
    mark_price: float
    index_price: float
    estimated_settle_price: float
    funding_rate: float
    next_funding_time: int
    timestamp: int


@dataclass
class Liquidation:
    """Liquidation (forceOrder) event."""
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: str
    time_in_force: str
    original_qty: float
    price: float
    avg_price: float
    order_status: str
    last_filled_qty: float
    filled_qty: float
    timestamp: int
    
    @property
    def is_long_liquidation(self) -> bool:
        """True if this is a long position being liquidated (forced sell)."""
        return self.side == "SELL"
    
    @property
    def notional(self) -> float:
        """Notional value of the liquidation."""
        return self.avg_price * self.filled_qty if self.avg_price > 0 else self.price * self.original_qty


@dataclass
class FundingRate:
    """Funding rate information."""
    symbol: str
    funding_rate: float
    funding_time: int
    mark_price: float | None = None


@dataclass
class OpenInterest:
    """Open interest data."""
    symbol: str
    open_interest: float  # In contracts/coins
    open_interest_notional: float  # In USDT
    timestamp: int


@dataclass
class OpenInterestHist:
    """Historical open interest entry."""
    symbol: str
    sum_open_interest: float
    sum_open_interest_value: float
    timestamp: int


@dataclass
class Ticker24h:
    """24h ticker statistics."""
    symbol: str
    price_change: float
    price_change_percent: float
    weighted_avg_price: float
    last_price: float
    last_qty: float
    open_price: float
    high_price: float
    low_price: float
    volume: float  # Base asset volume
    quote_volume: float  # Quote asset volume (USDT)
    open_time: int
    close_time: int
    first_trade_id: int
    last_trade_id: int
    trade_count: int


@dataclass
class Kline:
    """Candlestick/Kline data."""
    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trade_count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float
