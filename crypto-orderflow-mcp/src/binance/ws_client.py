"""Binance WebSocket client for USD-M Futures real-time data."""

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from src.config import get_settings
from src.utils import get_logger, timestamp_ms
from .types import (
    Trade,
    OrderbookUpdate,
    OrderbookLevel,
    MarkPrice,
    Liquidation,
)


class BinanceWebSocketClient:
    """WebSocket client for Binance USD-M Futures streams."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.binance_ws_url
        self.logger = get_logger("binance.ws")
        
        self._connections: dict[str, ClientConnection] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._reconnect_attempts: dict[str, int] = {}
        
        # Callbacks
        self._trade_callbacks: list[Callable[[Trade], Coroutine[Any, Any, None]]] = []
        self._orderbook_callbacks: list[Callable[[OrderbookUpdate], Coroutine[Any, Any, None]]] = []
        self._mark_price_callbacks: list[Callable[[MarkPrice], Coroutine[Any, Any, None]]] = []
        self._liquidation_callbacks: list[Callable[[Liquidation], Coroutine[Any, Any, None]]] = []
    
    def on_trade(self, callback: Callable[[Trade], Coroutine[Any, Any, None]]) -> None:
        """Register trade callback."""
        self._trade_callbacks.append(callback)
    
    def on_orderbook(self, callback: Callable[[OrderbookUpdate], Coroutine[Any, Any, None]]) -> None:
        """Register orderbook update callback."""
        self._orderbook_callbacks.append(callback)
    
    def on_mark_price(self, callback: Callable[[MarkPrice], Coroutine[Any, Any, None]]) -> None:
        """Register mark price callback."""
        self._mark_price_callbacks.append(callback)
    
    def on_liquidation(self, callback: Callable[[Liquidation], Coroutine[Any, Any, None]]) -> None:
        """Register liquidation callback."""
        self._liquidation_callbacks.append(callback)
    
    async def _emit_trade(self, trade: Trade) -> None:
        """Emit trade to all callbacks."""
        for callback in self._trade_callbacks:
            try:
                await callback(trade)
            except Exception as e:
                self.logger.error("trade_callback_error", error=str(e))
    
    async def _emit_orderbook(self, update: OrderbookUpdate) -> None:
        """Emit orderbook update to all callbacks."""
        for callback in self._orderbook_callbacks:
            try:
                await callback(update)
            except Exception as e:
                self.logger.error("orderbook_callback_error", error=str(e))
    
    async def _emit_mark_price(self, data: MarkPrice) -> None:
        """Emit mark price to all callbacks."""
        for callback in self._mark_price_callbacks:
            try:
                await callback(data)
            except Exception as e:
                self.logger.error("mark_price_callback_error", error=str(e))
    
    async def _emit_liquidation(self, liq: Liquidation) -> None:
        """Emit liquidation to all callbacks."""
        for callback in self._liquidation_callbacks:
            try:
                await callback(liq)
            except Exception as e:
                self.logger.error("liquidation_callback_error", error=str(e))
    
    def _parse_agg_trade(self, data: dict[str, Any]) -> Trade:
        """Parse aggregated trade message."""
        return Trade(
            id=data["a"],
            symbol=data["s"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            timestamp=data["T"],
            is_buyer_maker=data["m"],
        )
    
    def _parse_depth_update(self, data: dict[str, Any]) -> OrderbookUpdate:
        """Parse depth update message."""
        return OrderbookUpdate(
            symbol=data["s"],
            event_time=data["E"],
            transaction_time=data["T"],
            first_update_id=data["U"],
            last_update_id=data["u"],
            prev_last_update_id=data["pu"],
            bids=[OrderbookLevel(float(b[0]), float(b[1])) for b in data["b"]],
            asks=[OrderbookLevel(float(a[0]), float(a[1])) for a in data["a"]],
        )
    
    def _parse_mark_price(self, data: dict[str, Any]) -> MarkPrice:
        """Parse mark price message."""
        return MarkPrice(
            symbol=data["s"],
            mark_price=float(data["p"]),
            index_price=float(data["i"]),
            estimated_settle_price=float(data.get("P", 0)),
            funding_rate=float(data["r"]),
            next_funding_time=data["T"],
            timestamp=data["E"],
        )
    
    def _parse_liquidation(self, data: dict[str, Any]) -> Liquidation:
        """Parse liquidation (forceOrder) message."""
        order = data["o"]
        return Liquidation(
            symbol=order["s"],
            side=order["S"],
            order_type=order["o"],
            time_in_force=order["f"],
            original_qty=float(order["q"]),
            price=float(order["p"]),
            avg_price=float(order["ap"]),
            order_status=order["X"],
            last_filled_qty=float(order["l"]),
            filled_qty=float(order["z"]),
            timestamp=order["T"],
        )
    
    async def _handle_message(self, stream_name: str, data: dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        event_type = data.get("e")
        
        if event_type == "aggTrade":
            trade = self._parse_agg_trade(data)
            await self._emit_trade(trade)
        
        elif event_type == "depthUpdate":
            update = self._parse_depth_update(data)
            await self._emit_orderbook(update)
        
        elif event_type == "markPriceUpdate":
            mark_price = self._parse_mark_price(data)
            await self._emit_mark_price(mark_price)
        
        elif event_type == "forceOrder":
            liq = self._parse_liquidation(data)
            await self._emit_liquidation(liq)
    
    async def _connect_stream(self, stream_name: str) -> None:
        """Connect to a single stream with auto-reconnect."""
        url = f"{self.base_url}/ws/{stream_name}"
        self._reconnect_attempts[stream_name] = 0
        
        while self._running:
            try:
                self.logger.info("connecting", stream=stream_name)
                
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._connections[stream_name] = ws
                    self._reconnect_attempts[stream_name] = 0
                    self.logger.info("connected", stream=stream_name)
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._handle_message(stream_name, data)
                        except json.JSONDecodeError as e:
                            self.logger.error("json_decode_error", stream=stream_name, error=str(e))
                
            except websockets.ConnectionClosed as e:
                self.logger.warning("connection_closed", stream=stream_name, code=e.code, reason=e.reason)
            
            except Exception as e:
                self.logger.error("connection_error", stream=stream_name, error=str(e))
            
            finally:
                self._connections.pop(stream_name, None)
            
            # Reconnect logic
            if self._running:
                self._reconnect_attempts[stream_name] += 1
                if self._reconnect_attempts[stream_name] > self.settings.ws_max_reconnect_attempts:
                    self.logger.error("max_reconnect_exceeded", stream=stream_name)
                    break
                
                delay = self.settings.ws_reconnect_delay_sec * min(self._reconnect_attempts[stream_name], 5)
                self.logger.info("reconnecting", stream=stream_name, delay=delay)
                await asyncio.sleep(delay)
    
    async def _connect_combined_stream(self, streams: list[str]) -> None:
        """Connect to combined stream with multiple subscriptions."""
        stream_path = "/".join(streams)
        url = f"{self.base_url}/stream?streams={stream_path}"
        stream_name = "combined"
        self._reconnect_attempts[stream_name] = 0
        
        while self._running:
            try:
                self.logger.info("connecting_combined", streams=streams)
                
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._connections[stream_name] = ws
                    self._reconnect_attempts[stream_name] = 0
                    self.logger.info("connected_combined", streams=streams)
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        try:
                            wrapper = json.loads(message)
                            stream = wrapper.get("stream", "")
                            data = wrapper.get("data", {})
                            await self._handle_message(stream, data)
                        except json.JSONDecodeError as e:
                            self.logger.error("json_decode_error", error=str(e))
                
            except websockets.ConnectionClosed as e:
                self.logger.warning("connection_closed", code=e.code, reason=e.reason)
            
            except Exception as e:
                self.logger.error("connection_error", error=str(e))
            
            finally:
                self._connections.pop(stream_name, None)
            
            # Reconnect logic
            if self._running:
                self._reconnect_attempts[stream_name] += 1
                if self._reconnect_attempts[stream_name] > self.settings.ws_max_reconnect_attempts:
                    self.logger.error("max_reconnect_exceeded")
                    break
                
                delay = self.settings.ws_reconnect_delay_sec * min(self._reconnect_attempts[stream_name], 5)
                self.logger.info("reconnecting", delay=delay)
                await asyncio.sleep(delay)
    
    async def start(self, symbols: list[str]) -> None:
        """Start WebSocket connections for given symbols.
        
        Subscribes to:
        - aggTrade: Real-time trades
        - depth@100ms: Orderbook updates
        - markPrice@1s: Mark price updates
        - forceOrder: Liquidations
        """
        self._running = True
        
        # Build stream list
        streams: list[str] = []
        for symbol in symbols:
            s = symbol.lower()
            streams.extend([
                f"{s}@aggTrade",
                f"{s}@depth@100ms",
                f"{s}@markPrice@1s",
                f"{s}@forceOrder",
            ])
        
        # Use combined stream for efficiency
        task = asyncio.create_task(self._connect_combined_stream(streams))
        self._tasks.append(task)
        
        self.logger.info("started", symbols=symbols, stream_count=len(streams))
    
    async def stop(self) -> None:
        """Stop all WebSocket connections."""
        self._running = False
        
        # Close all connections
        for stream_name, ws in list(self._connections.items()):
            try:
                await ws.close()
            except Exception as e:
                self.logger.error("close_error", stream=stream_name, error=str(e))
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        self._connections.clear()
        self.logger.info("stopped")
    
    @property
    def is_connected(self) -> bool:
        """Check if any connection is active."""
        return len(self._connections) > 0 and self._running
