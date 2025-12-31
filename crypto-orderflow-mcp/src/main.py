"""Main entry point for Crypto Orderflow MCP Server."""

import asyncio
import signal
import sys
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config import get_settings
from src.utils import setup_logging, get_logger, timestamp_ms
from src.utils.helpers import get_day_start_ms
from src.binance import BinanceRestClient, BinanceWebSocketClient, Trade, Liquidation, MarkPrice, OrderbookUpdate
from src.data import DataStorage, MemoryCache, OrderbookManager
from src.data.backfill import AggTradesBackfiller, get_required_backfill_range
from src.indicators import (
    VWAPCalculator,
    VolumeProfileCalculator,
    SessionLevelsCalculator,
    FootprintCalculator,
    DeltaCVDCalculator,
    ImbalanceDetector,
    DepthDeltaCalculator,
    OrderbookHeatmapSampler,
    TPOProfileCalculator,
    HeatmapMetadataSampler,
)
from src.mcp import create_mcp_server, MCPTools
from src import __version__


class CryptoOrderflowServer:
    """Main server orchestrating all components."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("server")
        
        # Clients
        self.rest_client = BinanceRestClient()
        self.ws_client = BinanceWebSocketClient()
        
        # Data layer
        self.storage = DataStorage()
        self.cache = MemoryCache()
        self.orderbook = OrderbookManager(self.rest_client)
        
        # Indicators
        self.vwap = VWAPCalculator(self.storage)
        self.volume_profile = VolumeProfileCalculator(self.storage)
        self.tpo_profile = TPOProfileCalculator(self.storage)
        self.session_levels = SessionLevelsCalculator(self.storage)
        self.footprint = FootprintCalculator(self.storage)
        self.delta_cvd = DeltaCVDCalculator(self.storage)
        self.imbalance = ImbalanceDetector()
        self.depth_delta = DepthDeltaCalculator(self.storage, self.orderbook)
        self.heatmap = OrderbookHeatmapSampler(
            storage=self.storage,
            orderbook=self.orderbook,
            settings=self.settings,
            logger=self.logger,
        )
        self.heatmap_meta = HeatmapMetadataSampler(
            storage=self.storage,
            cache=self.cache,
            settings=self.settings,
            logger=self.logger,
        )
        
        # MCP Tools
        self.tools = MCPTools(
            cache=self.cache,
            storage=self.storage,
            orderbook=self.orderbook,
            rest_client=self.rest_client,
            vwap=self.vwap,
            volume_profile=self.volume_profile,
            tpo_profile=self.tpo_profile,
            session_levels=self.session_levels,
            footprint=self.footprint,
            delta_cvd=self.delta_cvd,
            imbalance=self.imbalance,
            depth_delta=self.depth_delta,
            heatmap=self.heatmap,
        )
        
        # Tasks
        self._background_tasks: list[asyncio.Task] = []
        self._backfill_task: asyncio.Task | None = None
        self._running = False

    async def _prime_cache_from_storage(self) -> None:
        """Prime in-memory values from persisted aggregates.

        This matters on restart:
        - VWAP/VolumeProfile are persisted and can rebuild on request.
        - Snapshot CVD would reset to 0 without a backfill; we restore it from cached day aggregates.
        """
        today = get_day_start_ms(timestamp_ms())
        for symbol in self.settings.symbol_list:
            try:
                cvd = await self.storage.get_day_cvd(symbol, today)
                if cvd is not None:
                    self.cache.set_cvd(symbol, float(cvd), reset_time=today)
                    self.logger.info("cvd_primed_from_storage", symbol=symbol, cvd=float(cvd), day_start=today)
            except Exception as e:
                self.logger.debug("cvd_prime_failed", symbol=symbol, error=str(e))
    
    async def _handle_trade(self, trade: Trade) -> None:
        """Process incoming trade from WebSocket."""
        # Update cache
        await self.cache.update_trade(trade)
        
        # Update indicators
        await self.vwap.update(
            symbol=trade.symbol,
            price=trade.price,
            volume=trade.quantity,
            timestamp=trade.timestamp,
        )
        
        await self.volume_profile.update(
            symbol=trade.symbol,
            price=trade.price,
            volume=trade.quantity,
            buy_volume=trade.buy_volume,
            sell_volume=trade.sell_volume,
            timestamp=trade.timestamp,
        )
        
        await self.session_levels.update(
            symbol=trade.symbol,
            price=trade.price,
            volume=trade.quantity,
            timestamp=trade.timestamp,
        )
        
        await self.footprint.update(
            symbol=trade.symbol,
            price=trade.price,
            volume=trade.quantity,
            is_buyer_maker=trade.is_buyer_maker,
            timestamp=trade.timestamp,
        )
        
        await self.delta_cvd.update(
            symbol=trade.symbol,
            volume=trade.quantity,
            is_buyer_maker=trade.is_buyer_maker,
            timestamp=trade.timestamp,
        )
    
    async def _handle_orderbook_update(self, update: OrderbookUpdate) -> None:
        """Process orderbook update from WebSocket."""
        await self.orderbook.process_update(update)
    
    async def _handle_mark_price(self, data: MarkPrice) -> None:
        """Process mark price update from WebSocket."""
        await self.cache.update_mark_price(data)
    
    async def _handle_liquidation(self, liq: Liquidation) -> None:
        """Process liquidation event from WebSocket."""
        await self.cache.add_liquidation(liq)
        self.logger.info("liquidation", 
                        symbol=liq.symbol, 
                        side=liq.side, 
                        qty=liq.original_qty,
                        price=liq.price)
    
    async def _depth_delta_task(self) -> None:
        """Periodically take depth delta snapshots."""
        while self._running:
            try:
                for symbol in self.settings.symbol_list:
                    if await self.depth_delta.should_take_snapshot(symbol):
                        await self.depth_delta.take_snapshot(symbol)
                
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("depth_delta_task_error", error=str(e))
                await asyncio.sleep(5)

    async def _heatmap_task(self) -> None:
        """Periodically snapshot a binned orderbook ladder for heatmap visualizations."""
        if not getattr(self.settings, "heatmap_enabled", False):
            # Should normally never be scheduled.
            while self._running:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    break
            return

        while self._running:
            try:
                for symbol in self.settings.symbol_list:
                    await self.heatmap.maybe_snapshot(symbol)

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("heatmap_task_error", error=str(e))
                await asyncio.sleep(5)

    async def _heatmap_metadata_task(self) -> None:
        """Lightweight metadata sampler for heatmap coverage."""
        if not getattr(self.settings, "heatmap_enabled", False):
            while self._running:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    break
            return

        while self._running:
            try:
                for symbol in self.settings.symbol_list:
                    await self.heatmap_meta.maybe_sample(symbol)
                await asyncio.sleep(max(1, self.settings.heatmap_sample_interval_ms / 1000))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug("heatmap_metadata_task_error", error=str(e))
                await asyncio.sleep(5)
    
    async def _ticker_update_task(self) -> None:
        """Periodically update ticker data from REST API."""
        while self._running:
            try:
                for symbol in self.settings.symbol_list:
                    ticker = await self.rest_client.get_ticker_24h(symbol)
                    await self.cache.update_ticker(
                        symbol=symbol,
                        high_24h=ticker.high_price,
                        low_24h=ticker.low_price,
                        volume_24h=ticker.volume,
                        quote_volume_24h=ticker.quote_volume,
                    )
                    
                    # Also update OI
                    try:
                        oi = await self.rest_client.get_open_interest(symbol)
                        await self.cache.update_open_interest(
                            symbol=symbol,
                            oi=oi.open_interest,
                            oi_notional=oi.open_interest_notional,
                        )
                    except Exception as e:
                        self.logger.error("oi_update_error", symbol=symbol, error=str(e))
                
                await asyncio.sleep(10)  # Update every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("ticker_update_error", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup_task(self) -> None:
        """Periodically cleanup old data."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run hourly
                deleted = await self.storage.cleanup_old_data()
                self.logger.info("cleanup_complete", deleted_records=deleted)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("cleanup_task_error", error=str(e))
    
    async def _day_rollover_task(self) -> None:
        """Handle day rollover for indicators."""
        while self._running:
            try:
                # Wait until next day start
                now = timestamp_ms()
                next_day = get_day_start_ms(now) + 86_400_000
                wait_ms = next_day - now
                
                await asyncio.sleep(wait_ms / 1000)
                
                # Reset daily indicators
                for symbol in self.settings.symbol_list:
                    self.vwap.reset_day(symbol)
                    self.volume_profile.reset_day(symbol)
                    self.session_levels.reset_day(symbol)
                    self.cache.reset_cvd(symbol)
                
                self.logger.info("day_rollover_complete")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("day_rollover_error", error=str(e))
                await asyncio.sleep(60)
    
    async def start(self) -> None:
        """Start all server components."""
        self.logger.info(
            "starting_server",
            symbols=self.settings.symbol_list,
            port=self.settings.mcp_port,
            version=__version__,
        )
        
        self._running = True
        
        # Initialize storage
        await self.storage.initialize()

        # Prime in-memory state from persisted aggregates so restart is accurate even before
        # any historical backfill completes.
        await self._prime_cache_from_storage()

        # ------------------------------------------------------------------
        # Historical backfill
        # ------------------------------------------------------------------
        # Many indicators (previous-day VWAP/profile, session highs/lows, etc.) require
        # historical trades.
        #
        # IMPORTANT:
        # - Backfill can be heavy and may hit Binance REST rate limits.
        # - If BACKFILL_BLOCK_STARTUP=true, FastAPI will remain in
        #   "Waiting for application startup" until backfill completes.
        #   For remote CherryStudio usage you usually want this FALSE.
        async def _startup_backfill() -> None:
            if not self.settings.backfill_enabled:
                return
            block_start_ms = timestamp_ms()
            max_block_ms = getattr(self.settings, "backfill_block_startup_timeout_ms", 300_000)

            start_ms, end_ms = get_required_backfill_range()
            backfiller = AggTradesBackfiller(self.rest_client, self.storage)
            clear_days = bool(getattr(self.settings, "backfill_rebuild", False))

            try:
                for symbol in self.settings.symbol_list:
                    if max_block_ms > 0 and (timestamp_ms() - block_start_ms) > max_block_ms:
                        self.logger.warning(
                            "backfill_blocking_timeout_exceeded",
                            timeout_ms=max_block_ms,
                            note="Startup backfill stopped early; server will continue boot.",
                        )
                        break
                    try:
                        self.logger.info(
                            "backfill_symbol_start",
                            symbol=symbol,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            rebuild=clear_days,
                        )
                        result = await backfiller.backfill_symbol_range(
                            symbol=symbol,
                            start_time=start_ms,
                            end_time=end_ms,
                            max_requests=(
                                self.settings.backfill_max_requests_per_symbol
                                if self.settings.backfill_max_requests_per_symbol > 0
                                else None
                            ),
                            pause_ms=self.settings.backfill_request_pause_ms,
                            clear_days=clear_days,
                        )

                        # Prime day CVD so snapshot numbers are meaningful.
                        day_start = get_day_start_ms(end_ms)
                        if result.cvd_end_day is not None:
                            self.cache.set_cvd(symbol, float(result.cvd_end_day), reset_time=day_start)
                        else:
                            try:
                                cvd = await self.storage.get_day_cvd(symbol, day_start)
                                if cvd is not None:
                                    self.cache.set_cvd(symbol, float(cvd), reset_time=day_start)
                            except Exception:
                                pass

                        self.logger.info("backfill_done", **result.__dict__)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # Do NOT fail the whole server if one symbol backfill fails.
                        self.logger.warning("backfill_symbol_failed", symbol=symbol, error=str(e))
                    if max_block_ms > 0 and (timestamp_ms() - block_start_ms) > max_block_ms:
                        self.logger.warning(
                            "backfill_blocking_timeout_exceeded",
                            timeout_ms=max_block_ms,
                            note="Startup backfill stopped early; server will continue boot.",
                        )
                        break
            finally:
                try:
                    await backfiller.close()
                except Exception:
                    pass

        if self.settings.backfill_enabled:
            if getattr(self.settings, "backfill_block_startup", False):
                try:
                    self.logger.warning(
                        "backfill_blocking_startup",
                        note="Set BACKFILL_BLOCK_STARTUP=false if you need the MCP server reachable immediately",
                    )
                    await _startup_backfill()
                except Exception as e:
                    self.logger.warning("backfill_failed", error=str(e))
            else:
                self._backfill_task = asyncio.create_task(_startup_backfill())
        
        # ------------------------------------------------------------------
        # Orderbook initialization
        # ------------------------------------------------------------------
        # v8 awaited REST snapshots here. If REST is rate-limited, FastAPI startup stalls
        # and CherryStudio can't connect.
        # In v9 we initialize orderbooks in the background with retry/backoff.

        async def _init_orderbook_with_retry(symbol: str) -> None:
            delay = 1.0
            while self._running:
                try:
                    await self.orderbook.initialize_orderbook(symbol)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.warning(
                        "orderbook_init_retry",
                        symbol=symbol,
                        error=str(e),
                        wait_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(60.0, delay * 2)

        # Collect background tasks (so stop() can cancel them cleanly)
        background: list[asyncio.Task] = []

        for symbol in self.settings.symbol_list:
            task = asyncio.create_task(_init_orderbook_with_retry(symbol))
            background.append(task)
        
        # Register WebSocket callbacks
        self.ws_client.on_trade(self._handle_trade)
        self.ws_client.on_orderbook(self._handle_orderbook_update)
        self.ws_client.on_mark_price(self._handle_mark_price)
        self.ws_client.on_liquidation(self._handle_liquidation)
        
        # Start WebSocket connections (non-blocking)
        try:
            await self.ws_client.start(self.settings.symbol_list)
        except Exception as e:
            self.logger.warning("websocket_start_skipped", error=str(e))
        
        # Start background tasks
        background.extend(
            [
                asyncio.create_task(self._depth_delta_task()),
                asyncio.create_task(self._heatmap_task()) if getattr(self.settings, "heatmap_enabled", False) else None,
                asyncio.create_task(self._heatmap_metadata_task()) if getattr(self.settings, "heatmap_enabled", False) else None,
                asyncio.create_task(self._ticker_update_task()),
                asyncio.create_task(self._cleanup_task()),
                asyncio.create_task(self._day_rollover_task()),
            ]
        )

        # Remove optional Nones
        background = [t for t in background if t is not None]
        if self._backfill_task is not None:
            background.append(self._backfill_task)

        # Publish the full task list
        self._background_tasks = background
        
        self.logger.info("server_started")
    
    async def stop(self) -> None:
        """Stop all server components."""
        self.logger.info("stopping_server")
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._background_tasks.clear()
        
        # Stop WebSocket
        await self.ws_client.stop()
        
        # Close REST client
        await self.rest_client.close()
        
        # Close storage
        await self.storage.close()
        
        self.logger.info("server_stopped")


# Global server instance
server: CryptoOrderflowServer | None = None


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager."""
    global server
    
    if server:
        await server.start()
    
    yield
    
    if server:
        await server.stop()


def main():
    """Main entry point."""
    global server
    
    # Setup logging
    setup_logging()
    logger = get_logger("main")
    
    settings = get_settings()
    
    # Create server
    server = CryptoOrderflowServer()
    
    # Create MCP app
    app, mcp = create_mcp_server(server.tools)
    
    # Add lifespan
    app.router.lifespan_context = lifespan
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("starting_uvicorn", 
               host=settings.mcp_host, 
               port=settings.mcp_port)
    
    # Run uvicorn
    uvicorn.run(
        app,
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    main()
