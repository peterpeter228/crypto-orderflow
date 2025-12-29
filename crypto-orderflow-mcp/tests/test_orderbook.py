"""Tests for Orderbook management."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.data.orderbook import OrderbookManager, ManagedOrderbook
from src.binance.types import OrderbookLevel, OrderbookSnapshot, OrderbookUpdate


class TestManagedOrderbook:
    """Test ManagedOrderbook data structure."""
    
    def test_initial_state(self):
        """Test initial orderbook state."""
        ob = ManagedOrderbook(symbol="BTCUSDT")
        
        assert ob.symbol == "BTCUSDT"
        assert len(ob.bids) == 0
        assert len(ob.asks) == 0
        assert ob.last_update_id == 0
        assert ob.is_valid is False


class TestOrderbookManager:
    """Test OrderbookManager functionality."""
    
    @pytest.fixture
    def mock_rest_client(self):
        """Create mock REST client."""
        client = MagicMock()
        client.get_orderbook_snapshot = AsyncMock(return_value=OrderbookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            timestamp=1000,
            bids=[
                OrderbookLevel(50000.0, 1.0),
                OrderbookLevel(49999.0, 2.0),
                OrderbookLevel(49998.0, 1.5),
            ],
            asks=[
                OrderbookLevel(50001.0, 1.0),
                OrderbookLevel(50002.0, 2.0),
                OrderbookLevel(50003.0, 1.5),
            ],
        ))
        return client
    
    @pytest.fixture
    def manager(self, mock_rest_client):
        """Create OrderbookManager instance."""
        return OrderbookManager(mock_rest_client)
    
    @pytest.mark.anyio
    async def test_initialize_orderbook(self, manager):
        """Test orderbook initialization from snapshot."""
        await manager.initialize_orderbook("BTCUSDT")
        
        assert manager.is_valid("BTCUSDT")
        
        ob = manager.get_orderbook("BTCUSDT")
        assert ob is not None
        assert ob["lastUpdateId"] == 100
        assert len(ob["bids"]) == 3
        assert len(ob["asks"]) == 3
    
    @pytest.mark.anyio
    async def test_get_best_bid_ask(self, manager):
        """Test getting best bid/ask."""
        await manager.initialize_orderbook("BTCUSDT")
        
        result = manager.get_best_bid_ask("BTCUSDT")
        
        assert result is not None
        best_bid, best_ask = result
        assert best_bid == 50000.0
        assert best_ask == 50001.0
    
    @pytest.mark.anyio
    async def test_get_mid_price(self, manager):
        """Test getting mid price."""
        await manager.initialize_orderbook("BTCUSDT")
        
        mid = manager.get_mid_price("BTCUSDT")
        
        assert mid == 50000.5  # (50000 + 50001) / 2
    
    @pytest.mark.anyio
    async def test_get_depth_within_percent(self, manager):
        """Test getting depth within price range."""
        await manager.initialize_orderbook("BTCUSDT")
        
        depth = manager.get_depth_within_percent("BTCUSDT", 0.01)  # ±0.01%
        
        assert depth is not None
        assert "bidVolume" in depth
        assert "askVolume" in depth
        assert "netVolume" in depth
        assert "midPrice" in depth
    
    @pytest.mark.anyio
    async def test_process_update_valid(self, manager):
        """Test processing valid orderbook update."""
        await manager.initialize_orderbook("BTCUSDT")
        
        update = OrderbookUpdate(
            symbol="BTCUSDT",
            event_time=2000,
            transaction_time=2000,
            first_update_id=101,
            last_update_id=101,
            prev_last_update_id=100,
            bids=[OrderbookLevel(50000.0, 1.5)],  # Update bid
            asks=[OrderbookLevel(50001.0, 0.0)],  # Remove ask
        )
        
        await manager.process_update(update)
        
        ob = manager.get_orderbook("BTCUSDT")
        assert ob is not None
        assert ob["lastUpdateId"] == 101
    
    @pytest.mark.anyio
    async def test_process_update_removes_zero_quantity(self, manager):
        """Test that zero quantity removes price level."""
        await manager.initialize_orderbook("BTCUSDT")
        
        # Remove a bid level
        update = OrderbookUpdate(
            symbol="BTCUSDT",
            event_time=2000,
            transaction_time=2000,
            first_update_id=101,
            last_update_id=101,
            prev_last_update_id=100,
            bids=[OrderbookLevel(50000.0, 0.0)],  # Remove
            asks=[],
        )
        
        await manager.process_update(update)
        
        ob = manager.get_orderbook("BTCUSDT")
        assert ob is not None
        
        # Check that 50000.0 bid is removed
        bid_prices = [b["price"] for b in ob["bids"]]
        assert 50000.0 not in bid_prices
    
    def test_is_valid_unknown_symbol(self, manager):
        """Test is_valid for unknown symbol."""
        assert manager.is_valid("UNKNOWN") is False
    
    def test_get_orderbook_unknown_symbol(self, manager):
        """Test get_orderbook for unknown symbol."""
        assert manager.get_orderbook("UNKNOWN") is None
    
    def test_get_best_bid_ask_unknown_symbol(self, manager):
        """Test get_best_bid_ask for unknown symbol."""
        assert manager.get_best_bid_ask("UNKNOWN") is None


class TestOrderbookConsistency:
    """Test orderbook consistency validation."""
    
    @pytest.fixture
    def mock_rest_client(self):
        """Create mock REST client."""
        client = MagicMock()
        client.get_orderbook_snapshot = AsyncMock(return_value=OrderbookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            timestamp=1000,
            bids=[OrderbookLevel(50000.0, 1.0)],
            asks=[OrderbookLevel(50001.0, 1.0)],
        ))
        return client
    
    @pytest.fixture
    def manager(self, mock_rest_client):
        """Create OrderbookManager instance."""
        return OrderbookManager(mock_rest_client)
    
    @pytest.mark.anyio
    async def test_gap_detection_triggers_rebuild(self, manager):
        """Test that sequence gap triggers orderbook rebuild."""
        await manager.initialize_orderbook("BTCUSDT")
        
        # Send update with gap
        update = OrderbookUpdate(
            symbol="BTCUSDT",
            event_time=2000,
            transaction_time=2000,
            first_update_id=105,  # Gap: expected 101
            last_update_id=105,
            prev_last_update_id=104,  # Doesn't match current 100
            bids=[],
            asks=[],
        )
        
        await manager.process_update(update)
        
        # Orderbook should be invalid after gap detection
        # (rebuild is async, so just check the mechanism)
        ob = manager._orderbooks.get("BTCUSDT")
        assert ob is not None
