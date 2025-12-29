"""Tests for Footprint calculator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.indicators.footprint import (
    FootprintBar,
    FootprintLevel,
    FootprintCalculator,
    aggregate_footprint_bars,
)


class TestFootprintLevel:
    """Test FootprintLevel calculations."""
    
    def test_total_volume(self):
        """Test total volume calculation."""
        level = FootprintLevel(price=100.0, buy_volume=10.0, sell_volume=5.0)
        assert level.total_volume == 15.0
    
    def test_delta_positive(self):
        """Test positive delta (more buys)."""
        level = FootprintLevel(price=100.0, buy_volume=10.0, sell_volume=5.0)
        assert level.delta == 5.0
    
    def test_delta_negative(self):
        """Test negative delta (more sells)."""
        level = FootprintLevel(price=100.0, buy_volume=5.0, sell_volume=10.0)
        assert level.delta == -5.0


class TestFootprintBar:
    """Test FootprintBar calculations."""
    
    def test_empty_bar(self):
        """Test empty footprint bar properties."""
        bar = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=1000)
        
        assert bar.total_buy_volume == 0
        assert bar.total_sell_volume == 0
        assert bar.total_volume == 0
        assert bar.delta == 0
        assert bar.high is None
        assert bar.low is None
        assert bar.poc_price is None
    
    def test_bar_with_levels(self):
        """Test footprint bar with multiple levels."""
        bar = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=1000)
        bar.levels = {
            100.0: FootprintLevel(100.0, buy_volume=10.0, sell_volume=5.0),
            101.0: FootprintLevel(101.0, buy_volume=20.0, sell_volume=15.0),  # Highest volume
            102.0: FootprintLevel(102.0, buy_volume=5.0, sell_volume=8.0),
        }
        
        assert bar.total_buy_volume == 35.0
        assert bar.total_sell_volume == 28.0
        assert bar.total_volume == 63.0
        assert bar.delta == 7.0
        assert bar.high == 102.0
        assert bar.low == 100.0
        assert bar.poc_price == 101.0  # Highest total volume
    
    def test_max_delta_price(self):
        """Test max delta price detection."""
        bar = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=1000)
        bar.levels = {
            100.0: FootprintLevel(100.0, buy_volume=10.0, sell_volume=5.0),   # delta = 5
            101.0: FootprintLevel(101.0, buy_volume=20.0, sell_volume=8.0),   # delta = 12 (max)
            102.0: FootprintLevel(102.0, buy_volume=5.0, sell_volume=10.0),   # delta = -5
        }
        
        assert bar.max_delta_price == 101.0
    
    def test_min_delta_price(self):
        """Test min delta price detection."""
        bar = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=1000)
        bar.levels = {
            100.0: FootprintLevel(100.0, buy_volume=10.0, sell_volume=5.0),   # delta = 5
            101.0: FootprintLevel(101.0, buy_volume=8.0, sell_volume=20.0),   # delta = -12 (min)
            102.0: FootprintLevel(102.0, buy_volume=5.0, sell_volume=10.0),   # delta = -5
        }
        
        assert bar.min_delta_price == 101.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        bar = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=1000)
        bar.levels = {
            100.0: FootprintLevel(100.0, buy_volume=10.0, sell_volume=5.0, trade_count=3),
        }
        
        d = bar.to_dict()
        
        assert d["symbol"] == "BTCUSDT"
        assert d["timeframe"] == "1m"
        assert d["timestamp"] == 1000
        assert d["buyVolume"] == 10.0
        assert d["sellVolume"] == 5.0
        assert d["delta"] == 5.0
        assert len(d["levels"]) == 1


class TestAggregateFootprintBars:
    """Test footprint bar aggregation."""
    
    def test_aggregate_1m_to_5m(self):
        """Test aggregating 1m bars to 5m."""
        bars = []
        base_time = 0  # 0ms
        
        for i in range(5):
            bar = FootprintBar(
                symbol="BTCUSDT",
                timeframe="1m",
                timestamp=base_time + i * 60000,
            )
            bar.levels = {
                100.0: FootprintLevel(100.0, buy_volume=10.0, sell_volume=5.0),
            }
            bars.append(bar)
        
        aggregated = aggregate_footprint_bars(bars, "5m")
        
        assert len(aggregated) == 1
        assert aggregated[0].timeframe == "5m"
        assert aggregated[0].total_buy_volume == 50.0  # 10 * 5
        assert aggregated[0].total_sell_volume == 25.0  # 5 * 5
    
    def test_aggregate_empty_list(self):
        """Test aggregating empty list."""
        aggregated = aggregate_footprint_bars([], "5m")
        assert len(aggregated) == 0
    
    def test_aggregate_preserves_price_levels(self):
        """Test that aggregation preserves all price levels."""
        bar1 = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=0)
        bar1.levels = {
            100.0: FootprintLevel(100.0, buy_volume=10.0, sell_volume=5.0),
        }
        
        bar2 = FootprintBar(symbol="BTCUSDT", timeframe="1m", timestamp=60000)
        bar2.levels = {
            101.0: FootprintLevel(101.0, buy_volume=8.0, sell_volume=12.0),
        }
        
        aggregated = aggregate_footprint_bars([bar1, bar2], "5m")
        
        assert len(aggregated) == 1
        assert 100.0 in aggregated[0].levels
        assert 101.0 in aggregated[0].levels


class TestFootprintCalculatorClass:
    """Test FootprintCalculator class."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock()
        storage.upsert_footprint = AsyncMock()
        storage.get_footprint_range = AsyncMock(return_value=[])
        return storage
    
    @pytest.fixture
    def calculator(self, mock_storage):
        """Create FootprintCalculator instance."""
        return FootprintCalculator(mock_storage)
    
    @pytest.mark.anyio
    async def test_update_creates_bar(self, calculator):
        """Test that update creates footprint bar."""
        await calculator.update(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.0,
            is_buyer_maker=False,  # Buy
            timestamp=0,
        )
        
        bar = calculator.get_current_bar("BTCUSDT", "1m")
        
        assert bar is not None
        assert bar.total_buy_volume == 1.0
        assert bar.total_sell_volume == 0.0
    
    @pytest.mark.anyio
    async def test_update_sell_trade(self, calculator):
        """Test update with sell trade (is_buyer_maker=True)."""
        await calculator.update(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.0,
            is_buyer_maker=True,  # Sell
            timestamp=0,
        )
        
        bar = calculator.get_current_bar("BTCUSDT", "1m")
        
        assert bar is not None
        assert bar.total_buy_volume == 0.0
        assert bar.total_sell_volume == 1.0
    
    @pytest.mark.anyio
    async def test_update_aggregates_at_price_level(self, calculator):
        """Test that trades at same price level are aggregated."""
        await calculator.update("BTCUSDT", 50000.0, 1.0, False, 0)
        await calculator.update("BTCUSDT", 50000.0, 0.5, True, 100)
        
        bar = calculator.get_current_bar("BTCUSDT", "1m")
        
        assert bar is not None
        assert 50000.0 in bar.levels
        assert bar.levels[50000.0].buy_volume == 1.0
        assert bar.levels[50000.0].sell_volume == 0.5
