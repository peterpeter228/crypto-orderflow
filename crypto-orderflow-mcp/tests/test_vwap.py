"""Tests for VWAP calculator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.indicators.vwap import VWAPCalculator, calculate_vwap_from_trades


class TestVWAPCalculation:
    """Test VWAP calculation logic."""
    
    def test_calculate_vwap_from_trades_basic(self):
        """Test basic VWAP calculation."""
        trades = [
            {"price": 100.0, "volume": 10.0},
            {"price": 110.0, "volume": 20.0},
            {"price": 105.0, "volume": 15.0},
        ]
        
        # Manual calculation:
        # Total PV = 100*10 + 110*20 + 105*15 = 1000 + 2200 + 1575 = 4775
        # Total Volume = 10 + 20 + 15 = 45
        # VWAP = 4775 / 45 = 106.111...
        
        vwap = calculate_vwap_from_trades(trades)
        assert vwap is not None
        assert abs(vwap - 106.1111) < 0.01
    
    def test_calculate_vwap_from_trades_empty(self):
        """Test VWAP with empty trades."""
        vwap = calculate_vwap_from_trades([])
        assert vwap is None
    
    def test_calculate_vwap_from_trades_single(self):
        """Test VWAP with single trade."""
        trades = [{"price": 50000.0, "volume": 1.5}]
        vwap = calculate_vwap_from_trades(trades)
        assert vwap == 50000.0
    
    def test_calculate_vwap_from_trades_weighted_correctly(self):
        """Test that VWAP is correctly weighted by volume."""
        # Large volume at low price should pull VWAP down
        trades = [
            {"price": 100.0, "volume": 100.0},  # Large volume
            {"price": 200.0, "volume": 1.0},    # Small volume
        ]
        
        vwap = calculate_vwap_from_trades(trades)
        assert vwap is not None
        # VWAP should be much closer to 100 than to 200
        assert vwap < 110
    
    def test_calculate_vwap_equal_weights(self):
        """Test VWAP with equal volumes gives simple average."""
        trades = [
            {"price": 100.0, "volume": 10.0},
            {"price": 200.0, "volume": 10.0},
        ]
        
        vwap = calculate_vwap_from_trades(trades)
        assert vwap == 150.0


class TestVWAPCalculatorClass:
    """Test VWAPCalculator class methods."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock()
        storage.get_vwap = AsyncMock(return_value=None)
        storage.update_vwap = AsyncMock()
        return storage
    
    @pytest.fixture
    def calculator(self, mock_storage):
        """Create VWAPCalculator instance."""
        return VWAPCalculator(mock_storage)
    
    @pytest.mark.anyio
    async def test_update_accumulates_values(self, calculator):
        """Test that update accumulates price*volume and volume."""
        symbol = "BTCUSDT"
        day_ts = 1700000000000  # Fixed timestamp
        
        await calculator.update(symbol, 50000.0, 1.0, day_ts)
        await calculator.update(symbol, 51000.0, 2.0, day_ts)
        
        vwap = calculator.get_vwap(symbol, day_ts - (day_ts % 86400000))
        
        # Expected: (50000*1 + 51000*2) / (1+2) = 152000/3 = 50666.67
        assert vwap is not None
        assert abs(vwap - 50666.67) < 1
    
    @pytest.mark.anyio
    async def test_get_vwap_returns_none_for_unknown_symbol(self, calculator):
        """Test that get_vwap returns None for unknown symbol."""
        vwap = calculator.get_vwap("UNKNOWN")
        assert vwap is None
    
    def test_reset_day_clears_data(self, calculator):
        """Test that reset_day clears accumulated data."""
        symbol = "BTCUSDT"
        calculator._cumulative[symbol] = {0: {"pv": 1000.0, "vol": 10.0}}
        
        calculator.reset_day(symbol)
        
        # Should have new entry for today
        from src.utils.helpers import get_day_start_ms, timestamp_ms
        today = get_day_start_ms(timestamp_ms())
        assert symbol in calculator._cumulative
        assert today in calculator._cumulative[symbol]
        assert calculator._cumulative[symbol][today]["pv"] == 0
        assert calculator._cumulative[symbol][today]["vol"] == 0
