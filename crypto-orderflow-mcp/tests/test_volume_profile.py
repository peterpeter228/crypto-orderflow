"""Tests for Volume Profile calculator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.indicators.volume_profile import (
    VolumeProfileCalculator,
    calculate_volume_profile,
)


class TestVolumeProfileCalculation:
    """Test Volume Profile calculation logic."""
    
    def test_calculate_poc_basic(self):
        """Test POC calculation - price with highest volume."""
        mock_storage = MagicMock()
        calculator = VolumeProfileCalculator(mock_storage)
        
        profile = {
            100.0: 50.0,
            101.0: 100.0,  # Highest volume - should be POC
            102.0: 75.0,
            103.0: 25.0,
        }
        
        poc = calculator.calculate_poc(profile)
        assert poc == 101.0
    
    def test_calculate_poc_empty(self):
        """Test POC with empty profile."""
        mock_storage = MagicMock()
        calculator = VolumeProfileCalculator(mock_storage)
        
        poc = calculator.calculate_poc({})
        assert poc is None
    
    def test_calculate_value_area_70_percent(self):
        """Test Value Area calculation with 70% threshold."""
        mock_storage = MagicMock()
        calculator = VolumeProfileCalculator(mock_storage)
        
        # Create profile where 70% is easy to calculate
        # Total volume = 100
        # 70% = 70 volume needed
        profile = {
            100.0: 10.0,
            101.0: 15.0,
            102.0: 30.0,  # POC - highest volume
            103.0: 25.0,
            104.0: 20.0,
        }
        
        poc, vah, val = calculator.calculate_value_area(profile, 70.0)
        
        assert poc == 102.0  # Highest volume
        assert vah is not None
        assert val is not None
        assert val <= poc <= vah
        
        # Check that value area contains >= 70% volume
        total = sum(profile.values())
        va_volume = sum(profile[p] for p in profile if val <= p <= vah)
        assert va_volume / total >= 0.70
    
    def test_calculate_value_area_empty(self):
        """Test Value Area with empty profile."""
        mock_storage = MagicMock()
        calculator = VolumeProfileCalculator(mock_storage)
        
        poc, vah, val = calculator.calculate_value_area({})
        assert poc is None
        assert vah is None
        assert val is None
    
    def test_calculate_value_area_single_level(self):
        """Test Value Area with single price level."""
        mock_storage = MagicMock()
        calculator = VolumeProfileCalculator(mock_storage)
        
        profile = {100.0: 50.0}
        poc, vah, val = calculator.calculate_value_area(profile, 70.0)
        
        assert poc == 100.0
        assert vah == 100.0
        assert val == 100.0


class TestVolumeProfileFromTrades:
    """Test Volume Profile calculation from trade data."""
    
    def test_calculate_volume_profile_basic(self):
        """Test basic volume profile from trades."""
        trades = [
            {"price": 100.5, "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0},
            {"price": 100.7, "volume": 15.0, "buy_volume": 10.0, "sell_volume": 5.0},
            {"price": 100.5, "volume": 5.0, "buy_volume": 2.0, "sell_volume": 3.0},  # Same level
            {"price": 101.2, "volume": 20.0, "buy_volume": 8.0, "sell_volume": 12.0},
        ]
        
        tick_size = 0.1
        profile = calculate_volume_profile(trades, tick_size)
        
        # Should have 3 price levels
        assert len(profile) == 3
        
        # Check aggregation at 100.5
        assert 100.5 in profile
        assert profile[100.5]["volume"] == 15.0  # 10 + 5
        assert profile[100.5]["buy_volume"] == 8.0  # 6 + 2
        assert profile[100.5]["sell_volume"] == 7.0  # 4 + 3
        assert profile[100.5]["delta"] == 1.0  # 8 - 7
    
    def test_calculate_volume_profile_with_rounding(self):
        """Test that prices are rounded to tick size."""
        trades = [
            {"price": 100.12, "volume": 10.0, "buy_volume": 10.0, "sell_volume": 0.0},
            {"price": 100.18, "volume": 5.0, "buy_volume": 5.0, "sell_volume": 0.0},
        ]
        
        tick_size = 0.1
        profile = calculate_volume_profile(trades, tick_size)
        
        # Both trades should round to 100.1
        assert len(profile) == 1
        # Handle floating point precision - get the key
        price_key = list(profile.keys())[0]
        assert abs(price_key - 100.1) < 0.001
        assert profile[price_key]["volume"] == 15.0


class TestVolumeProfileCalculatorClass:
    """Test VolumeProfileCalculator class methods."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock()
        storage.upsert_daily_trade = AsyncMock()
        storage.get_daily_trades = AsyncMock(return_value=[])
        return storage
    
    @pytest.fixture
    def calculator(self, mock_storage):
        """Create VolumeProfileCalculator instance."""
        return VolumeProfileCalculator(mock_storage)
    
    @pytest.mark.anyio
    async def test_update_accumulates_volume(self, calculator, monkeypatch):
        """Test that update accumulates volume at price levels."""
        symbol = "BTCUSDT"
        timestamp = 1700000000000

        monkeypatch.setattr("src.indicators.volume_profile.timestamp_ms", lambda: timestamp)
        
        await calculator.update(symbol, 50000.0, 1.0, 0.6, 0.4, timestamp)
        await calculator.update(symbol, 50000.0, 0.5, 0.3, 0.2, timestamp)
        
        profile = await calculator.get_today_profile(symbol)
        
        assert 50000.0 in profile
        assert profile[50000.0] == 75000.0  # quote volume (notional) for 1.5 base volume
    
    def test_reset_day_clears_profile(self, calculator):
        """Test that reset_day clears the profile."""
        symbol = "BTCUSDT"
        calculator._profiles[symbol] = {100.0: 50.0, 101.0: 30.0}
        
        calculator.reset_day(symbol)
        
        assert symbol in calculator._profiles
        assert len(calculator._profiles[symbol]) == 0

    @pytest.mark.anyio
    async def test_get_key_levels_fallback_to_daily_trades(self, monkeypatch):
        """When footprint is empty but daily trades exist, previous day key levels should be populated."""
        # Fixed timestamp aligned to a UTC day boundary for deterministic ranges.
        fixed_ts = 1_700_000_000_000
        day_start = (fixed_ts // 86_400_000) * 86_400_000
        monkeypatch.setattr("src.indicators.volume_profile.timestamp_ms", lambda: fixed_ts)

        storage = MagicMock()
        storage.get_profile_range = AsyncMock(side_effect=[[], []])
        storage.get_footprint_coverage = AsyncMock(return_value={"minute_buckets": 0, "min_ts": None, "max_ts": None})
        storage.get_daily_trades = AsyncMock(
            side_effect=[
                [],  # today fallback (empty)
                [
                    {"price_level": 100.0, "notional": 200.0},
                    {"price_level": 101.0, "notional": 300.0},
                ],
            ]
        )

        calculator = VolumeProfileCalculator(storage)
        levels = await calculator.get_key_levels("BTCUSDT", date=day_start)

        assert levels["previousDay"]["POC"] is not None
        assert levels["previousDay"]["VAH"] is not None
        assert levels["previousDay"]["VAL"] is not None
