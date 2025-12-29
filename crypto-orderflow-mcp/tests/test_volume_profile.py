"""Tests for Volume Profile calculator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

import src.indicators.volume_profile as volume_profile
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
    
    @pytest.mark.asyncio
    async def test_update_accumulates_volume(self, calculator, monkeypatch):
        """Test that update accumulates volume at price levels."""
        symbol = "BTCUSDT"
        timestamp = 1700000000000

        monkeypatch.setattr(volume_profile, "timestamp_ms", lambda: timestamp)
        
        await calculator.update(symbol, 50000.0, 1.0, 0.6, 0.4, timestamp)
        await calculator.update(symbol, 50000.0, 0.5, 0.3, 0.2, timestamp)
        
        profile = await calculator.get_today_profile(symbol)
        
        assert 50000.0 in profile
        assert profile[50000.0] == 75000.0  # (1.0 + 0.5) * 50,000 notional
    
    def test_reset_day_clears_profile(self, calculator):
        """Test that reset_day clears the profile."""
        symbol = "BTCUSDT"
        calculator._profiles[symbol] = {100.0: 50.0, 101.0: 30.0}
        
        calculator.reset_day(symbol)
        
        assert symbol in calculator._profiles
        assert len(calculator._profiles[symbol]) == 0

    @pytest.mark.asyncio
    async def test_get_key_levels_previous_day_low_coverage_warns(self, mock_storage, monkeypatch):
        """Low previous-day coverage should mark incomplete and emit warning."""
        calculator = VolumeProfileCalculator(mock_storage)

        now_ms = 1_700_000_000_000
        day_start = volume_profile.get_day_start_ms(now_ms)
        monkeypatch.setattr(volume_profile, "timestamp_ms", lambda: now_ms)

        # Empty profiles; coverage data drives the warning.
        mock_storage.get_profile_range = AsyncMock(side_effect=[[], []])
        mock_storage.get_footprint_coverage = AsyncMock(
            side_effect=[
                {"minuteBuckets": 500, "minTs": None, "maxTs": None},  # developing (above 0.3)
                {"minuteBuckets": 500, "minTs": None, "maxTs": None},  # previous day (below 0.9)
            ]
        )

        result = await calculator.get_key_levels("BTCUSDT", date=day_start)

        assert result["previousDay"]["complete"] is False
        assert result["previousDay"]["provisional"] is True
        assert any("Previous day profile incomplete" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_get_key_levels_developing_low_coverage_warns(self, mock_storage, monkeypatch):
        """Developing coverage below threshold should be flagged as provisional."""
        calculator = VolumeProfileCalculator(mock_storage)

        # Six hours into the day -> expected minutes ~360
        day_start = volume_profile.get_day_start_ms(1_700_000_000_000)
        now_ms = day_start + 6 * 60 * 60 * 1000
        monkeypatch.setattr(volume_profile, "timestamp_ms", lambda: now_ms)

        mock_storage.get_profile_range = AsyncMock(side_effect=[[], []])
        mock_storage.get_footprint_coverage = AsyncMock(
            side_effect=[
                {"minuteBuckets": 50, "minTs": None, "maxTs": None},   # developing (~13.9% < 30%)
                {"minuteBuckets": 1440, "minTs": None, "maxTs": None},  # previous day full
            ]
        )

        result = await calculator.get_key_levels("BTCUSDT", date=day_start)

        assert result["developing"]["provisional"] is True
        assert any("Developing profile coverage low" in w for w in result["warnings"])
