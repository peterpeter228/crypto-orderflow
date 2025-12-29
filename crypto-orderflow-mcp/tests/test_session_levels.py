"""Tests for Session Levels calculator."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.indicators.session_levels import SessionLevelsCalculator


class TestSessionTimeDetection:
    """Test session time detection logic."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock()
        storage.update_session_levels = AsyncMock()
        storage.get_session_levels = AsyncMock(return_value={})
        return storage
    
    @pytest.fixture
    def calculator(self, mock_storage):
        """Create SessionLevelsCalculator instance."""
        return SessionLevelsCalculator(mock_storage)
    
    def test_tokyo_session_detection(self, calculator):
        """Test Tokyo session time detection (00:00-09:00 UTC)."""
        # 3:00 UTC - should be in Tokyo session
        dt = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        sessions = calculator._get_session_for_time(ts)
        assert "tokyo" in sessions
    
    def test_london_session_detection(self, calculator):
        """Test London session time detection (07:00-16:00 UTC)."""
        # 10:00 UTC - should be in London session
        dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        sessions = calculator._get_session_for_time(ts)
        assert "london" in sessions
    
    def test_ny_session_detection(self, calculator):
        """Test NY session time detection (13:00-22:00 UTC)."""
        # 15:00 UTC - should be in NY session
        dt = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        sessions = calculator._get_session_for_time(ts)
        assert "ny" in sessions
    
    def test_overlapping_sessions(self, calculator):
        """Test overlapping session detection."""
        # 8:00 UTC - should be in both Tokyo and London
        dt = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        sessions = calculator._get_session_for_time(ts)
        assert "tokyo" in sessions
        assert "london" in sessions
    
    def test_no_session_active(self, calculator):
        """Test when no session is active."""
        # 23:00 UTC - no session active
        dt = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        sessions = calculator._get_session_for_time(ts)
        assert len(sessions) == 0


class TestSessionHighLow:
    """Test session high/low tracking."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock()
        storage.update_session_levels = AsyncMock()
        storage.get_session_levels = AsyncMock(return_value={})
        return storage
    
    @pytest.fixture
    def calculator(self, mock_storage):
        """Create SessionLevelsCalculator instance."""
        return SessionLevelsCalculator(mock_storage)
    
    @pytest.mark.asyncio
    async def test_update_tracks_high(self, calculator):
        """Test that update tracks session high correctly."""
        symbol = "BTCUSDT"
        
        # 3:00 UTC - Tokyo session
        dt = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        await calculator.update(symbol, 50000.0, 1.0, ts)
        await calculator.update(symbol, 51000.0, 1.0, ts + 1000)  # New high
        await calculator.update(symbol, 50500.0, 1.0, ts + 2000)
        
        levels = await calculator.get_today_levels(symbol)
        
        assert "tokyo" in levels
        assert levels["tokyo"]["high"] == 51000.0
    
    @pytest.mark.asyncio
    async def test_update_tracks_low(self, calculator):
        """Test that update tracks session low correctly."""
        symbol = "BTCUSDT"
        
        # 3:00 UTC - Tokyo session
        dt = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        await calculator.update(symbol, 50000.0, 1.0, ts)
        await calculator.update(symbol, 49000.0, 1.0, ts + 1000)  # New low
        await calculator.update(symbol, 49500.0, 1.0, ts + 2000)
        
        levels = await calculator.get_today_levels(symbol)
        
        assert "tokyo" in levels
        assert levels["tokyo"]["low"] == 49000.0
    
    @pytest.mark.asyncio
    async def test_update_tracks_volume(self, calculator):
        """Test that update accumulates session volume."""
        symbol = "BTCUSDT"
        
        # 3:00 UTC - Tokyo session
        dt = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        ts = int(dt.timestamp() * 1000)
        
        await calculator.update(symbol, 50000.0, 1.0, ts)
        await calculator.update(symbol, 50100.0, 2.0, ts + 1000)
        await calculator.update(symbol, 50200.0, 0.5, ts + 2000)
        
        levels = await calculator.get_today_levels(symbol)
        
        assert "tokyo" in levels
        assert levels["tokyo"]["volume"] == 3.5  # 1.0 + 2.0 + 0.5
    
    def test_reset_day_clears_sessions(self, calculator):
        """Test that reset_day clears all sessions."""
        symbol = "BTCUSDT"
        calculator._sessions[symbol] = {
            "tokyo": MagicMock(high=50000, low=49000),
            "london": MagicMock(high=51000, low=50000),
        }
        
        calculator.reset_day(symbol)
        
        assert symbol in calculator._sessions
        assert len(calculator._sessions[symbol]) == 0
    
    def test_get_current_active_session(self, calculator):
        """Test getting current active sessions."""
        sessions = calculator.get_current_active_session()
        assert isinstance(sessions, list)
