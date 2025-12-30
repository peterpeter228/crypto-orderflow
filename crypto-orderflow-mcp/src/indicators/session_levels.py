"""Session High/Low calculator.

Sessions are configurable via the `SESSIONS` env var (see `.env.example`).
By default we use Exocharts-style non-overlapping "Main sessions":
  A: 00:00-06:00, L: 06:00-12:00, N: 12:00-20:00, E: 20:00-00:00 (UTC)
"""

from typing import Any
from dataclasses import dataclass

from src.data.storage import DataStorage
from src.config import get_settings, SessionDef
from src.utils import get_logger, timestamp_ms, ms_to_datetime
from src.utils.helpers import get_day_start_ms


@dataclass
class SessionRange:
    """Session time range and levels."""
    name: str
    high: float | None = None
    low: float | None = None
    high_time: int | None = None
    low_time: int | None = None
    volume: float = 0.0
    is_active: bool = False


class SessionLevelsCalculator:
    """Calculate session high/low levels."""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.settings = get_settings()
        self.logger = get_logger("indicators.session_levels")

        # Ensure legacy sessions (tokyo/london/ny) remain available for backward compatibility/tests.
        self.session_defs = list(self.settings.session_defs)
        legacy = [
            SessionDef(name="tokyo", time=self.settings.tokyo),
            SessionDef(name="london", time=self.settings.london),
            SessionDef(name="ny", time=self.settings.ny),
        ]
        legacy_names = {s.name.lower() for s in self.session_defs}
        for s in legacy:
            if s.name.lower() not in legacy_names:
                self.session_defs.append(s)
        
        # In-memory tracking for per-day session ranges:
        #   symbol -> day_start_ms -> session_name -> SessionRange
        self._sessions: dict[str, dict[int, dict[str, SessionRange]]] = {}
        # Track which (symbol, day_start_ms) we've warmed from storage
        self._warmed: set[tuple[str, int]] = set()
    
    def _sessions_for_timestamp(self, timestamp: int) -> list[tuple[str, int]]:
        """Return active sessions for a timestamp.

        For sessions that span midnight (end <= start), trades after midnight
        belong to the *previous* session day.

        Returns:
            List[(session_name, session_day_start_ms)].
        """

        dt = ms_to_datetime(timestamp)
        minutes_of_day = dt.hour * 60 + dt.minute
        day_start = get_day_start_ms(timestamp)

        out: list[tuple[str, int]] = []
        for s in self.session_defs:
            start = s.start_minutes
            end = s.end_minutes

            # Normal session
            if end > start:
                if start <= minutes_of_day < end:
                    out.append((s.name, day_start))
                continue

            # Spans midnight (e.g. 20:00-02:00)
            if minutes_of_day >= start:
                out.append((s.name, day_start))
            elif minutes_of_day < end:
                out.append((s.name, day_start - 86_400_000))

        return out
    
    def _get_or_create_session(self, symbol: str, day_start: int, session_name: str) -> SessionRange:
        """Get or create session range for symbol+day."""
        symbol = symbol.upper()

        if symbol not in self._sessions:
            self._sessions[symbol] = {}
        if day_start not in self._sessions[symbol]:
            self._sessions[symbol][day_start] = {}
        if session_name not in self._sessions[symbol][day_start]:
            self._sessions[symbol][day_start][session_name] = SessionRange(name=session_name)
        return self._sessions[symbol][day_start][session_name]
    
    async def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: int,
    ) -> None:
        """Update session levels with new trade data.
        
        Args:
            symbol: Trading pair symbol
            price: Trade price
            volume: Trade volume
            timestamp: Trade timestamp in milliseconds
        """
        symbol = symbol.upper()
        active_sessions = self._sessions_for_timestamp(timestamp)

        for session_name, session_day in active_sessions:
            session = self._get_or_create_session(symbol, session_day, session_name)
            session.is_active = True
            session.volume += volume
            
            # Update high
            if session.high is None or price > session.high:
                session.high = price
                session.high_time = timestamp
            
            # Update low
            if session.low is None or price < session.low:
                session.low = price
                session.low_time = timestamp
            
            # Persist to storage
            await self.storage.update_session_levels(
                symbol=symbol,
                date=session_day,
                session=session_name,
                price=price,
                timestamp=timestamp,
                volume=volume,
            )

        # Mark inactive sessions for the current calendar day.
        current_day = get_day_start_ms(timestamp)
        active_for_current = {name for (name, d) in active_sessions if d == current_day}
        for s in self.session_defs:
            session = self._get_or_create_session(symbol, current_day, s.name)
            session.is_active = s.name in active_for_current

    async def get_levels_for_day(self, symbol: str, day_start: int) -> dict[str, dict[str, Any]]:
        """Get session levels for a specific day (warm from storage if needed)."""

        symbol = symbol.upper()
        key = (symbol, day_start)

        if key not in self._warmed:
            if symbol in self._sessions and day_start in self._sessions[symbol]:
                self._warmed.add(key)
            else:
                try:
                    stored = await self.storage.get_session_levels(symbol, day_start)

                    if symbol not in self._sessions:
                        self._sessions[symbol] = {}
                    self._sessions[symbol][day_start] = {}

                    # Determine actives for *now* (only makes sense for today's day_start)
                    now_active = {
                        name
                        for (name, d) in self._sessions_for_timestamp(timestamp_ms())
                        if d == day_start
                    }

                    for sdef in self.session_defs:
                        s = SessionRange(name=sdef.name)
                        row = stored.get(sdef.name)
                        if row:
                            s.high = row.get("high")
                            s.low = row.get("low")
                            s.high_time = row.get("high_time")
                            s.low_time = row.get("low_time")
                            s.volume = float(row.get("volume") or 0.0)
                        s.is_active = sdef.name in now_active
                        self._sessions[symbol][day_start][sdef.name] = s

                    self._warmed.add(key)
                except Exception as e:
                    self.logger.debug(
                        "session_levels_warm_failed", symbol=symbol, day_start=day_start, error=str(e)
                    )
                    self._warmed.add(key)

        sessions = self._sessions.get(symbol, {}).get(day_start)
        if not sessions:
            return {}

        return {
            name: {
                "high": s.high,
                "low": s.low,
                "highTime": s.high_time,
                "lowTime": s.low_time,
                "volume": s.volume,
                "isActive": s.is_active,
            }
            for name, s in sessions.items()
        }

    async def get_today_levels(self, symbol: str) -> dict[str, dict[str, Any]]:
        """Get today's session levels."""
        symbol = symbol.upper()
        if symbol in self._sessions and self._sessions[symbol]:
            day_start = max(self._sessions[symbol].keys())
        else:
            day_start = get_day_start_ms(timestamp_ms())
        return await self.get_levels_for_day(symbol, day_start)
    
    async def get_yesterday_levels(self, symbol: str) -> dict[str, dict[str, Any]]:
        """Get yesterday's session levels from storage.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Dict with session levels
        """
        symbol = symbol.upper()
        yesterday = get_day_start_ms(timestamp_ms()) - 86_400_000
        stored = await self.storage.get_session_levels(symbol, yesterday)

        result: dict[str, dict[str, Any]] = {}
        for session_name, data in stored.items():
            result[session_name] = {
                "high": data["high"],
                "low": data["low"],
                "highTime": data["high_time"],
                "lowTime": data["low_time"],
                "volume": data["volume"],
                "isActive": False,
            }
        return result
    
    async def get_key_levels(self, symbol: str, date: int | None = None) -> dict[str, Any]:
        """Get all session key levels.
        
        Args:
            symbol: Trading pair symbol
            date: Date to calculate for (defaults to today)
        
        Returns:
            Dict with configured session high/low levels
        """
        symbol = symbol.upper()
        
        if date is None:
            day_start = get_day_start_ms(timestamp_ms())
        else:
            day_start = int(date)

        today_levels = await self.get_levels_for_day(symbol, day_start)
        stored_yesterday = await self.storage.get_session_levels(symbol, day_start - 86_400_000)
        
        # Format response
        result: dict[str, Any] = {
            "symbol": symbol,
            "timestamp": timestamp_ms(),
            "sessions": {
                "timezone": "UTC",
                **{s.name: {"hours": f"{s.time.start_hour:02d}:{s.time.start_minute:02d}-{(0 if s.time.end_minutes==1440 else s.time.end_hour):02d}:{s.time.end_minute:02d}"} for s in self.session_defs},
            },
            "today": {},
            "yesterday": {},
            "unit": "USDT",
        }
        
        # Add today's levels
        for s in self.session_defs:
            data = today_levels.get(s.name)
            if not data:
                continue
            result["today"][f"{s.name}H"] = data["high"]
            result["today"][f"{s.name}L"] = data["low"]
            result["today"][f"{s.name}Volume"] = data["volume"]
            result["today"][f"{s.name}Active"] = data["isActive"]
        
        # Add yesterday's levels
        for s in self.session_defs:
            row = stored_yesterday.get(s.name)
            if not row:
                continue
            result["yesterday"][f"{s.name}H"] = row.get("high")
            result["yesterday"][f"{s.name}L"] = row.get("low")
            result["yesterday"][f"{s.name}Volume"] = float(row.get("volume") or 0.0)
        
        return result
    
    def reset_day(self, symbol: str) -> None:
        """Reset daily session levels (called at day rollover)."""
        symbol = symbol.upper()
        self._sessions[symbol] = {}
        # Remove warmed markers for that symbol
        self._warmed = {k for k in self._warmed if k[0] != symbol}
        self.logger.info("sessions_reset", symbol=symbol)

    def get_current_active_session(self) -> list[str]:
        """Get currently active sessions."""
        return [name for (name, _day) in self._sessions_for_timestamp(timestamp_ms())]

    def _legacy_sessions_for_time(self, timestamp: int) -> list[str]:
        """Legacy helper using tokyo/london/ny windows only (test/backward compat)."""
        dt = ms_to_datetime(timestamp)
        minutes_of_day = dt.hour * 60 + dt.minute
        legacy_defs = [
            ("tokyo", self.settings.tokyo.start_minutes, self.settings.tokyo.end_minutes),
            ("london", self.settings.london.start_minutes, self.settings.london.end_minutes),
            ("ny", self.settings.ny.start_minutes, self.settings.ny.end_minutes),
        ]
        out: list[str] = []
        for name, start, end in legacy_defs:
            if end > start:
                if start <= minutes_of_day < end:
                    out.append(name)
            else:
                if minutes_of_day >= start or minutes_of_day < end:
                    out.append(name)
        return out

    def _get_session_for_time(self, timestamp: int) -> list[str]:
        """Public, backward-compatible helper for tests/consumers."""
        legacy = self._legacy_sessions_for_time(timestamp)
        return legacy
