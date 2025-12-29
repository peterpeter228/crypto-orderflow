"""VWAP (Volume Weighted Average Price) calculator."""

from typing import Any

from src.data.storage import DataStorage
from src.utils import get_logger, timestamp_ms, get_day_start_ms
from src.utils.helpers import get_yesterday_range_ms


class VWAPCalculator:
    """Calculate developing VWAP (dVWAP) and previous day VWAP (pdVWAP)."""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.logger = get_logger("indicators.vwap")
        
        # In-memory tracking for real-time updates
        self._cumulative: dict[str, dict[str, float]] = {}  # symbol -> {date -> {pv, vol}}
    
    async def update(self, symbol: str, price: float, volume: float, timestamp: int) -> None:
        """Update VWAP with new trade data.
        
        Args:
            symbol: Trading pair symbol
            price: Trade price
            volume: Trade volume
            timestamp: Trade timestamp in milliseconds
        """
        symbol = symbol.upper()
        date = get_day_start_ms(timestamp)
        
        # Update in-memory cache
        if symbol not in self._cumulative:
            self._cumulative[symbol] = {}
        
        if date not in self._cumulative[symbol]:
            # Load from storage if exists
            stored = await self.storage.get_vwap(symbol, date)
            if stored:
                self._cumulative[symbol][date] = {
                    "pv": stored["cumulative_pv"],
                    "vol": stored["cumulative_volume"],
                }
            else:
                self._cumulative[symbol][date] = {"pv": 0.0, "vol": 0.0}
        
        # Update cumulative values
        self._cumulative[symbol][date]["pv"] += price * volume
        self._cumulative[symbol][date]["vol"] += volume
        
        # Persist to storage (batched updates for efficiency)
        await self.storage.update_vwap(symbol, date, price, volume)
    
    def get_vwap(self, symbol: str, date: int | None = None) -> float | None:
        """Get VWAP for a specific date.
        
        Args:
            symbol: Trading pair symbol
            date: Day start timestamp (ms), defaults to today
        
        Returns:
            VWAP value or None if no data
        """
        symbol = symbol.upper()
        
        if date is None:
            date = get_day_start_ms(timestamp_ms())
        
        if symbol not in self._cumulative:
            return None
        
        if date not in self._cumulative[symbol]:
            return None
        
        data = self._cumulative[symbol][date]
        
        if data["vol"] == 0:
            return None
        
        return data["pv"] / data["vol"]
    
    async def get_dvwap(self, symbol: str) -> float | None:
        """Get developing VWAP (today's VWAP).

        For correctness we prefer persisted VWAP (vwap_data table). This ensures we return
        correct values after a restart or after historical backfill.
        """
        symbol = symbol.upper()
        today = get_day_start_ms(timestamp_ms())

        # 1) Prefer storage (covers restarts & backfill)
        stored = await self.storage.get_vwap(symbol, today)
        if stored and stored["cumulative_volume"] > 0:
            # Warm in-memory cache so subsequent calls are fast
            self._cumulative.setdefault(symbol, {})
            self._cumulative[symbol][today] = {
                "pv": float(stored["cumulative_pv"]),
                "vol": float(stored["cumulative_volume"]),
            }
            return float(stored["cumulative_pv"]) / float(stored["cumulative_volume"])

        # 2) Fallback to in-memory (may be partial if DB writes are disabled)
        vwap = self.get_vwap(symbol, today)
        return vwap

    async def get_pdvwap(self, symbol: str) -> float | None:
        """Get previous day's VWAP (pdVWAP)."""
        symbol = symbol.upper()
        today = get_day_start_ms(timestamp_ms())
        yesterday = today - 86_400_000

        stored = await self.storage.get_vwap(symbol, yesterday)
        if stored and stored["cumulative_volume"] > 0:
            # Warm cache for completeness
            self._cumulative.setdefault(symbol, {})
            self._cumulative[symbol][yesterday] = {
                "pv": float(stored["cumulative_pv"]),
                "vol": float(stored["cumulative_volume"]),
            }
            return float(stored["cumulative_pv"]) / float(stored["cumulative_volume"])

        return self.get_vwap(symbol, yesterday)

    async def get_key_levels(self, symbol: str, date: int | None = None) -> dict[str, Any]:
        """Get VWAP-based key levels.
        
        Args:
            symbol: Trading pair symbol
            date: Date to calculate for (defaults to today)
        
        Returns:
            Dict with dVWAP, pdVWAP values
        """
        symbol = symbol.upper()
        
        # `date` is the UTC day-start timestamp (ms). If provided, return VWAP for that day
        # instead of always "today".
        now_ms = timestamp_ms()
        today_start = get_day_start_ms(now_ms)
        day_start = date if date is not None else today_start

        async def _vwap_for_day(d: int) -> float | None:
            stored = await self.storage.get_vwap(symbol, d)
            if stored and stored.get("cumulative_volume", 0) > 0:
                # Warm cache
                self._cumulative.setdefault(symbol, {})
                self._cumulative[symbol][d] = {
                    "pv": float(stored["cumulative_pv"]),
                    "vol": float(stored["cumulative_volume"]),
                }
                return float(stored["cumulative_pv"]) / float(stored["cumulative_volume"])

            # Fallback (may be None if not loaded)
            return self.get_vwap(symbol, d)

        dvwap = await _vwap_for_day(day_start)
        pdvwap = await _vwap_for_day(day_start - 86_400_000)
        
        return {
            "symbol": symbol,
            "timestamp": now_ms,
            "dVWAP": dvwap,
            "pdVWAP": pdvwap,
            "unit": "USDT",
        }
    
    def reset_day(self, symbol: str) -> None:
        """Reset daily VWAP (called at day rollover)."""
        symbol = symbol.upper()
        today = get_day_start_ms(timestamp_ms())
        
        if symbol in self._cumulative:
            self._cumulative[symbol][today] = {"pv": 0.0, "vol": 0.0}
            self.logger.info("vwap_reset", symbol=symbol, date=today)


def calculate_vwap_from_trades(trades: list[dict[str, float]]) -> float | None:
    """Calculate VWAP from a list of trades.
    
    Args:
        trades: List of dicts with 'price' and 'volume' keys
    
    Returns:
        VWAP value or None if no trades
    """
    if not trades:
        return None
    
    total_pv = sum(t["price"] * t["volume"] for t in trades)
    total_volume = sum(t["volume"] for t in trades)
    
    if total_volume == 0:
        return None
    
    return total_pv / total_volume
