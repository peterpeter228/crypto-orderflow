"""Helper utility functions."""

import math
from decimal import Decimal
from functools import lru_cache
from datetime import datetime, timezone
from typing import Literal

# Timeframe to milliseconds mapping
TIMEFRAME_MS: dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds (UTC)."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime (UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


@lru_cache(maxsize=128)
def _tick_decimals(tick_size: float) -> int:
    """Return decimal places implied by tick_size.

    Uses Decimal(str(tick_size)) to avoid common binary float artifacts.
    """
    d = Decimal(str(tick_size)).normalize()
    return max(0, -d.as_tuple().exponent)


def round_to_tick(price: float, tick_size: float, direction: Literal["down", "up", "nearest"] = "down") -> float:
    """Round price to tick size.

    Args:
        price: The price to round
        tick_size: The tick size to round to
        direction: 'down' for floor, 'up' for ceil, 'nearest' for round

    Returns:
        Rounded price (cleanly rounded to tick precision)
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")

    if direction == "down":
        value = math.floor(price / tick_size) * tick_size
    elif direction == "up":
        value = math.ceil(price / tick_size) * tick_size
    else:  # nearest
        value = round(price / tick_size) * tick_size

    # Avoid floating artifacts like 2929.3300000000004
    return round(value, _tick_decimals(tick_size))


def get_timeframe_ms(timeframe: str) -> int:
    """Get timeframe duration in milliseconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h')
    
    Returns:
        Duration in milliseconds
    
    Raises:
        ValueError: If timeframe is not supported
    """
    tf = timeframe.lower()
    if tf not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(TIMEFRAME_MS.keys())}")
    return TIMEFRAME_MS[tf]


def align_timestamp_to_timeframe(timestamp_ms: int, timeframe: str) -> int:
    """Align timestamp to the start of the timeframe period.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        timeframe: Timeframe string (e.g., '1m', '5m', '1h')
    
    Returns:
        Aligned timestamp in milliseconds
    """
    tf_ms = get_timeframe_ms(timeframe)
    return (timestamp_ms // tf_ms) * tf_ms


def get_day_start_ms(timestamp_ms: int) -> int:
    """Get the start of day (UTC) for a given timestamp.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
    
    Returns:
        Start of day timestamp in milliseconds
    """
    dt = ms_to_datetime(timestamp_ms)
    day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime_to_ms(day_start)


def get_yesterday_range_ms(timestamp_ms: int) -> tuple[int, int]:
    """Get yesterday's start and end timestamps (UTC).
    
    Args:
        timestamp_ms: Current timestamp in milliseconds
    
    Returns:
        Tuple of (yesterday_start_ms, yesterday_end_ms)
    """
    day_start = get_day_start_ms(timestamp_ms)
    yesterday_start = day_start - 86_400_000  # 24 hours in ms
    yesterday_end = day_start - 1
    return yesterday_start, yesterday_end


def format_price(price: float, decimals: int = 2) -> str:
    """Format price with specified decimal places."""
    return f"{price:.{decimals}f}"


def format_quantity(qty: float, decimals: int = 4) -> str:
    """Format quantity with specified decimal places."""
    return f"{qty:.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero division."""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    return ((new_value - old_value) / abs(old_value)) * 100
