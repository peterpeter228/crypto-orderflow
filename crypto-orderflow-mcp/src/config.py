"""Configuration management for Crypto Orderflow MCP Server."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SessionTime:
    """Parse session time string like '00:00-09:00' into start/end hours."""
    
    def __init__(self, time_str: str):
        parts = time_str.split("-")
        start_parts = parts[0].split(":")
        end_parts = parts[1].split(":")
        self.start_hour = int(start_parts[0])
        self.start_minute = int(start_parts[1])
        self.end_hour = int(end_parts[0])
        self.end_minute = int(end_parts[1])
    
    @property
    def start_minutes(self) -> int:
        return self.start_hour * 60 + self.start_minute
    
    @property
    def end_minutes(self) -> int:
        # Treat 00:00 as 24:00 so ranges like 20:00-00:00 are interpreted
        # as ending at the end of the current day (not at its start).
        if self.end_hour == 0 and self.end_minute == 0:
            return 24 * 60
        return self.end_hour * 60 + self.end_minute

    def contains_minute(self, minute_of_day: int) -> bool:
        """Return True if minute_of_day (0-1439) is within this session."""
        start = self.start_minutes
        end = self.end_minutes
        if end > start:
            return start <= minute_of_day < end
        # Spans midnight (e.g. 20:00-02:00)
        return minute_of_day >= start or minute_of_day < end


@dataclass(frozen=True)
class SessionDef:
    """A named intraday session (UTC by default)."""

    name: str
    time: SessionTime

    @property
    def start_minutes(self) -> int:
        return self.time.start_minutes

    @property
    def end_minutes(self) -> int:
        return self.time.end_minutes

    @property
    def spans_midnight(self) -> bool:
        return self.end_minutes <= self.start_minutes


_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def parse_sessions(value: str) -> list[SessionDef]:
    """Parse session list from env.

    Supported formats:
      - "A=00:00-06:00,L=06:00-12:00"  (recommended)
      - "A:00:00-06:00,L:06:00-12:00"  (also supported)

    Returns:
        List[SessionDef] in the given order.
    """

    out: list[SessionDef] = []
    value = (value or "").strip()
    if not value:
        return out

    parts = [p.strip() for p in value.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            name, rng = p.split("=", 1)
        elif ":" in p:
            name, rng = p.split(":", 1)
        else:
            raise ValueError(
                f"Invalid session definition '{p}'. Use NAME=HH:MM-HH:MM (e.g. A=00:00-06:00)."
            )

        name = name.strip()
        rng = rng.strip()
        if not name or not _SESSION_NAME_RE.match(name):
            raise ValueError(
                f"Invalid session name '{name}'. Allowed: letters/numbers/underscore."
            )
        out.append(SessionDef(name=name, time=SessionTime(rng)))
    return out


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )
    
    # Server Settings
    mcp_host: str = Field(default="0.0.0.0", description="MCP Server host")
    mcp_port: int = Field(default=8022, description="MCP Server port")
    mcp_public_url: str | None = Field(
        default=None,
        description=(
            "Public base URL used in the legacy /sse handshake (endpoint event). "
            "Set this when running behind a reverse proxy / NAT so clients receive a reachable URL."
        ),
    )
    # NOTE: Some remote clients / proxies (incl. CherryStudio remote connections)
    # may aggressively close idle HTTP streams. A shorter ping interval helps
    # keep the SSE connection stable.
    mcp_sse_ping_interval_sec: int = Field(
        default=10,
        description="SSE keepalive ping interval (seconds) for legacy /sse transport",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    debug: bool = Field(default=False)
    
    # Binance API
    binance_rest_url: str = Field(default="https://fapi.binance.com")
    binance_ws_url: str = Field(default="wss://fstream.binance.com")
    # Public historical data mirror (used to avoid Binance REST 429s during backfill)
    binance_vision_base_url: str = Field(
        default="https://data.binance.vision",
        description="Base URL for Binance Vision historical datasets",
    )
    vision_cache_dir: str = Field(
        default="./data/binance_vision_cache",
        description="Local cache dir for downloaded Binance Vision zip files",
    )
    binance_api_key: str | None = Field(default=None)
    binance_api_secret: str | None = Field(default=None)
    
    # Symbols
    symbols: str = Field(default="BTCUSDT,ETHUSDT")
    
    # Database
    cache_db_path: str = Field(default="./data/orderflow_cache.db")
    data_retention_days: int = Field(default=7)
    
    # ------------------------------------------------------------------
    # Sessions (UTC)
    #
    # Exocharts "Main sessions" default (non-overlapping):
    #   A: 00:00-06:00
    #   L: 06:00-12:00
    #   N: 12:00-20:00
    #   E: 20:00-00:00
    #
    # You can override by setting SESSIONS, format:
    #   SESSIONS=A=00:00-06:00,L=06:00-12:00,N=12:00-20:00,E=20:00-00:00
    # ------------------------------------------------------------------
    sessions: str = Field(
        default="A=00:00-06:00,L=06:00-12:00,N=12:00-20:00,E=20:00-00:00",
        description="Session definitions, format NAME=HH:MM-HH:MM,... (UTC)",
    )

    # Legacy (overlapping) sessions kept for backward compatibility.
    # Only used when SESSIONS is set to an empty string.
    tokyo_session: str = Field(default="00:00-09:00")
    london_session: str = Field(default="07:00-16:00")
    ny_session: str = Field(default="13:00-22:00")
    
    # Orderflow Configuration
    default_timeframe: str = Field(default="1m")
    footprint_tick_size_btc: float = Field(default=0.1)
    footprint_tick_size_eth: float = Field(default=0.01)
    # TPO / Market Profile tick sizes (coarser than footprint ticks)
    tpo_tick_size_btc: float = Field(
        default=70.0,
        description="Price bucket size for BTC TPO profiles (e.g., 70 = 70 USDT step)",
    )
    tpo_tick_size_eth: float = Field(
        default=5.0,
        description="Price bucket size for ETH TPO profiles (coarser than footprint ticks)",
    )
    imbalance_ratio_threshold: float = Field(default=3.0)
    imbalance_consecutive_levels: int = Field(default=3)
    # TPO / Market Profile
    tpo_use_volume_for_va_default: bool = Field(
        default=False,
        description="Default for TPO POC+VA calculation: True uses volume (notional) distribution (Exocharts option).",
    )

    
    # Orderbook Configuration
    orderbook_depth_percent: float = Field(default=1.0)
    orderbook_update_interval_sec: int = Field(default=5)
    orderbook_snapshot_limit: int = Field(default=1000)

    # Optional: Orderbook heatmap ("liquidity surface") snapshots.
    # This is inspired by projects like flowsurface. We store a binned depth ladder
    # periodically so an external UI can render a heatmap.
    heatmap_enabled: bool = Field(default=False)
    heatmap_lookback_minutes: int = Field(
        default=180,
        description="Default lookback window (minutes) for heatmap metadata coverage.",
    )
    heatmap_interval_sec: int = Field(default=60, description="Heatmap snapshot interval")
    heatmap_sample_interval_ms: int = Field(
        default=15_000,
        description="Metadata sampler interval (ms) when heatmap is enabled.",
    )
    # Price bin size in quote units (USDT). For BTCUSDT, 10 means "10 USDT per bin".
    heatmap_bin_ticks: int = Field(default=10)
    # How far around mid price to record (percent). If 1.0 => +/-1%.
    heatmap_depth_percent: float = Field(default=1.0)
    
    # Liquidation
    liquidation_cache_size: int = Field(default=1000)
    
    # Rate Limiting
    # NOTE: Binance enforces multiple limits; this value is our local token bucket.
    rest_rate_limit_per_min: int = Field(default=600)
    ws_reconnect_delay_sec: int = Field(default=5)
    ws_max_reconnect_attempts: int = Field(default=10)

    # Historical Backfill
    #
    # The orderflow indicators in this project (Key Levels, VWAP, developing/previous-day
    # volume profile, delta/CVD, footprint) depend on having trades for the requested time range.
    # If you start the server mid-session without backfilling, many fields will be `null` or
    # reflect only the last few minutes of data.
    backfill_enabled: bool = Field(default=True)
    # Backfill source:
    # - auto: use Binance Vision for full past days when available, REST for the remaining ranges
    # - vision: force Binance Vision downloads (falls back to REST if file not found)
    # - rest: REST only
    backfill_source: Literal["auto", "vision", "rest"] = Field(default="auto")
    # How many hours of aggTrades to backfill on startup (covers yesterday+today by default)
    backfill_lookback_hours: int = Field(default=36)
    # Safety cap for how many aggTrades requests we allow per symbol during one startup backfill.
    # Set to 0 to disable the cap.
    backfill_max_requests_per_symbol: int = Field(default=0)

    # Backfill behavior
    # If true, the FastAPI startup will wait for the backfill to complete.
    # If false (recommended), the server starts immediately and backfill runs in background.
    backfill_block_startup: bool = Field(default=False)

    # If true, delete existing cached aggregates for the selected backfill days and rebuild from Binance.
    # If false (recommended), only missing days will be backfilled.
    backfill_rebuild: bool = Field(default=False)
    # Upper bound for how long we will block FastAPI startup while running backfill.
    # Set to 0 to disable the guard. Helps avoid being stuck on repeated 429s when BACKFILL_BLOCK_STARTUP=true.
    backfill_block_startup_timeout_ms: int = Field(default=300_000)

    # When using startTime+endTime on Binance aggTrades, the window must be <= 1 hour.
    # We chunk the backfill range into windows of this size (minutes). Max 60.
    backfill_chunk_minutes: int = Field(default=60)

    # Pause between aggTrades REST requests during startup backfill.
    # Some shared IPs / VPS providers can hit Binance anti-abuse heuristics if you hammer the
    # endpoint at full speed. A small pause also helps keep the server responsive on boot.
    backfill_request_pause_ms: int = Field(default=250)

    # (heatmap settings declared above under Orderbook Configuration)
    
    @property
    def symbol_list(self) -> list[str]:
        """Get symbols as list."""
        return [s.strip().upper() for s in self.symbols.split(",")]
    
    @property
    def tokyo(self) -> SessionTime:
        return SessionTime(self.tokyo_session)
    
    @property
    def london(self) -> SessionTime:
        return SessionTime(self.london_session)
    
    @property
    def ny(self) -> SessionTime:
        return SessionTime(self.ny_session)

    @property
    def session_defs(self) -> list[SessionDef]:
        """Return configured sessions.

        By default this matches Exocharts "Main sessions" (A/L/N/E).
        If you want the legacy Tokyo/London/NY windows, set `SESSIONS=` (empty)
        in your .env and configure TOKYO_SESSION/LONDON_SESSION/NY_SESSION.
        """

        parsed = parse_sessions(self.sessions)
        if parsed:
            return parsed

        # Fallback to legacy 3-session config
        return [
            SessionDef(name="tokyo", time=self.tokyo),
            SessionDef(name="london", time=self.london),
            SessionDef(name="ny", time=self.ny),
        ]

    @property
    def session_name_map(self) -> dict[str, SessionDef]:
        """Map normalized session name -> SessionDef."""
        return {s.name.lower(): s for s in self.session_defs}
    
    def get_tick_size(self, symbol: str) -> float:
        """Get tick size for footprint aggregation based on symbol."""
        symbol = symbol.upper()
        if "BTC" in symbol:
            return float(self.footprint_tick_size_btc)
        if "ETH" in symbol:
            return float(self.footprint_tick_size_eth)
        # Fallback to a reasonable default if the legacy attribute is absent
        # (older envs/configs may not define it).
        if hasattr(self, "footprint_tick_size_default"):
            return float(self.footprint_tick_size_default)
        return 0.1

    def get_tpo_tick_size(self, symbol: str) -> float:
        """Get tick size for TPO profile aggregation based on symbol.

        Note: This is intentionally *separate* from footprint tick size.
        Footprints typically need fine-grained ticks (e.g. 0.1), while TPO
        profiles are usually built on a much coarser price step.
        """
        symbol = symbol.upper()
        has_btc_tick = hasattr(self, "tpo_tick_size_btc")
        has_eth_tick = hasattr(self, "tpo_tick_size_eth")

        if "BTC" in symbol and has_btc_tick:
            return float(self.tpo_tick_size_btc)
        elif "ETH" in symbol and has_eth_tick:
            return float(self.tpo_tick_size_eth)

        # Fallback: derive a reasonable coarse step from the symbol's native tick.
        # (Better than defaulting to 0.1 for unknown symbols.)
        base_tick = float(self.get_tick_size(symbol))
        return max(base_tick * 50.0, base_tick)
    
    def ensure_data_dir(self) -> Path:
        """Ensure data directory exists."""
        db_path = Path(self.cache_db_path)
        # If a relative path is provided (default), resolve it against the project root so the
        # cache DB is stable even when the process is launched from different working directories
        # (e.g. some MCP clients / process managers).
        if not db_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            db_path = project_root / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure Binance Vision cache dir exists (used by the improved backfill).
        vision_dir = Path(self.vision_cache_dir)
        if not vision_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            vision_dir = project_root / vision_dir
        vision_dir.mkdir(parents=True, exist_ok=True)
        return db_path

    def ensure_vision_cache_dir(self) -> Path:
        """Ensure Binance Vision cache directory exists and return absolute path."""
        cache_path = Path(self.vision_cache_dir)
        if not cache_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            cache_path = project_root / cache_path
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance."""
    return settings
