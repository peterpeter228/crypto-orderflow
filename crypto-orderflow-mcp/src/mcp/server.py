"""MCP Server implementation.

This project supports two transports:

1) **Streamable HTTP** (recommended): POST JSON-RPC requests to `/mcp`.
2) **Legacy HTTP+SSE** (for clients like Cherry Studio's "SSE" mode):
   - Open an EventSource connection to `/sse`
   - The first SSE event is `endpoint`, whose `data` contains the POST URL to send
     JSON-RPC messages (e.g. `/messages/?session_id=...`).
   - Server replies to requests by emitting SSE events of type `message`.

Why both?
- Newer MCP specs recommend Streamable HTTP.
- Cherry Studio and some existing clients still use the older HTTP+SSE handshake.

"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from src.config import get_settings
from src import __version__
from src.utils import get_logger, timestamp_ms

from .tools import MCPTools


@dataclass
class _SseSession:
    """State for a legacy HTTP+SSE session."""

    session_id: str
    queue: "asyncio.Queue[dict[str, Any]]"
    created_at: int
    last_seen: int


def create_mcp_server(tools: MCPTools) -> tuple[FastAPI, None]:
    """Create FastAPI app with MCP tools and transports.

    Notes:
        We implement MCP's JSON-RPC surface directly to avoid depending on any
        particular python-mcp server wrapper behavior (and to be maximally
        compatible with CherryStudio and other clients).

    Returns:
        (FastAPI app, None)
    """

    settings = get_settings()
    logger = get_logger("mcp.server")

    # ------------------------------------------------------------------
    # Tool registry (JSON schema for tools/list)
    # ------------------------------------------------------------------

    TOOL_DEFS: list[dict[str, Any]] = [
        {
            "name": "get_market_snapshot",
            "description": (
                "Get real-time market snapshot including latest price, mark price, 24h stats, "
                "funding rate, and open interest for a trading pair."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_key_levels",
            "description": (
                "Get key price levels including developing VWAP (dVWAP), previous day VWAP (pdVWAP), "
                "Volume Profile (POC, VAH, VAL), and session high/low (Tokyo, London, NY)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                    },
                    "sessionTZ": {
                        "type": "string",
                        "description": "Session timezone (default: UTC)",
                        "default": "UTC",
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_footprint",
            "description": (
                "Footprint output. Default view='statistics' returns compact per-bar metrics "
                "(Vol/Delta/DeltaMax/DeltaMin) to avoid huge payloads. "
                "Set view='levels' to return detailed per-price levels."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "timeframe": {
                        "type": "string",
                        "description": "Candle timeframe",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        "default": "30m",
                    },
                    "startTime": {
                        "type": "integer",
                        "description": "Start timestamp in milliseconds (optional)",
                    },
                    "endTime": {
                        "type": "integer",
                        "description": "End timestamp in milliseconds (optional)",
                    },
                    "view": {
                        "type": "string",
                        "description": "statistics (default) or levels",
                        "enum": ["statistics", "levels"],
                        "default": "statistics",
                    },
                    "maxLevelsPerBar": {
                        "type": "integer",
                        "description": "When view='levels', cap returned price levels per bar (default 200)",
                        "default": 200,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_footprint_statistics",
            "description": "Compact footprint statistics per bar (Vol/Delta/DeltaMax/DeltaMin in quote).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "timeframe": {
                        "type": "string",
                        "description": "Candle timeframe",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        "default": "30m",
                    },
                    "startTime": {
                        "type": "integer",
                        "description": "Start timestamp in milliseconds (optional)",
                    },
                    "endTime": {
                        "type": "integer",
                        "description": "End timestamp in milliseconds (optional)",
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_orderflow_metrics",
            "description": (
                "Get orderflow metrics including delta bars, cumulative volume delta (CVD), "
                "and stacked imbalance detection. The returned currentCVD equals the last "
                "cvdSequence value for the requested window."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "timeframe": {
                        "type": "string",
                        "description": "Candle timeframe",
                        "enum": ["1m", "5m", "15m", "30m", "1h"],
                        "default": "15m",
                    },
                    "startTime": {"type": "integer", "description": "Start timestamp in milliseconds"},
                    "endTime": {"type": "integer", "description": "End timestamp in milliseconds"},
                },
                "required": ["symbol", "timeframe", "startTime", "endTime"],
            },
        },
        {
            "name": "get_orderbook_depth_delta",
            "description": (
                "Get orderbook depth delta showing bid/ask volume changes within a price range over time."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "percent": {
                        "type": "number",
                        "description": "Price range percentage from mid price (default: 1.0 for ±1%)",
                        "default": 1.0,
                    },
                    "windowSec": {
                        "type": "integer",
                        "description": "Output aggregation window in seconds (default: 60). Use 0 for raw snapshots.",
                        "default": 60,
                    },
                    "lookbackSec": {
                        "type": "integer",
                        "description": "Lookback period in seconds (default: 3600)",
                        "default": 3600,
                    },
                    "maxPoints": {
                        "type": "integer",
                        "description": "Max returned points (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_orderbook_heatmap",
            "description": (
                "Get a compact summary of the stored orderbook heatmap (top liquidity bins + coverage metadata)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "lookbackMinutes": {
                        "type": "integer",
                        "description": "Lookback window for coverage metadata (default: 180)",
                        "default": 180,
                    },
                    "maxLevels": {
                        "type": "integer",
                        "description": "How many top bid/ask bins to return (default: 15)",
                        "default": 15,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "stream_liquidations",
            "description": "Get recent liquidation events (cached from websocket forceOrder stream).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of liquidations to return (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_session_profile",
            "description": (
                "Session Volume Profile (vPOC/vVAH/vVAL) + Session totals (Vol/Delta) for configured sessions. "
                "Sessions come from env `SESSIONS` (default: Exocharts-style A/L/N/E). "
                "Profile is derived from local footprint_1m (requires backfill for historical)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "date": {
                        "type": "string",
                        "description": "UTC date in YYYY-MM-DD (optional, default today)",
                    },
                    "session": {
                        "type": "string",
                        "description": "Session name (as configured by `SESSIONS`), or 'all'",
                        "default": "all",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Kline interval used for totals (default 15m)",
                        "enum": ["1m", "5m", "15m", "30m", "1h"],
                        "default": "15m",
                    },
                    "valueAreaPercent": {
                        "type": "number",
                        "description": "Value area percentage (default 70)",
                        "default": 70.0,
                    },
                    "includeProfileLevels": {
                        "type": "boolean",
                        "description": "If true, include distribution levels (can be large)",
                        "default": False,
                    },
                    "maxProfileLevels": {
                        "type": "integer",
                        "description": "Cap returned profile levels (default 400)",
                        "default": 400,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_tpo_profile",
            "description": (
                "TPO Profile (Market Profile / Time-Price-Opportunity). Returns both time-based and volume-based "
                "POC/VA markers. Set useVolumeForVA=true to mimic Exocharts option: 'Use VOLUME for TPO POC+VA calculation'."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "date": {
                        "type": "string",
                        "description": "UTC date YYYY-MM-DD (optional, default today)",
                    },
                    "session": {
                        "type": "string",
                        "description": "Session name (as configured by `SESSIONS`), or 'all'",
                        "default": "all",
                    },
                    "periodMinutes": {
                        "type": "integer",
                        "description": "TPO period in minutes (default 30)",
                        "default": 30,
                    },
                    "tickSize": {
                        "type": "number",
                        "description": "Optional price bucket size for TPO levels (e.g. 50 for BTC daily MP)",
                    },
                    "valueAreaPercent": {
                        "type": "number",
                        "description": "Value area percentage (default 70)",
                        "default": 70.0,
                    },
                    "useVolumeForVA": {
                        "type": "boolean",
                        "description": "If true, use volume (notional) distribution for POC/VA calculation. If omitted, defaults to env TPO_USE_VOLUME_FOR_VA_DEFAULT.",
                        "default": bool(settings.tpo_use_volume_for_va_default),
                    },
                    "includeLevels": {
                        "type": "boolean",
                        "description": "If true, include price-level distribution (can be large)",
                        "default": False,
                    },
                    "maxLevels": {
                        "type": "integer",
                        "description": "When includeLevels=true, cap returned levels (default 240)",
                        "default": 240,
                    },
                    "includePeriodProfiles": {
                        "type": "boolean",
                        "description": "If true, include per-period (e.g., 30m) volume POC/VAH/VAL and stats",
                        "default": True,
                    },
                    "includeSinglePrints": {
                        "type": "boolean",
                        "description": "If true, compute single prints and tails using period brackets",
                        "default": True,
                    },
                    "singlePrintsMode": {
                        "type": "string",
                        "description": "Single prints output size: 'compact' (ranges + small tails) or 'full' (raw level arrays)",
                        "enum": ["compact", "full"],
                        "default": "compact",
                    },
                    "tailMinLength": {
                        "type": "integer",
                        "description": "Minimum consecutive single prints to qualify as a tail (default 2)",
                        "default": 2,
                    },
                    "ibPeriods": {
                        "type": "integer",
                        "description": "Initial Balance periods (default 2). Used for IB range stats.",
                        "default": 2,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_swing_liquidity",
            "description": (
                "Swing Liquidity levels based on pivot highs/lows (buy-side and sell-side liquidity)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "interval": {
                        "type": "string",
                        "description": "Kline interval (default 15m)",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h"],
                        "default": "15m",
                    },
                    "lookbackBars": {
                        "type": "integer",
                        "description": "Number of bars to analyze (default 300)",
                        "default": 300,
                    },
                    "pivotLeft": {
                        "type": "integer",
                        "description": "Left pivot window (default 10)",
                        "default": 10,
                    },
                    "pivotRight": {
                        "type": "integer",
                        "description": "Right pivot window (default 15)",
                        "default": 15,
                    },
                    "activeOnly": {
                        "type": "boolean",
                        "description": "If true, only return unswept (active) liquidity levels",
                        "default": False,
                    },
                    "maxLevels": {
                        "type": "integer",
                        "description": "Max returned levels (default 150)",
                        "default": 150,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_market_structure",
            "description": "Market structure: swings, trend, BOS/CHoCH, optional ZigZag.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "interval": {
                        "type": "string",
                        "description": "Kline interval (default 15m)",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h"],
                        "default": "15m",
                    },
                    "lookbackBars": {
                        "type": "integer",
                        "description": "Number of bars to analyze (default 400)",
                        "default": 400,
                    },
                    "pivotLeft": {
                        "type": "integer",
                        "description": "Swing pivot left window (default 10)",
                        "default": 10,
                    },
                    "pivotRight": {
                        "type": "integer",
                        "description": "Swing pivot right window (default 10)",
                        "default": 10,
                    },
                    "zigzagLegMinPercent": {
                        "type": "number",
                        "description": "Optional ZigZag reversal threshold in percent (e.g. 0.5)",
                    },
                    "maxPoints": {
                        "type": "integer",
                        "description": "Cap swing/zigzag points returned (default 80)",
                        "default": 80,
                    },
                    "endTime": {
                        "type": "integer",
                        "description": "Optional ms timestamp window end (UTC)",
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_open_interest",
            "description": "Get current and recent Open Interest (OI) data.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"},
                    "period": {
                        "type": "string",
                        "description": "Time period",
                        "enum": ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
                        "default": "5m",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of data points (default 100)",
                        "default": 100,
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_funding_rate",
            "description": "Get current funding rate and recent funding rate history.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair symbol"}
                },
                "required": ["symbol"],
            },
        },
    ]

    async def _handle_list_tools() -> dict[str, Any]:
        return {"tools": TOOL_DEFS}

    async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tools/call request to the concrete MCPTools method."""

        if not isinstance(arguments, dict):
            arguments = {}

        if name == "get_market_snapshot":
            return await tools.get_market_snapshot(arguments["symbol"])

        if name == "get_key_levels":
            return await tools.get_key_levels(
                arguments["symbol"],
                date=arguments.get("date"),
                session_tz=arguments.get("sessionTZ", "UTC"),
            )

        if name == "get_footprint":
            return await tools.get_footprint(
                arguments["symbol"],
                timeframe=arguments.get("timeframe", "30m"),
                start_time=arguments.get("startTime"),
                end_time=arguments.get("endTime"),
                view=arguments.get("view", "statistics"),
                max_levels_per_bar=int(arguments.get("maxLevelsPerBar", 200) or 200),
            )

        if name == "get_footprint_statistics":
            return await tools.get_footprint_statistics(
                arguments["symbol"],
                timeframe=arguments.get("timeframe", "30m"),
                start_time=arguments.get("startTime"),
                end_time=arguments.get("endTime"),
            )

        if name == "get_orderflow_metrics":
            return await tools.get_orderflow_metrics(
                arguments["symbol"],
                timeframe=arguments["timeframe"],
                start_time=int(arguments["startTime"]),
                end_time=int(arguments["endTime"]),
            )

        if name == "get_orderbook_depth_delta":
            lookback = arguments.get("lookbackSec", arguments.get("lookback", 3600))
            return await tools.get_orderbook_depth_delta(
                arguments["symbol"],
                percent=float(arguments.get("percent", 1.0)),
                window_sec=int(arguments.get("windowSec", 60)),
                lookback=int(lookback or 3600),
                max_points=int(arguments.get("maxPoints", 30)),
            )

        if name == "get_orderbook_heatmap":
            return await tools.get_orderbook_heatmap(
                arguments["symbol"],
                lookback_minutes=int(arguments.get("lookbackMinutes", 180)),
                max_levels=int(arguments.get("maxLevels", 15)),
            )

        if name == "stream_liquidations":
            return await tools.stream_liquidations(
                arguments["symbol"],
                limit=int(arguments.get("limit", 100)),
            )

        if name == "get_session_profile":
            return await tools.get_session_profile(
                arguments["symbol"],
                date=arguments.get("date"),
                session=arguments.get("session", "all"),
                interval=arguments.get("interval", "15m"),
                value_area_percent=float(arguments.get("valueAreaPercent", 70.0)),
                include_profile_levels=bool(arguments.get("includeProfileLevels", False)),
                max_profile_levels=int(arguments.get("maxProfileLevels", 400)),
            )

        if name == "get_tpo_profile":
            tick = arguments.get("tickSize")
            tick_size = float(tick) if tick is not None else None
            return await tools.get_tpo_profile(
                arguments["symbol"],
                date=arguments.get("date"),
                session=arguments.get("session", "all"),
                period_minutes=int(arguments.get("periodMinutes", 30)),
                tick_size=tick_size,
                value_area_percent=float(arguments.get("valueAreaPercent", 70.0)),
                use_volume_for_va=bool(arguments.get("useVolumeForVA", settings.tpo_use_volume_for_va_default)),
                include_levels=bool(arguments.get("includeLevels", False)),
                max_levels=int(arguments.get("maxLevels", 240)),
                include_period_profiles=bool(arguments.get("includePeriodProfiles", True)),
                include_single_prints=bool(arguments.get("includeSinglePrints", True)),
                single_prints_mode=str(arguments.get("singlePrintsMode", "compact")),
                tail_min_len=int(arguments.get("tailMinLength", 2)),
                ib_periods=int(arguments.get("ibPeriods", 2)),
            )

        if name == "get_swing_liquidity":
            return await tools.get_swing_liquidity(
                arguments["symbol"],
                interval=arguments.get("interval", "15m"),
                lookback_bars=int(arguments.get("lookbackBars", 300)),
                pivot_left=int(arguments.get("pivotLeft", 10)),
                pivot_right=int(arguments.get("pivotRight", 15)),
                active_only=bool(arguments.get("activeOnly", False)),
                max_levels=int(arguments.get("maxLevels", 150)),
            )

        if name == "get_market_structure":
            zig = arguments.get("zigzagLegMinPercent")
            zigzag = float(zig) if zig is not None else None
            end_time = arguments.get("endTime")
            end_ms = int(end_time) if end_time is not None else None
            return await tools.get_market_structure(
                arguments["symbol"],
                interval=arguments.get("interval", "15m"),
                lookback_bars=int(arguments.get("lookbackBars", 400)),
                pivot_left=int(arguments.get("pivotLeft", 10)),
                pivot_right=int(arguments.get("pivotRight", 10)),
                zigzag_leg_min_percent=zigzag,
                max_points=int(arguments.get("maxPoints", 80)),
                end_time=end_ms,
            )

        if name == "get_open_interest":
            return await tools.get_open_interest(
                arguments["symbol"],
                period=arguments.get("period", "5m"),
                limit=int(arguments.get("limit", 100)),
            )

        if name == "get_funding_rate":
            return await tools.get_funding_rate(arguments["symbol"])

        raise ValueError(f"Unknown tool: {name}")

    async def _handle_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not tool_name:
            raise ValueError("tools/call missing params.name")

        result = await _dispatch_tool(tool_name, arguments)
        # MCP tool call results are a list of content blocks.
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                }
            ]
        }

    async def _process_one_jsonrpc(body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process a single JSON-RPC message.

        Returns:
            - A JSON-RPC response dict when the input is a request.
            - None for notifications.

        Notes:
            MCP sends notifications such as `notifications/initialized` (no `id`).
            Those must *not* produce a response.
        """

        method = body.get("method")
        params = body.get("params", {}) or {}
        request_id = body.get("id", None)
        is_notification = ("id" not in body) or (request_id is None)

        # Common MCP notifications
        if isinstance(method, str) and method.startswith("notifications/"):
            logger.debug("mcp_notification", method=method)
            return None

        try:
            if method == "tools/list":
                result = await _handle_list_tools()
            elif method == "tools/call":
                result = await _handle_tool_call(params.get("name"), params.get("arguments", {}) or {})
            elif method == "initialize":
                # Legacy protocol version used by many clients (including Cherry Studio)
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "crypto-orderflow-mcp", "version": __version__},
                    "capabilities": {"tools": {}},
                }
            else:
                # Unknown method
                if is_notification:
                    # Ignore unknown notifications instead of failing the session
                    logger.debug("unknown_notification_ignored", method=method)
                    return None
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            if is_notification:
                return None

            return {"jsonrpc": "2.0", "id": request_id, "result": result}

        except Exception as e:
            logger.error("jsonrpc_processing_error", method=method, error=str(e))
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)},
            }

    async def _process_jsonrpc(body: Any) -> Optional[Any]:
        """Process a JSON-RPC request or batch.

        Returns:
            - dict response for single request
            - list of dict responses for batch requests
            - None for notifications-only payloads
        """

        if isinstance(body, list):
            responses = []
            for item in body:
                if not isinstance(item, dict):
                    continue
                r = await _process_one_jsonrpc(item)
                if r is not None:
                    responses.append(r)
            return responses if responses else None

        if not isinstance(body, dict):
            raise ValueError("Invalid JSON-RPC payload")

        return await _process_one_jsonrpc(body)

    # -------------------------
    # FastAPI app
    # -------------------------

    app = FastAPI(
        title="Crypto Orderflow MCP Server",
        description="Market Data & Orderflow Indicators for Binance USD-M Futures",
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------
    # Legacy SSE sessions
    # -------------------------

    sessions: Dict[str, _SseSession] = {}
    sessions_lock = asyncio.Lock()

    async def _queue_put_safe(q: "asyncio.Queue[dict[str, Any]]", msg: dict[str, Any]) -> None:
        """Put without deadlocking if the client is slow.

        Strategy:
        - If the queue is full, drop the oldest message and insert the newest.
        """

        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                _ = q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # Give up; at least don't block request handlers.
                logger.warning("sse_queue_full_drop", dropped=True)

    def _ping_message() -> ServerSentEvent:
        # Send a lightweight ping event so both proxies *and* some SSE clients keep the stream alive.
        return ServerSentEvent(event="ping", data="{}")

    @app.get("/healthz")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": timestamp_ms(),
            "service": "crypto-orderflow-mcp",
            "version": __version__,
        }

    @app.get("/")
    async def root():
        return {
            "name": "Crypto Orderflow MCP Server",
            "version": __version__,
            "description": "Market Data & Orderflow Indicators for Binance USD-M Futures",
            "endpoints": {
                "sse": "/sse",
                "messages": "/messages/?session_id=...",
                "mcp": "/mcp",
                "health": "/healthz",
            },
            "tools": [t["name"] for t in TOOL_DEFS],
            "symbols": settings.symbol_list,
        }

    # -------------------------
    # Legacy HTTP+SSE transport
    # -------------------------

    @app.get("/sse")
    @app.get("/sse/")
    async def sse_endpoint(request: Request):
        """Legacy MCP HTTP+SSE connect endpoint.

        Cherry Studio expects this endpoint to:
        1) Keep an SSE connection open.
        2) Emit an `endpoint` event first, where data is the POST URL to send JSON-RPC.
        3) Emit `message` events for JSON-RPC responses.
        """

        session_id = secrets.token_hex(16)
        q: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue(maxsize=1000)
        now = timestamp_ms()

        async with sessions_lock:
            sessions[session_id] = _SseSession(
                session_id=session_id,
                queue=q,
                created_at=now,
                last_seen=now,
            )

        logger.info("sse_connection_started", session_id=session_id)

        async def event_generator():
            try:
                # The first event MUST be `endpoint`
                # Some MCP clients (incl. CherryStudio) expect an **absolute** URL here.
                base = (settings.mcp_public_url or str(request.base_url)).rstrip("/")
                yield ServerSentEvent(
                    event="endpoint",
                    data=f"{base}/messages?session_id={session_id}",
                    retry=3000,
                )

                # Stream outgoing messages
                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        msg = await asyncio.wait_for(
                            q.get(), timeout=max(5, int(settings.mcp_sse_ping_interval_sec))
                        )
                    except asyncio.TimeoutError:
                        yield _ping_message()
                        continue

                    yield ServerSentEvent(
                        event="message",
                        data=json.dumps(msg, separators=(",", ":")),
                    )

            except asyncio.CancelledError:
                raise
            finally:
                async with sessions_lock:
                    sessions.pop(session_id, None)
                logger.info("sse_connection_closed", session_id=session_id)

        return EventSourceResponse(
            event_generator(),
            ping=settings.mcp_sse_ping_interval_sec,
            ping_message_factory=_ping_message,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/messages")
    @app.post("/messages/")
    async def messages_endpoint(
        request: Request,
        session_id: str = Query(..., description="Legacy SSE session id"),
    ):
        """Legacy MCP message endpoint.

        Clients POST JSON-RPC messages to the URL provided by the `endpoint` SSE event.
        Responses are emitted back on the SSE stream as `message` events.
        """

        async with sessions_lock:
            session = sessions.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Unknown or expired SSE session")

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        session.last_seen = timestamp_ms()

        response_payload = await _process_jsonrpc(body)
        if response_payload is not None:
            # For batch responses, enqueue each item as its own SSE message.
            if isinstance(response_payload, list):
                for item in response_payload:
                    if isinstance(item, dict):
                        await _queue_put_safe(session.queue, item)
            elif isinstance(response_payload, dict):
                await _queue_put_safe(session.queue, response_payload)

        # In legacy SSE transport, HTTP responses are typically 202 (responses go over SSE)
        return Response(status_code=202)

    # -------------------------
    # Streamable HTTP transport
    # -------------------------

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        """Streamable HTTP MCP endpoint.

        For maximum compatibility, we return JSON responses (application/json).
        Notifications return 202.
        """

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
            )

        try:
            response_payload = await _process_jsonrpc(body)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": str(e)},
                },
            )

        if response_payload is None:
            return Response(status_code=202)

        return JSONResponse(content=response_payload)

    return app, None
