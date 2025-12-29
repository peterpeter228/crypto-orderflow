"""Binance REST API client for USD-M Futures.

This project is typically deployed on VPS instances and accessed remotely via SSE.
We therefore need a REST rate limiter that:

1) Never blocks FastAPI startup for long periods.
2) Does not accidentally throttle to ~1 request/min because of shared-IP headers.
3) Produces smooth request pacing (avoid "burn all tokens then sleep 55s" bursts).

v8 used the Binance `x-mbx-used-weight-1m` header to overwrite local remaining tokens.
On shared IPs (or when other processes on the host also call Binance), this header can
already be close to the limit, causing our local limiter to think we are out of tokens
and sleep ~60s repeatedly.

In v9 we switch to a simple *token bucket* that only limits *our own* request rate.
Real Binance limits are still respected via 418/429 handling.
"""

import asyncio
import time
from typing import Any

import aiohttp

from src.config import get_settings
from src.utils import get_logger, timestamp_ms
from .types import (
    AggTrade,
    FundingRate,
    Kline,
    OpenInterest,
    OpenInterestHist,
    OrderbookLevel,
    OrderbookSnapshot,
    Ticker24h,
    MarkPrice,
)


class BinanceRestClient:
    """Async REST client for Binance USD-M Futures API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.binance_rest_url
        self.logger = get_logger("binance.rest")
        self._session: aiohttp.ClientSession | None = None
        # Optional API credentials.
        # Note: most market-data endpoints do NOT require a key; it is used only when present.
        self.api_key: str | None = (self.settings.binance_api_key or "").strip() or None
        self.api_secret: str | None = (self.settings.binance_api_secret or "").strip() or None
        # ------------------------------------------------------------------
        # Local (client-side) rate limiter
        # ------------------------------------------------------------------
        # We use a token bucket with continuous refill.
        # - capacity = REST_RATE_LIMIT_PER_MIN
        # - refill rate = capacity / 60 tokens per second
        # This produces smooth pacing and avoids long sleeps.
        self._rate_capacity = max(1, int(self.settings.rest_rate_limit_per_min))
        self._rate_tokens = float(self._rate_capacity)
        self._rate_refill_per_sec = float(self._rate_capacity) / 60.0
        self._rate_last_refill = time.monotonic()
        self._rate_lock = asyncio.Lock()
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


    async def _acquire_rate_limit(self, weight: int = 1) -> None:
        """Acquire a local rate-limit token.

        Waiting for the next rate-limit window should NOT consume retry attempts.
        """
        weight = max(1, int(weight))

        # NOTE: Waiting for local tokens should NOT consume retry attempts.
        while True:
            async with self._rate_lock:
                now = time.monotonic()
                elapsed = now - self._rate_last_refill
                if elapsed > 0:
                    self._rate_tokens = min(
                        float(self._rate_capacity),
                        self._rate_tokens + elapsed * self._rate_refill_per_sec,
                    )
                    self._rate_last_refill = now

                if self._rate_tokens >= weight:
                    self._rate_tokens -= weight
                    return

                # How long until we have enough tokens?
                missing = float(weight) - self._rate_tokens
                if self._rate_refill_per_sec <= 0:
                    wait_seconds = 1.0
                else:
                    wait_seconds = missing / self._rate_refill_per_sec

                wait_seconds = max(0.05, wait_seconds)

            # Sleep outside the lock
            if wait_seconds >= 1.0:
                self.logger.warning("rate_limit_wait", wait_seconds=wait_seconds)
            else:
                self.logger.debug("rate_limit_wait", wait_seconds=wait_seconds)
            await asyncio.sleep(wait_seconds)
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an API request with basic retry/backoff.

        We mainly call public endpoints, but an API key (if provided) increases rate limits in practice
        and is also required for some endpoints (e.g. allForceOrders).
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else None
        max_attempts = 6

        for attempt in range(1, max_attempts + 1):
            # Acquire local rate-limit token (does not consume retry attempts)
            await self._acquire_rate_limit(weight=1)

            try:
                async with session.request(method, url, params=params, headers=headers) as response:
                    # NOTE (v9): We intentionally do NOT overwrite the local limiter based on
                    # `x-mbx-used-weight-1m` because it may reflect shared-IP usage.
                    # Real limits are enforced by Binance via 418/429.

                    # Retry on rate-limit and transient server errors
                    if response.status in (418, 429):
                        retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
                        wait = 0.0
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except ValueError:
                                wait = 0.0
                        if wait <= 0:
                            wait = min(60.0, 2.0 ** attempt)
                        self.logger.warning("rate_limited", status=response.status, wait_seconds=wait, attempt=attempt)
                        await asyncio.sleep(wait)
                        continue

                    if 500 <= response.status < 600:
                        wait = min(60.0, 2.0 ** attempt)
                        self.logger.warning("server_error_retry", status=response.status, wait_seconds=wait, attempt=attempt)
                        await asyncio.sleep(wait)
                        continue

                    if response.status >= 400:
                        # IMPORTANT:
                        # - 418/429 are handled above (rate limiting)
                        # - 5xx are handled above (transient)
                        # For other 4xx, retrying is usually useless and can spam logs.
                        body = await response.text()
                        self.logger.error(
                            "request_failed",
                            endpoint=endpoint,
                            status=response.status,
                            params=params,
                            body=body[:2000],
                        )
                        raise RuntimeError(
                            f"Binance API error {response.status} for {endpoint}: {body[:2000]}"
                        )

                    return await response.json()

            except aiohttp.ClientError as e:
                wait = min(60.0, 2.0 ** attempt)
                self.logger.warning("client_error_retry", endpoint=endpoint, error=str(e), attempt=attempt, wait_seconds=wait)
                await asyncio.sleep(wait)

        raise RuntimeError(f"Binance request failed after {max_attempts} attempts: {endpoint}")

    async def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        return await self._request("GET", "/fapi/v1/exchangeInfo")
    
    async def get_ticker_24h(self, symbol: str) -> Ticker24h:
        """Get 24hr ticker statistics."""
        data = await self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})
        return Ticker24h(
            symbol=data["symbol"],
            price_change=float(data["priceChange"]),
            price_change_percent=float(data["priceChangePercent"]),
            weighted_avg_price=float(data["weightedAvgPrice"]),
            last_price=float(data["lastPrice"]),
            last_qty=float(data["lastQty"]),
            open_price=float(data["openPrice"]),
            high_price=float(data["highPrice"]),
            low_price=float(data["lowPrice"]),
            volume=float(data["volume"]),
            quote_volume=float(data["quoteVolume"]),
            open_time=data["openTime"],
            close_time=data["closeTime"],
            first_trade_id=data["firstId"],
            last_trade_id=data["lastId"],
            trade_count=data["count"],
        )
    
    async def get_mark_price(self, symbol: str) -> MarkPrice:
        """Get mark price and funding rate."""
        data = await self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return MarkPrice(
            symbol=data["symbol"],
            mark_price=float(data["markPrice"]),
            index_price=float(data["indexPrice"]),
            estimated_settle_price=float(data.get("estimatedSettlePrice", 0)),
            funding_rate=float(data["lastFundingRate"]),
            next_funding_time=data["nextFundingTime"],
            timestamp=data["time"],
        )
    
    async def get_funding_rate(self, symbol: str, limit: int = 1) -> list[FundingRate]:
        """Get funding rate history."""
        data = await self._request(
            "GET", "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": limit}
        )
        return [
            FundingRate(
                symbol=item["symbol"],
                funding_rate=float(item["fundingRate"]),
                funding_time=item["fundingTime"],
                mark_price=float(item.get("markPrice", 0)) if item.get("markPrice") else None,
            )
            for item in data
        ]
    
    async def get_open_interest(self, symbol: str) -> OpenInterest:
        """Get current open interest."""
        data = await self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol})
        # Get notional value from another endpoint
        oi_data = await self._request(
            "GET", "/futures/data/openInterestHist",
            {"symbol": symbol, "period": "5m", "limit": 1}
        )
        notional = float(oi_data[0]["sumOpenInterestValue"]) if oi_data else 0
        
        return OpenInterest(
            symbol=data["symbol"],
            open_interest=float(data["openInterest"]),
            open_interest_notional=notional,
            timestamp=data.get("time", timestamp_ms()),
        )
    
    async def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[OpenInterestHist]:
        """Get historical open interest.
        
        Args:
            symbol: Trading pair symbol
            period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
            limit: Number of records (max 500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("GET", "/futures/data/openInterestHist", params)

        hist = [
            OpenInterestHist(
                symbol=symbol,
                sum_open_interest=float(item["sumOpenInterest"]),
                sum_open_interest_value=float(item["sumOpenInterestValue"]),
                timestamp=int(item["timestamp"]),
            )
            for item in data
        ]

        # Binance sometimes returns newest-first; normalize to ascending by timestamp
        hist.sort(key=lambda x: x.timestamp)
        return hist

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 1000) -> OrderbookSnapshot:
        """Get orderbook depth snapshot.
        
        Args:
            symbol: Trading pair symbol
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)
        """
        data = await self._request(
            "GET", "/fapi/v1/depth",
            {"symbol": symbol, "limit": limit}
        )
        return OrderbookSnapshot(
            symbol=symbol,
            last_update_id=data["lastUpdateId"],
            timestamp=data.get("T", timestamp_ms()),
            bids=[OrderbookLevel(float(b[0]), float(b[1])) for b in data["bids"]],
            asks=[OrderbookLevel(float(a[0]), float(a[1])) for a in data["asks"]],
        )
    
    async def get_agg_trades(
        self,
        symbol: str,
        from_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Get aggregated trades.
        
        Args:
            symbol: Trading pair symbol
            from_id: Trade ID to fetch from
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of trades (max 1000)
        """
        symbol = symbol.upper()
        # Binance USDT-M Futures aggTrades constraints:
        # - `limit` max 1000
        # - if `startTime` and `endTime` are both sent, the window must be <= 1 hour
        # - sending `fromId` together with time params can trigger 400/"Internal error: 1" (Binance-side)
        limit = max(1, min(int(limit), 1000))
        params: dict[str, Any] = {"symbol": symbol, "limit": limit}

        if from_id is not None:
            params["fromId"] = int(from_id)
            # Intentionally ignore time params when paginating by id (we stop client-side by timestamp).
            if start_time is not None or end_time is not None:
                self.logger.debug(
                    "aggtrades_ignoring_time_params_with_fromid",
                    symbol=symbol,
                    from_id=from_id,
                    start_time=start_time,
                    end_time=end_time,
                )
        else:
            if start_time is not None:
                params["startTime"] = int(start_time)

            if end_time is not None:
                if start_time is not None and (int(end_time) - int(start_time) > 3_600_000):
                    # Docs require <= 1h when both are provided; we filter endTime client-side instead.
                    self.logger.debug(
                        "aggtrades_dropping_endtime_over_1h_window",
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                    )
                else:
                    params["endTime"] = int(end_time)
        
        data = await self._request("GET", "/fapi/v1/aggTrades", params)
        return [
            AggTrade(
                agg_trade_id=item["a"],
                symbol=symbol,
                price=float(item["p"]),
                quantity=float(item["q"]),
                first_trade_id=item["f"],
                last_trade_id=item["l"],
                timestamp=item["T"],
                is_buyer_maker=item["m"],
            )
            for item in data
        ]
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 500,
    ) -> list[Kline]:
        """Get kline/candlestick data.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of klines (max 1500)
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("GET", "/fapi/v1/klines", params)
        return [
            Kline(
                symbol=symbol,
                interval=interval,
                open_time=item[0],
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                close_time=item[6],
                quote_volume=float(item[7]),
                trade_count=item[8],
                taker_buy_volume=float(item[9]),
                taker_buy_quote_volume=float(item[10]),
            )
            for item in data
        ]
    async def fetch_trades_range(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
    ) -> list[AggTrade]:
        """Fetch *all* aggTrades in a time range without gaps.

        Binance aggTrades can contain many rows with the same timestamp.
        Paginating by `last_timestamp + 1` can drop trades when a page hits the server `limit`.
        We bootstrap using `startTime` then paginate strictly by `fromId`.
        """
        all_trades: list[AggTrade] = []

        first_page = await self.get_agg_trades(symbol=symbol, start_time=start_time, limit=1000)
        if not first_page:
            return []

        first_page = [t for t in first_page if start_time <= t.timestamp <= end_time]
        all_trades.extend(first_page)

        if first_page and first_page[-1].timestamp >= end_time:
            return all_trades

        last_id = first_page[-1].agg_trade_id if first_page else None
        if last_id is None:
            return all_trades

        seen_last_ids = {last_id}
        while True:
            page = await self.get_agg_trades(symbol=symbol, from_id=last_id + 1, limit=1000)
            if not page:
                break

            if page[0].timestamp > end_time:
                break

            for t in page:
                if t.timestamp < start_time:
                    continue
                if t.timestamp > end_time:
                    return all_trades
                all_trades.append(t)

            new_last_id = page[-1].agg_trade_id
            if new_last_id == last_id or new_last_id in seen_last_ids:
                self.logger.warning("pagination_stalled", symbol=symbol, last_id=last_id)
                break

            last_id = new_last_id
            seen_last_ids.add(last_id)

            await asyncio.sleep(0.05)

        return all_trades
