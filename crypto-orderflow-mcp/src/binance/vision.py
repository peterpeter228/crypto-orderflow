"""Binance Vision downloader for daily aggTrades.

Binance provides public historical data at https://data.binance.vision.
Downloading a single daily zip is significantly more efficient than
paginating `/fapi/v1/aggTrades` and helps avoid REST 429 rate limiting.

This module is intentionally dependency-light and streams rows from the zip
without loading the full file into memory.
"""

from __future__ import annotations

import asyncio
import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import AsyncIterator, Optional

import aiohttp

from .types import AggTrade
from ..config import Settings


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    return v in {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class VisionFileInfo:
    symbol: str
    day: date
    url: str
    cache_path: Path


class BinanceVisionAggTrades:
    """Download and iterate daily aggTrades from Binance Vision."""

    def __init__(self, settings: Settings, logger) -> None:
        self.settings = settings
        self.logger = logger
        self.cache_dir = settings.ensure_vision_cache_dir()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=600)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def daily_aggtrades_info(self, symbol: str, day: date) -> VisionFileInfo:
        sym = symbol.upper()
        day_str = day.strftime("%Y-%m-%d")
        url = (
            f"{self.settings.binance_vision_base_url.rstrip('/')}/data/futures/um/"
            f"daily/aggTrades/{sym}/{sym}-aggTrades-{day_str}.zip"
        )
        cache_path = self.cache_dir / f"{sym}-aggTrades-{day_str}.zip"
        return VisionFileInfo(symbol=sym, day=day, url=url, cache_path=cache_path)

    async def ensure_daily_aggtrades_zip(self, symbol: str, day: date) -> Optional[Path]:
        """Ensure the daily zip is present locally.

        Returns the local path, or None if the file is not available (404) or
        could not be downloaded.
        """

        info = self.daily_aggtrades_info(symbol, day)
        path = info.cache_path
        if path.exists() and path.stat().st_size > 0:
            return path

        # Download with a temp file and atomic replace.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            sess = await self._get_session()
            async with sess.get(info.url) as resp:
                if resp.status == 404:
                    self.logger.info(
                        "vision_daily_not_found",
                        symbol=info.symbol,
                        day=str(info.day),
                        url=info.url,
                    )
                    return None
                resp.raise_for_status()

                path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        f.write(chunk)

                tmp_path.replace(path)
                self.logger.info(
                    "vision_daily_downloaded",
                    symbol=info.symbol,
                    day=str(info.day),
                    bytes=path.stat().st_size,
                )
                return path
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning(
                "vision_daily_download_failed",
                symbol=info.symbol,
                day=str(info.day),
                error=str(e),
            )
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    async def iter_daily_aggtrades(
        self,
        symbol: str,
        day: date,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> AsyncIterator[AggTrade]:
        """Yield AggTrade rows from the daily zip.

        The iterator yields trades in the file order, which is typically sorted
        by time / aggTradeId.
        """

        zip_path = await self.ensure_daily_aggtrades_zip(symbol, day)
        if zip_path is None:
            # IMPORTANT:
            # Binance Vision sometimes lags for the most recent day(s). When the
            # daily zip is missing (404) or couldn't be downloaded, we MUST let
            # callers (the backfiller) fall back to REST.
            raise FileNotFoundError(
                f"Binance Vision daily aggTrades not available for {symbol} {day.isoformat()}"
            )

        # The zip typically contains one CSV.
        with zipfile.ZipFile(zip_path) as zf:
            name: Optional[str] = None
            for n in zf.namelist():
                if n.lower().endswith(".csv"):
                    name = n
                    break
            if name is None:
                # Fallback to the first file.
                names = zf.namelist()
                if not names:
                    return
                name = names[0]

            with zf.open(name, "r") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.reader(text)

                first_row = next(reader, None)
                if first_row is None:
                    return

                # Detect header.
                col_map: dict[str, int] = {}
                if first_row and not str(first_row[0]).strip().isdigit():
                    header = [c.strip().lower() for c in first_row]
                    for idx, col in enumerate(header):
                        col_map[col] = idx
                else:
                    # No header, treat the first row as data.
                    # Common futures aggTrades daily files are:
                    # aggTradeId,price,quantity,firstTradeId,lastTradeId,timestamp,isBuyerMaker
                    col_map = {
                        "aggtradeid": 0,
                        "price": 1,
                        "quantity": 2,
                        "timestamp": 5,
                        "isbuyermaker": 6,
                    }
                    # Process the first row as data.
                    row = first_row
                    tr = self._row_to_aggtrade(row, col_map)
                    if tr is not None:
                        if start_ms is None or tr.timestamp >= start_ms:
                            if end_ms is None or tr.timestamp < end_ms:
                                yield tr
                    # Continue with the remaining rows.

                for row in reader:
                    tr = self._row_to_aggtrade(row, col_map)
                    if tr is None:
                        continue
                    if start_ms is not None and tr.timestamp < start_ms:
                        continue
                    if end_ms is not None and tr.timestamp >= end_ms:
                        break
                    yield tr

    def _row_to_aggtrade(self, row: list[str], col_map: dict[str, int]) -> Optional[AggTrade]:
        """Convert a CSV row to AggTrade.

        Returns None for malformed rows.
        """

        try:
            # Try multiple possible column names.
            def idx(*names: str, default: int) -> int:
                for n in names:
                    if n in col_map:
                        return col_map[n]
                return default

            i_agg = idx("aggtradeid", "agg_trade_id", default=0)
            i_price = idx("price", default=1)
            i_qty = idx("quantity", "qty", default=2)
            i_ts = idx("timestamp", "time", default=5)
            i_m = idx("isbuyermaker", "is_buyer_maker", "isbuymaker", default=6)

            agg_id = int(row[i_agg])
            price = float(row[i_price])
            qty = float(row[i_qty])
            ts = int(row[i_ts])
            is_buyer_maker = _parse_bool(row[i_m]) if i_m < len(row) else False

            return AggTrade(
                agg_trade_id=agg_id,
                price=price,
                quantity=qty,
                timestamp=ts,
                is_buyer_maker=is_buyer_maker,
            )
        except Exception:
            return None
