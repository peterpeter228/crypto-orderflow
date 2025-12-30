"""TPO (Time Price Opportunity) profile calculator.

This module builds a TPO profile from the local `footprint_1m` table:

- A *TPO* at a price means: that price traded at least once during a given time period.
- By default, Exocharts uses 30-minute periods ("letters") for TPO charts. 
- VAH/VAL/POC are then derived from the chosen distribution (time by default, or volume if configured). 

We intentionally compute from footprint (aggTrades -> minute footprint) for precision instead of
using candle high/low, because high/low can include prices with tiny prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any

from src.data.storage import DataStorage
from src.utils import get_logger
from .profile_engine import ValueAreaCalculator


@dataclass
class TPOLevel:
    price: float
    tpo_count: int
    notional: float
    delta_notional: float
    trade_count: int


class TPOProfileCalculator:
    """Compute TPO profile statistics for an arbitrary time window."""

    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.logger = get_logger("indicators.tpo_profile")

    @staticmethod
    def calculate_poc(profile: dict[float, float], mid_price: float | None = None) -> float | None:
        """Return the POC price (max value). Deterministic tie-break."""
        if not profile:
            return None

        max_v = max(profile.values())
        prices = [p for p, v in profile.items() if v == max_v]
        if len(prices) == 1:
            return prices[0]

        prices.sort()
        if mid_price is None:
            # choose the middle of tied prices (deterministic)
            return prices[len(prices) // 2]

        # choose the one closest to mid_price
        return min(prices, key=lambda p: (abs(p - mid_price), p))

    @classmethod
    def calculate_value_area(
        cls,
        profile: dict[float, float],
        percentage: float = 70.0,
        poc: float | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Calculate (POC, VAH, VAL) using shared ValueAreaCalculator."""
        # ValueAreaCalculator computes poc internally; retain legacy poc override for compatibility.
        if poc is not None and poc not in profile:
            poc = None
        if poc is None:
            return ValueAreaCalculator.compute(profile, percentage)
        if not profile:
            return None, None, None
        temp = dict(profile)
        return ValueAreaCalculator.compute(temp, percentage)

    async def build_period_profiles(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        period_ms: int,
        tick_size: float | None,
        value_area_percent: float,
        tail_min_len: int = 2,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build per-period (e.g., 30m) volume profiles and single prints/tails.

        This is the data you need for:
        - 30m vPOC line per bar
        - Single prints (price levels that only appear in one period)
        - Tails (single prints at the extreme of the profile)

        The computation is based on `footprint_1m` aggregates to stay consistent with
        Exocharts-style profile logic.
        """

        rows = await self.storage.get_tpo_period_matrix(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            period_ms=period_ms,
            tick_size=tick_size,
        )

        if not rows:
            return [], {
                "levels": [],
                "lowTailLevels": [],
                "highTailLevels": [],
                "bodyLevels": [],
                "tailMinLength": tail_min_len,
            }

        period_profile: dict[int, dict[float, float]] = defaultdict(lambda: defaultdict(float))
        period_trade_count: dict[int, int] = defaultdict(int)
        period_total_notional: dict[int, float] = defaultdict(float)
        period_total_delta_notional: dict[int, float] = defaultdict(float)

        for r in rows:
            idx = int(r["period_idx"])
            price = float(r["price_level"])
            notional = float(r.get("notional") or 0.0)
            delta_notional = float(r.get("delta_notional") or 0.0)
            tc = int(r.get("trade_count") or 0)

            period_profile[idx][price] += notional
            period_trade_count[idx] += tc
            period_total_notional[idx] += notional
            period_total_delta_notional[idx] += delta_notional

        # Count in how many periods each price level appears
        price_period_count: dict[float, int] = defaultdict(int)
        for idx, profile in period_profile.items():
            for price in profile.keys():
                price_period_count[price] += 1

        periods: list[dict[str, Any]] = []
        for idx in sorted(period_profile.keys()):
            profile = period_profile[idx]
            if not profile:
                continue

            prices = sorted(profile.keys())
            low = prices[0]
            high = prices[-1]
            mid = (low + high) / 2.0

            v_poc = self.calculate_poc(profile, mid)
            # Volume-based value area inside the period
            _, v_vah, v_val = self.calculate_value_area(
                profile,
                percentage=value_area_percent,
                poc=v_poc,
            )

            periods.append(
                {
                    "index": idx,
                    "startTime": start_time + idx * period_ms,
                    "endTime": min(end_time, start_time + (idx + 1) * period_ms),
                    "low": low,
                    "high": high,
                    "vPOC": v_poc,
                    "vVAH": v_vah,
                    "vVAL": v_val,
                    "volumeNotional": period_total_notional[idx],
                    "deltaNotional": period_total_delta_notional[idx],
                    "tradeCount": period_trade_count[idx],
                    "priceLevelCount": len(prices),
                }
            )

        all_prices = sorted(price_period_count.keys())
        single_levels = [p for p in all_prices if price_period_count[p] == 1]

        low_tail: list[float] = []
        for p in all_prices:
            if price_period_count[p] == 1:
                low_tail.append(p)
            else:
                break

        high_tail: list[float] = []
        for p in reversed(all_prices):
            if price_period_count[p] == 1:
                high_tail.append(p)
            else:
                break
        high_tail = list(reversed(high_tail))

        # Apply tail min length (Exocharts typically expects a tail to be more than 1 print)
        if len(low_tail) < tail_min_len:
            low_tail = []
        if len(high_tail) < tail_min_len:
            high_tail = []

        tail_set = set(low_tail) | set(high_tail)
        body_levels = [p for p in single_levels if p not in tail_set]

        single_prints = {
            "levels": single_levels,
            "lowTailLevels": low_tail,
            "highTailLevels": high_tail,
            "bodyLevels": body_levels,
            "tailMinLength": tail_min_len,
        }

        return periods, single_prints

    async def build_profile(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        period_minutes: int = 30,
        tick_size: float | None = None,
        value_area_percent: float = 70.0,
        use_volume_for_va: bool = False,
        include_levels: bool = False,
        max_levels: int = 240,
        include_period_profiles: bool = False,
        include_single_prints: bool = False,
        tail_min_len: int = 2,
    ) -> dict[str, Any]:
        """Build a TPO profile for a given time window.

        Args:
            symbol: e.g. BTCUSDT
            start_time/end_time: ms UTC timestamps
            period_minutes: TPO period size (Exocharts letters are 30m by default) 
            tick_size: optional price bucket size; if None we use the footprint's native price levels
            value_area_percent: VA% (Exocharts commonly uses 68 or 70) 
            use_volume_for_va: if True, compute VAH/VAL/POC from volume instead of time 
            include_levels: return per-price distribution (can be large)
            max_levels: cap number of levels returned when include_levels=True

            include_period_profiles: return per-period (e.g. 30m) vPOC/VAH/VAL
            include_single_prints: return single prints and tails (derived from per-period matrix)
            tail_min_len: minimum contiguous single prints required to qualify as a tail

        Returns:
            dict containing both time-based and volume-based markers.
        """
        symbol = symbol.upper().strip()
        period_ms = int(period_minutes) * 60_000

        rows = await self.storage.get_tpo_aggregates(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            period_ms=period_ms,
            tick_size=tick_size,
        )

        # Build distributions
        tpo_profile: dict[float, float] = {}
        vol_profile: dict[float, float] = {}
        delta_profile: dict[float, float] = {}
        trade_profile: dict[float, int] = {}

        for r in rows:
            try:
                price = float(r.get("price_level"))
            except Exception:
                continue

            tpo = int(r.get("tpo_count") or 0)
            notional = float(r.get("notional") or 0.0)
            delta_notional = float(r.get("delta_notional") or 0.0)
            trade_count = int(r.get("trade_count") or 0)

            if tpo <= 0 and notional == 0.0:
                continue

            tpo_profile[price] = float(tpo)
            vol_profile[price] = float(notional)
            delta_profile[price] = float(delta_notional)
            trade_profile[price] = trade_count

        sorted_prices = sorted(tpo_profile.keys())
        mid_price = (sorted_prices[0] + sorted_prices[-1]) / 2.0 if sorted_prices else None

        # Time-based markers (TPO POC / VAH / VAL)
        tpo_poc, tpo_vah, tpo_val = self.calculate_value_area(
            tpo_profile,
            percentage=value_area_percent,
            poc=self.calculate_poc(tpo_profile, mid_price),
        )

        # Volume-based markers (vPOC / vVAH / vVAL)
        vpoc, vvah, vval = self.calculate_value_area(
            vol_profile,
            percentage=value_area_percent,
            poc=self.calculate_poc(vol_profile, mid_price),
        )

        active_poc = vpoc if use_volume_for_va else tpo_poc
        active_vah = vvah if use_volume_for_va else tpo_vah
        active_val = vval if use_volume_for_va else tpo_val

        total_tpo = int(sum(tpo_profile.values())) if tpo_profile else 0
        total_notional = float(sum(vol_profile.values())) if vol_profile else 0.0
        total_delta_notional = float(sum(delta_profile.values())) if delta_profile else 0.0


        def _above_below(profile: dict[float, float], poc_price: float | None) -> dict[str, Any]:
            if not profile or poc_price is None:
                return {"above": 0.0, "below": 0.0, "abovePct": None, "belowPct": None}

            total = float(sum(profile.values()))
            above = float(sum(v for p, v in profile.items() if p > poc_price))
            below = float(sum(v for p, v in profile.items() if p < poc_price))
            above_pct = (above / total * 100.0) if total > 0 else None
            below_pct = (below / total * 100.0) if total > 0 else None
            return {"above": above, "below": below, "abovePct": above_pct, "belowPct": below_pct}

        def _delta_above_below(delta_map: dict[float, float], poc_price: float | None) -> dict[str, Any]:
            if not delta_map or poc_price is None:
                return {"above": 0.0, "below": 0.0}
            above = float(sum(v for p, v in delta_map.items() if p > poc_price))
            below = float(sum(v for p, v in delta_map.items() if p < poc_price))
            return {"above": above, "below": below}

        distribution_stats: dict[str, Any] = {
            "tpoAboveBelow_tpoPOC": _above_below(tpo_profile, tpo_poc),
            "tpoAboveBelow_vPOC": _above_below(tpo_profile, vpoc),
            "volAboveBelow_tpoPOC": _above_below(vol_profile, tpo_poc),
            "volAboveBelow_vPOC": _above_below(vol_profile, vpoc),
            "deltaNotionalAboveBelow_tpoPOC": _delta_above_below(delta_profile, tpo_poc),
            "deltaNotionalAboveBelow_vPOC": _delta_above_below(delta_profile, vpoc),
        }
        out: dict[str, Any] = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "periodMinutes": int(period_minutes),
            "tickSize": tick_size,
            "valueAreaPercent": float(value_area_percent),
            "useVolumeForVA": bool(use_volume_for_va),
            "priceLevelCount": len(vol_profile) if use_volume_for_va else len(tpo_profile),
            "totals": {
                "tpoTotalCount": total_tpo,
                "notional": total_notional,
                "deltaNotional": total_delta_notional,
            },
            "timeBased": {"poc": tpo_poc, "vah": tpo_vah, "val": tpo_val},
            "volumeBased": {"vpoc": vpoc, "vvah": vvah, "vval": vval},
            "active": {"poc": active_poc, "vah": active_vah, "val": active_val},
            "distributionStats": distribution_stats,
        }

        if include_levels and tpo_profile:
            # Return a capped, ordered list (most useful levels first: by tpo_count then notional)
            levels: list[TPOLevel] = []
            for p in sorted(tpo_profile.keys()):
                levels.append(
                    TPOLevel(
                        price=p,
                        tpo_count=int(tpo_profile.get(p, 0.0)),
                        notional=float(vol_profile.get(p, 0.0)),
                        delta_notional=float(delta_profile.get(p, 0.0)),
                        trade_count=int(trade_profile.get(p, 0)),
                    )
                )

            # sort by TPO count desc, then notional desc
            levels.sort(key=lambda x: (x.tpo_count, x.notional), reverse=True)

            capped = levels[: max(1, int(max_levels))]
            out["levels"] = [
                {
                    "price": lv.price,
                    "tpoCount": lv.tpo_count,
                    "notional": lv.notional,
                    "deltaNotional": lv.delta_notional,
                    "tradeCount": lv.trade_count,
                }
                for lv in capped
            ]


        # Optional: per-period profiles (e.g. 30m vPOC) + single prints/tails
        if include_period_profiles or include_single_prints:
            try:
                period_profiles, single_prints = await self.build_period_profiles(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    period_ms=period_ms,
                    tick_size=tick_size,
                    value_area_percent=value_area_percent,
                    tail_min_len=tail_min_len,
                )
                if include_period_profiles:
                    out["periodProfiles"] = period_profiles
                if include_single_prints:
                    out["singlePrints"] = single_prints
            except Exception as e:
                self.logger.warning(
                    "tpo_period_features_failed",
                    symbol=symbol,
                    error=str(e),
                )

        return out
