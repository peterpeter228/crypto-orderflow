"""Stacked Imbalance detector."""

from typing import Any
from dataclasses import dataclass

from src.config import get_settings
from src.utils import get_logger, timestamp_ms
from .footprint import FootprintBar, FootprintLevel


@dataclass
class Imbalance:
    """Single imbalance at a price level."""
    price: float
    buy_volume: float
    sell_volume: float
    ratio: float
    direction: str  # 'buy' or 'sell'
    
    @property
    def dominant_volume(self) -> float:
        return self.buy_volume if self.direction == "buy" else self.sell_volume


@dataclass
class StackedImbalance:
    """Group of consecutive imbalances forming a stacked imbalance."""
    start_price: float
    end_price: float
    direction: str  # 'buy' or 'sell'
    levels: list[Imbalance]
    
    @property
    def level_count(self) -> int:
        return len(self.levels)
    
    @property
    def total_volume(self) -> float:
        return sum(i.dominant_volume for i in self.levels)
    
    @property
    def avg_ratio(self) -> float:
        if not self.levels:
            return 0.0
        return sum(i.ratio for i in self.levels) / len(self.levels)


class ImbalanceDetector:
    """Detect stacked imbalances in footprint data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("indicators.imbalance")
        
        # Configuration
        self.ratio_threshold = self.settings.imbalance_ratio_threshold
        self.min_consecutive = self.settings.imbalance_consecutive_levels
    
    def detect_imbalance(
        self,
        buy_volume: float,
        sell_volume: float,
    ) -> Imbalance | None:
        """Check if a price level has an imbalance.
        
        Args:
            buy_volume: Buy volume at level
            sell_volume: Sell volume at level
        
        Returns:
            Imbalance object if detected, None otherwise
        """
        # Avoid division by zero
        if buy_volume == 0 and sell_volume == 0:
            return None
        
        # Check buy imbalance
        if sell_volume > 0 and buy_volume / sell_volume >= self.ratio_threshold:
            return Imbalance(
                price=0,  # Set later
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                ratio=buy_volume / sell_volume,
                direction="buy",
            )
        
        # Check sell imbalance
        if buy_volume > 0 and sell_volume / buy_volume >= self.ratio_threshold:
            return Imbalance(
                price=0,  # Set later
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                ratio=sell_volume / buy_volume,
                direction="sell",
            )
        
        # Edge case: one side is zero
        if buy_volume > 0 and sell_volume == 0:
            return Imbalance(
                price=0,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                ratio=self.ratio_threshold * 100.0,
                direction="buy",
            )
        
        if sell_volume > 0 and buy_volume == 0:
            return Imbalance(
                price=0,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                ratio=self.ratio_threshold * 100.0,
                direction="sell",
            )
        
        return None
    
    def find_stacked_imbalances(
        self,
        bar: FootprintBar,
        ratio_threshold: float | None = None,
        min_consecutive: int | None = None,
    ) -> list[StackedImbalance]:
        """Find stacked imbalances in a footprint bar.
        
        Args:
            bar: Footprint bar to analyze
            ratio_threshold: Override default ratio threshold
            min_consecutive: Override minimum consecutive levels
        
        Returns:
            List of stacked imbalances found
        """
        if ratio_threshold is None:
            ratio_threshold = self.ratio_threshold
        if min_consecutive is None:
            min_consecutive = self.min_consecutive
        
        if not bar.levels:
            return []
        
        # Sort levels by price descending
        sorted_prices = sorted(bar.levels.keys(), reverse=True)
        
        stacked_imbalances: list[StackedImbalance] = []
        current_stack: list[Imbalance] = []
        current_direction: str | None = None
        
        for price in sorted_prices:
            level = bar.levels[price]
            imbalance = self.detect_imbalance(level.buy_volume, level.sell_volume)
            
            if imbalance:
                imbalance.price = price
                
                if current_direction is None or imbalance.direction == current_direction:
                    current_stack.append(imbalance)
                    current_direction = imbalance.direction
                else:
                    # Direction changed, check if we have a valid stack
                    if len(current_stack) >= min_consecutive:
                        stacked_imbalances.append(StackedImbalance(
                            start_price=current_stack[0].price,
                            end_price=current_stack[-1].price,
                            direction=current_direction,
                            levels=current_stack.copy(),
                        ))
                    
                    # Start new stack
                    current_stack = [imbalance]
                    current_direction = imbalance.direction
            else:
                # No imbalance, check and reset stack
                if len(current_stack) >= min_consecutive:
                    stacked_imbalances.append(StackedImbalance(
                        start_price=current_stack[0].price,
                        end_price=current_stack[-1].price,
                        direction=current_direction or "unknown",
                        levels=current_stack.copy(),
                    ))
                
                current_stack = []
                current_direction = None
        
        # Check final stack
        if len(current_stack) >= min_consecutive:
            stacked_imbalances.append(StackedImbalance(
                start_price=current_stack[0].price,
                end_price=current_stack[-1].price,
                direction=current_direction or "unknown",
                levels=current_stack.copy(),
            ))
        
        return stacked_imbalances
    
    def analyze_footprint(
        self,
        bar: FootprintBar,
        ratio_threshold: float | None = None,
        min_consecutive: int | None = None,
    ) -> dict[str, Any]:
        """Analyze footprint bar for imbalances.
        
        Args:
            bar: Footprint bar to analyze
            ratio_threshold: Override default ratio threshold
            min_consecutive: Override minimum consecutive levels
        
        Returns:
            Analysis results
        """
        stacked = self.find_stacked_imbalances(bar, ratio_threshold, min_consecutive)
        
        buy_stacks = [s for s in stacked if s.direction == "buy"]
        sell_stacks = [s for s in stacked if s.direction == "sell"]
        
        return {
            "symbol": bar.symbol,
            "timeframe": bar.timeframe,
            "timestamp": bar.timestamp,
            "analysisTime": timestamp_ms(),
            "config": {
                "ratioThreshold": ratio_threshold or self.ratio_threshold,
                "minConsecutive": min_consecutive or self.min_consecutive,
            },
            "summary": {
                "totalStackedImbalances": len(stacked),
                "buyStacks": len(buy_stacks),
                "sellStacks": len(sell_stacks),
                "totalBuyLevels": sum(s.level_count for s in buy_stacks),
                "totalSellLevels": sum(s.level_count for s in sell_stacks),
            },
            "buyImbalances": [
                {
                    "startPrice": s.start_price,
                    "endPrice": s.end_price,
                    "levelCount": s.level_count,
                    "totalVolume": s.total_volume,
                    "avgRatio": s.avg_ratio,
                }
                for s in buy_stacks
            ],
            "sellImbalances": [
                {
                    "startPrice": s.start_price,
                    "endPrice": s.end_price,
                    "levelCount": s.level_count,
                    "totalVolume": s.total_volume,
                    "avgRatio": s.avg_ratio,
                }
                for s in sell_stacks
            ],
        }


def detect_diagonal_imbalance(
    bars: list[FootprintBar],
    direction: str = "up",
    ratio_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Detect diagonal imbalances across multiple bars.
    
    A diagonal imbalance is when imbalances at progressively higher/lower
    prices form across consecutive bars.
    
    Args:
        bars: List of footprint bars (should be consecutive)
        direction: 'up' for ascending prices, 'down' for descending
        ratio_threshold: Imbalance ratio threshold
    
    Returns:
        List of diagonal imbalances found
    """
    if len(bars) < 2:
        return []
    
    detector = ImbalanceDetector()
    diagonal_imbalances: list[dict[str, Any]] = []
    
    # Track potential diagonal chains
    chains: list[list[tuple[int, float, Imbalance]]] = []  # [(bar_index, price, imbalance)]
    
    for bar_idx, bar in enumerate(bars):
        stacked = detector.find_stacked_imbalances(bar)
        
        # Get all imbalance levels
        imbalance_prices = []
        for stack in stacked:
            for imb in stack.levels:
                imbalance_prices.append((imb.price, imb))
        
        # Try to extend existing chains
        new_chains = []
        for chain in chains:
            last_bar_idx, last_price, _ = chain[-1]
            
            # Look for continuation
            for price, imb in imbalance_prices:
                if direction == "up" and price > last_price and bar_idx == last_bar_idx + 1:
                    extended = chain + [(bar_idx, price, imb)]
                    new_chains.append(extended)
                elif direction == "down" and price < last_price and bar_idx == last_bar_idx + 1:
                    extended = chain + [(bar_idx, price, imb)]
                    new_chains.append(extended)
        
        # Start new chains from current bar
        for price, imb in imbalance_prices:
            new_chains.append([(bar_idx, price, imb)])
        
        chains = new_chains
    
    # Filter chains with at least 3 bars
    for chain in chains:
        if len(chain) >= 3:
            diagonal_imbalances.append({
                "direction": direction,
                "barCount": len(chain),
                "startBar": chain[0][0],
                "endBar": chain[-1][0],
                "startPrice": chain[0][1],
                "endPrice": chain[-1][1],
                "levels": [
                    {"barIndex": c[0], "price": c[1], "direction": c[2].direction}
                    for c in chain
                ],
            })
    
    return diagonal_imbalances
