import math

import pytest

from src.mcp.tools import build_quote_profile


def test_build_quote_profile_computes_value_area():
    rows = [
        {"price_level": 100.0, "buy_volume": 1.0},
        {"price_level": 100.4, "sell_volume": 0.5},
        {"price_level": 101.0, "buy_volume": 2.0, "sell_volume": 1.0},
        {"price_level": 102.0, "buy_volume": 0.5},
    ]

    result = build_quote_profile(rows, bin_size=1.0, value_area_percent=70.0)

    assert result["success"] is True
    assert result["quality_flags"] == []
    assert math.isclose(result["volumeQuote"], 504.2, rel_tol=1e-6)
    assert math.isclose(result["deltaQuote"], 201.8, rel_tol=1e-6)
    assert result["vPOC"] == 101.0
    assert result["vVAH"] == 101.0
    assert result["vVAL"] == 100.0
