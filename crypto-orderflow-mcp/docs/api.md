# API Reference

## MCP Tools

### get_market_snapshot

获取指定交易对的市场快照。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 (e.g., "BTCUSDT") |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timestamp": 1705286400000,
  "lastPrice": 42500.5,
  "markPrice": 42501.2,
  "indexPrice": 42500.8,
  "high24h": 43000.0,
  "low24h": 42000.0,
  "volume24h": 125000.5,
  "quoteVolume24h": 5312500000,
  "fundingRate": 0.0001,
  "nextFundingTime": 1705320000000,
  "openInterest": 85000.5,
  "openInterestNotional": 3612500000,
  "cvd": 1250.5,
  "lastTradeTime": 1705286399500,
  "priceUnit": "USDT",
  "volumeUnit": "BTC"
}
```

---

### get_key_levels

获取关键价位，包括 VWAP、Volume Profile、Session H/L。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| date | string | 否 | 日期 (YYYY-MM-DD)，默认今天 |
| sessionTZ | string | 否 | 时区，默认 "UTC" |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timestamp": 1705286400000,
  "date": "2024-01-15",
  "sessionTimezone": "UTC",
  "vwap": {
    "dVWAP": 42350.5,
    "pdVWAP": 42100.2
  },
  "volumeProfile": {
    "developing": {
      "POC": 42400.0,
      "VAH": 42600.0,
      "VAL": 42200.0,
      "totalVolume": 50000.5,
      "priceLevels": 150
    },
    "previousDay": {
      "POC": 42000.0,
      "VAH": 42300.0,
      "VAL": 41800.0,
      "totalVolume": 120000.2,
      "priceLevels": 200
    }
  },
  "sessions": {
    "sessions": {
      "timezone": "UTC",
      "tokyo": {"hours": "00:00-09:00"},
      "london": {"hours": "07:00-16:00"},
      "ny": {"hours": "13:00-22:00"}
    },
    "today": {
      "tokyoH": 42500.0,
      "tokyoL": 42100.0,
      "tokyoVolume": 15000.0,
      "tokyoActive": false,
      "londonH": 42600.0,
      "londonL": 42200.0,
      "londonVolume": 25000.0,
      "londonActive": true
    },
    "yesterday": {
      "tokyoH": 42200.0,
      "tokyoL": 41800.0,
      "tokyoVolume": 12000.0
    }
  },
  "priceUnit": "USDT"
}
```

---

### get_session_profile

按 Session（Tokyo/London/NY）输出 **Session Profile**：

- **session totals**（Quote Volume / Delta）：来自 Binance kline（用来对齐你截图中的 Session Vol/Delta）
- **session volume profile**（vPOC / vVAH / vVAL）：来自本地缓存的 `footprint_1m` 聚合（需要开启 backfill，才能对历史更准确）

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| date | string | 否 | 日期 (YYYY-MM-DD)，默认今天（UTC） |
| session | string | 否 | tokyo/london/ny/all（默认 all） |
| interval | string | 否 | 用于统计 session totals 的 kline 周期（默认 15m） |
| valueAreaPercent | number | 否 | 价值区域百分比（默认 70） |
| includeProfileLevels | boolean | 否 | 是否返回全部价位明细（默认 false） |

---

### get_swing_liquidity

输出 **Swing Liquidity**（Pivot Swing High/Low）关键流动性价位：

- Pivot swing highs => buy-side liquidity（上方流动性）
- Pivot swing lows  => sell-side liquidity（下方流动性）

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| interval | string | 否 | K 线周期（默认 15m） |
| lookbackBars | integer | 否 | 回看 bar 数（默认 300） |
| pivotLeft | integer | 否 | pivot 左侧确认 bar（默认 10） |
| pivotRight | integer | 否 | pivot 右侧确认 bar（默认 15） |
| activeOnly | boolean | 否 | 只返回当前价格附近“未被扫掉”的流动性（默认 false） |
| maxLevels | integer | 否 | 最大返回数量（默认 150） |

---

### get_footprint

获取 Footprint 图数据。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| timeframe | string | 是 | 时间周期 (1m/5m/15m/30m/1h) |
| startTime | integer | 是 | 开始时间戳 (ms) |
| endTime | integer | 是 | 结束时间戳 (ms) |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timeframe": "5m",
  "startTime": 1705276800000,
  "endTime": 1705280400000,
  "timestamp": 1705286400000,
  "barCount": 12,
  "bars": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "5m",
      "timestamp": 1705276800000,
      "open": 42350.0,
      "high": 42450.0,
      "low": 42300.0,
      "close": 42400.0,
      "buyVolume": 150.5,
      "sellVolume": 120.3,
      "totalVolume": 270.8,
      "delta": 30.2,
      "maxDeltaPrice": 42400.0,
      "minDeltaPrice": 42320.0,
      "pocPrice": 42380.0,
      "levels": [
        {
          "price": 42450.0,
          "buyVolume": 10.5,
          "sellVolume": 8.2,
          "delta": 2.3,
          "totalVolume": 18.7,
          "tradeCount": 45
        }
      ],
      "levelCount": 15
    }
  ],
  "volumeUnit": "BTC",
  "priceUnit": "USDT"
}
```

---

### get_orderflow_metrics

获取 Orderflow 指标。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| timeframe | string | 是 | 时间周期 |
| startTime | integer | 是 | 开始时间戳 (ms) |
| endTime | integer | 是 | 结束时间戳 (ms) |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timeframe": "15m",
  "startTime": 1705276800000,
  "endTime": 1705280400000,
  "timestamp": 1705286400000,
  "delta": {
    "totalBuyVolume": 5000.5,
    "totalSellVolume": 4800.2,
    "totalDelta": 200.3,
    "totalVolume": 9800.7,
    "deltaPercent": 2.04,
    "positiveDeltaBars": 8,
    "negativeDeltaBars": 4,
    "barCount": 12
  },
  "deltaSequence": [
    {"timestamp": 1705276800000, "delta": 25.5, "deltaPercent": 3.2}
  ],
  "cvdSequence": [
    {"timestamp": 1705276800000, "cvd": 25.5}
  ],
  "currentCVD": 1250.5,
  "imbalances": {
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "timestamp": 1705280100000,
    "config": {
      "ratioThreshold": 3.0,
      "minConsecutive": 3
    },
    "summary": {
      "totalStackedImbalances": 2,
      "buyStacks": 1,
      "sellStacks": 1
    },
    "buyImbalances": [
      {
        "startPrice": 42400.0,
        "endPrice": 42380.0,
        "levelCount": 3,
        "totalVolume": 45.5,
        "avgRatio": 4.2
      }
    ],
    "sellImbalances": []
  },
  "volumeUnit": "BTC"
}
```

---

### get_orderbook_depth_delta

获取订单簿深度变化。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| percent | number | 否 | 价格范围% (默认 1.0) |
| windowSec | integer | 否 | 快照间隔秒 (默认 5) |
| lookback | integer | 否 | 回溯秒数 (默认 3600) |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "percentRange": 1.0,
  "windowSec": 5,
  "lookbackSec": 3600,
  "timestamp": 1705286400000,
  "snapshots": [
    {
      "timestamp": 1705282800000,
      "bidVolume": 250.5,
      "askVolume": 220.3,
      "netVolume": 30.2,
      "midPrice": 42400.5,
      "bidDelta": 5.2,
      "askDelta": -3.1,
      "netDelta": 8.3,
      "priceDelta": 10.5
    }
  ],
  "analysis": {
    "avgNetVolume": 25.5,
    "maxNetVolume": 50.2,
    "minNetVolume": -15.3,
    "currentNetVolume": 30.2,
    "bidTrendStrength": 0.7,
    "askTrendStrength": 0.3,
    "dominantSide": "bids",
    "snapshotCount": 720
  },
  "current": {
    "symbol": "BTCUSDT",
    "timestamp": 1705286395000,
    "midPrice": 42400.5,
    "bidVolume": 250.5,
    "askVolume": 220.3,
    "netVolume": 30.2,
    "bidAskRatio": 1.137,
    "percentRange": 1.0,
    "dominantSide": "bids"
  },
  "volumeUnit": "BTC"
}
```

---

### stream_liquidations

获取近期清算事件。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| limit | integer | 否 | 返回数量 (默认 100) |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timestamp": 1705286400000,
  "count": 25,
  "statistics": {
    "longLiquidations": 15,
    "shortLiquidations": 10,
    "totalLongNotional": 2500000,
    "totalShortNotional": 1800000
  },
  "liquidations": [
    {
      "timestamp": 1705286350000,
      "symbol": "BTCUSDT",
      "side": "SELL",
      "price": 42350.0,
      "avgPrice": 42345.5,
      "originalQty": 2.5,
      "filledQty": 2.5,
      "notional": 105863.75,
      "isLongLiquidation": true,
      "orderStatus": "FILLED"
    }
  ],
  "notionalUnit": "USDT",
  "volumeUnit": "BTC"
}
```

---

### get_open_interest

获取持仓量数据。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |
| period | string | 否 | 历史周期 (默认 "5m") |
| limit | integer | 否 | 历史记录数 (默认 100) |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timestamp": 1705286400000,
  "current": {
    "openInterest": 85000.5,
    "openInterestNotional": 3612500000
  },
  "delta": {
    "period": "5m",
    "openInterestDelta": 250.5,
    "openInterestDeltaNotional": 10625000
  },
  "history": [
    {
      "timestamp": 1705286100000,
      "openInterest": 84750.0,
      "openInterestNotional": 3601875000
    }
  ],
  "oiUnit": "BTC",
  "notionalUnit": "USDT"
}
```

---

### get_funding_rate

获取资金费率。

**参数：**
| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 交易对 |

**返回示例：**
```json
{
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "marketType": "linear_perpetual",
  "timestamp": 1705286400000,
  "current": {
    "fundingRate": 0.0001,
    "fundingRatePercent": 0.01,
    "nextFundingTime": 1705320000000
  },
  "history": [
    {
      "fundingTime": 1705276800000,
      "fundingRate": 0.00008
    }
  ]
}
```
