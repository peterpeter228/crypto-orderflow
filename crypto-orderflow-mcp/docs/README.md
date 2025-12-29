# Documentation

## Contents

- [API Reference](./api.md) - MCP Tools 详细文档
- [Cherry Studio Setup](./cherry_studio_setup.md) - Cherry Studio 配置指南
- [Troubleshooting](./troubleshooting.md) - 常见问题排查

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Cherry Studio                           │
│                    (MCP Client)                             │
└─────────────────────────┬───────────────────────────────────┘
                          │ SSE / HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Crypto Orderflow MCP Server                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  FastAPI    │  │  MCP Tools  │  │   Indicators        │  │
│  │  + SSE      │──│             │──│  - VWAP             │  │
│  │  Transport  │  │             │  │  - Volume Profile   │  │
│  └─────────────┘  └─────────────┘  │  - Session Levels   │  │
│                          │         │  - Footprint        │  │
│  ┌─────────────┐         │         │  - Delta/CVD        │  │
│  │  Data Layer │◄────────┘         │  - Imbalance        │  │
│  │  - Cache    │                   │  - Depth Delta      │  │
│  │  - Storage  │                   └─────────────────────┘  │
│  │  - Orderbook│                                            │
│  └─────────────┘                                            │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Binance USD-M Futures                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  REST API   │  │  WebSocket  │  │   Streams           │  │
│  │  - Ticker   │  │  - aggTrade │  │  - depth@100ms      │  │
│  │  - OI Hist  │  │  - markPrice│  │  - forceOrder       │  │
│  │  - Funding  │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **WebSocket Streams** 提供实时数据：
   - `aggTrade`: 聚合交易数据，用于计算 VWAP、Footprint、Delta
   - `depth@100ms`: 订单簿增量更新
   - `markPrice@1s`: 标记价格和资金费率
   - `forceOrder`: 清算事件

2. **数据处理**：
   - 实时数据更新到内存缓存 (MemoryCache)
   - 聚合数据定期写入 SQLite (DataStorage)
   - 订单簿维护增量一致性 (OrderbookManager)

3. **指标计算**：
   - VWAP: 累积 price * volume 和 volume
   - Volume Profile: 按 tick size 聚合成交量
   - Session Levels: 按时间窗口跟踪高低点
   - Footprint: 按 timeframe 和价格等级聚合买卖量

4. **MCP 响应**：
   - Tools 调用时实时计算或从缓存返回
   - 响应格式为结构化 JSON
