# Cherry Studio 配置指南

## 连接 Crypto Orderflow MCP Server

### 前提条件

1. Crypto Orderflow MCP Server 已启动运行
2. Cherry Studio 已安装并更新到最新版本
3. 网络可访问服务器地址

### 步骤一：获取服务器 URL

根据你的部署方式，确定服务器 URL：

- **本地运行**: `http://localhost:8022`
- **Docker 部署**: `http://localhost:8022` 或 `http://服务器IP:8022`
- **远程服务器**: `http://your-server-ip:8022`

### 步骤二：在 Cherry Studio 中配置

1. 打开 Cherry Studio
2. 点击设置图标（齿轮）
3. 选择 "MCP Servers" 或 "模型上下文协议服务器"
4. 点击 "添加服务器" 或 "+"

### 步骤三：填写配置

**方式一：SSE 连接（推荐）**

```yaml
名称: Crypto Orderflow
URL: http://localhost:8022/sse
传输类型: SSE
```

**方式二：HTTP 连接**

```yaml
名称: Crypto Orderflow
URL: http://localhost:8022/mcp
传输类型: HTTP
```

### 步骤四：测试连接

1. 保存配置
2. 启用该 MCP 服务器
3. 在对话中尝试询问：
   - "获取 BTCUSDT 的市场快照"
   - "查看 BTC 今天的关键价位"
   - "分析最近 1 小时的 ETH 订单流"

## 使用示例

### 示例 1：市场概览

```
请给我 BTCUSDT 的当前市场概况，包括价格、资金费率和持仓量。
```

Cherry Studio 将调用 `get_market_snapshot` 返回实时数据。

### 示例 2：关键价位分析

```
分析 BTC 今天的关键价位，包括 VWAP、POC 和各时段高低点。
```

Cherry Studio 将调用 `get_key_levels` 返回完整的价位分析。

### 示例 3：订单流分析

```
查看 BTCUSDT 最近 4 小时的 15 分钟级别订单流数据，
包括 Delta、CVD 变化和是否有大量 Imbalance。
```

Cherry Studio 将调用：
1. `get_orderflow_metrics` 获取 Delta/CVD
2. 分析 Imbalance 数据

### 示例 4：清算监控

```
最近有什么 BTC 大额清算？多空各占多少？
```

Cherry Studio 将调用 `stream_liquidations` 分析清算数据。

### 示例 5：深度分析

```
BTC 的买卖盘深度如何？过去一小时变化趋势？
```

Cherry Studio 将调用 `get_orderbook_depth_delta` 分析订单簿。

## 高级用法

### 组合多个工具

你可以在一次对话中要求多维度分析：

```
帮我全面分析 BTCUSDT 当前的市场状况：
1. 当前价格相对于今日 VWAP 和 POC 的位置
2. 最近 1 小时的 Delta 和 CVD 趋势
3. 订单簿买卖盘强弱
4. 近期清算情况
5. 资金费率和持仓量变化

基于以上数据，给我一个交易建议。
```

### 设置时间范围

指定具体时间范围获取历史数据：

```
获取 2024-01-15 14:00 到 15:00 (UTC) 的 ETHUSDT 5分钟 Footprint 数据。
```

## 故障排除

### 问题：连接失败

1. 检查服务是否运行：
   ```bash
   curl http://localhost:8022/healthz
   ```
   
2. 检查防火墙设置
3. 确认 URL 正确

### 问题：数据延迟

1. 检查网络连接
2. 确认服务器负载
3. 查看服务器日志

### 问题：工具调用失败

1. 检查参数格式是否正确
2. 确认交易对支持（BTCUSDT, ETHUSDT）
3. 查看错误消息详情

## 性能优化

### 建议配置

对于频繁使用，建议：

1. 使用 SSE 连接而非 HTTP
2. 避免过大的时间范围查询
3. 合理设置 limit 参数

### 网络优化

如果服务器在远程：

1. 确保网络稳定
2. 考虑使用 VPN 或内网
3. 监控延迟情况
