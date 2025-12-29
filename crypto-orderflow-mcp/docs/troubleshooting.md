# 故障排除指南

## 常见问题

### 1. 服务启动失败

**症状**：运行 `python run.py`（或 `python -m src.main`）后立即退出或报错

**可能原因及解决方案**：

#### 一直卡在 `Waiting for application startup.`（端口连不上）
如果你看到 Uvicorn 一直停留在：

```
INFO:     Waiting for application startup.
```

并且日志中反复出现 `rate_limit_wait`，通常是因为你在 `.env` 里设置了：

```
BACKFILL_BLOCK_STARTUP=true
```

这会让服务在启动时先执行历史回补（backfill），直到回补结束才会对外提供 HTTP/SSE 服务。

**解决**：把它改成 `false`（推荐，CherryStudio 远程连接场景必开）：

```
BACKFILL_BLOCK_STARTUP=false
```

然后重启服务。

#### 依赖缺失
```bash
# 确保安装了所有依赖
pip install -r requirements.txt

# 如果特定包失败，尝试单独安装
pip install aiosqlite websockets
```

#### Python 版本不兼容
```bash
# 检查 Python 版本，建议 3.10+
python --version

# 如果版本过低（<3.10），建议升级到 3.10/3.11，并安装 venv 支持
sudo apt update
sudo apt install -y python3 python3-venv
```

#### 端口被占用
```bash
# 检查端口占用
sudo lsof -i :8022

# 修改配置使用其他端口
export MCP_PORT=8023
```

---

### 2. WebSocket 连接失败

**症状**：日志显示 `connection_error` 或 `websocket connection failed`

**解决方案**：

#### 检查网络连接
```bash
# 测试能否访问 Binance
curl -I https://fapi.binance.com/fapi/v1/ping

# 测试 WebSocket 连接
wscat -c wss://fstream.binance.com/ws/btcusdt@aggTrade
```

#### 防火墙/代理问题
```bash
# 如果在企业网络，可能需要设置代理
export HTTPS_PROXY=http://proxy:port
export WSS_PROXY=http://proxy:port
```

#### 中国大陆访问
如果在中国大陆，可能需要：
1. 使用 VPN
2. 或使用 Binance 国内可访问的域名（如有）

---

### 3. 数据不更新

**症状**：调用 API 返回空数据或旧数据

**解决方案**：

#### 检查 WebSocket 状态
查看日志中是否有 `connected` 消息：
```
{"event": "connected_combined", "streams": [...]}
```

#### 重启服务
```bash
# Docker
docker-compose restart

# Systemd
sudo systemctl restart crypto-orderflow-mcp
```

#### 检查数据库
```bash
# 检查数据库文件
ls -la data/orderflow_cache.db

# 如果损坏，删除重建
rm data/orderflow_cache.db
# 重启服务会自动重建
```

---

### 4. MCP 工具调用失败

**症状**：Cherry Studio 显示工具调用错误

**解决方案**：

#### 检查参数格式
确保：
- `symbol` 是大写（如 `BTCUSDT`）
- `timeframe` 是支持的值（`1m`, `5m`, `15m`, `30m`, `1h`）
- `startTime` 和 `endTime` 是毫秒时间戳

#### 检查健康状态
```bash
curl http://localhost:8022/healthz
```

应返回：
```json
{"status": "healthy", "timestamp": 1705286400000}
```

---

### 5. 内存占用过高

**症状**：服务器内存持续增长

**解决方案**：

#### 限制数据保留
```bash
# 在 .env 中设置
DATA_RETENTION_DAYS=3  # 减少保留天数
```

#### 重启服务释放内存
```bash
sudo systemctl restart crypto-orderflow-mcp
```

#### 检查清理任务
查看日志中是否有 `cleanup_complete` 消息。

---

### 6. 响应速度慢

**症状**：API 响应需要很长时间

**解决方案**：

#### 减少查询范围
- 使用更短的时间范围
- 减少 `limit` 参数值

#### 检查数据库性能
```bash
# 优化 SQLite
sqlite3 data/orderflow_cache.db "VACUUM;"
```

#### 增加资源
```bash
# 增加 Docker 内存限制
# docker-compose.yml
services:
  crypto-orderflow-mcp:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

### 7. SSE 连接断开

**症状**：Cherry Studio 显示连接中断

**解决方案**：

#### 检查网络稳定性
长连接对网络要求较高，确保：
- 网络稳定
- 没有代理超时设置
- 防火墙允许长连接

#### 使用 HTTP 方式
如果 SSE 不稳定，尝试切换到 HTTP 传输：
```
URL: http://localhost:8022/mcp
传输类型: HTTP
```

#### 反向代理/Nginx 场景（强烈建议检查）
如果你把服务放在 Nginx / Caddy / Traefik 后面，默认可能会 **buffer SSE** 或者 **超时断开**。Nginx 参考配置：

```nginx
location /sse {
    proxy_pass http://127.0.0.1:8022;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    chunked_transfer_encoding off;
    proxy_read_timeout 3600;
    proxy_send_timeout 3600;
}

location /messages/ {
    proxy_pass http://127.0.0.1:8022;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_read_timeout 3600;
    proxy_send_timeout 3600;
}
```

并在 `.env` 里设置（让 Cherry Studio 远程握手拿到正确的 POST 地址）：

```bash
MCP_PUBLIC_URL=http://你的域名或公网IP:8022
```

---

## 日志分析

### 查看日志

**Docker**：
```bash
docker-compose logs -f crypto-orderflow-mcp
```

**Systemd**：
```bash
journalctl -u crypto-orderflow-mcp -f
```

**本地运行**：
日志直接输出到控制台。

### 常见日志消息

| 消息 | 含义 |
|------|------|
| `starting_server` | 服务正在启动 |
| `connected_combined` | WebSocket 连接成功 |
| `database_initialized` | 数据库初始化完成 |
| `orderbook_initialized` | 订单簿初始化完成 |
| `tool_called` | MCP 工具被调用 |
| `cleanup_complete` | 数据清理完成 |
| `connection_error` | 连接错误（需要关注）|
| `rate_limit_hit` | 触发限流（需要关注）|

### 开启调试模式

```bash
# 在 .env 中设置
LOG_LEVEL=DEBUG
DEBUG=true
```

---

## 获取帮助

如果以上方案无法解决问题：

1. 收集日志信息
2. 记录错误消息
3. 描述复现步骤
4. 提交 Issue 到项目仓库
