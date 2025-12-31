import asyncio
import time

import pytest

from src.main import CryptoOrderflowServer


@pytest.mark.anyio
async def test_startup_backfill_timeout_unblocks(monkeypatch):
    server = CryptoOrderflowServer()
    settings = server.settings
    monkeypatch.setattr(settings, "backfill_enabled", True)
    monkeypatch.setattr(settings, "backfill_block_startup", True)
    monkeypatch.setattr(settings, "backfill_block_startup_timeout_ms", 3_000)

    async def slow_backfill():
        await asyncio.sleep(5)

    monkeypatch.setattr(server, "_startup_backfill", slow_backfill)

    start = time.monotonic()
    await server._launch_backfill()
    elapsed = time.monotonic() - start

    assert elapsed < 4
    assert server._backfill_task is not None
    assert not server._backfill_task.done()

    # Cleanup running task
    server._backfill_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await server._backfill_task
