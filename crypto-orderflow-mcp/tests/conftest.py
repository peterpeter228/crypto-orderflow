"""Test configuration and helpers."""

import asyncio
import inspect

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ini options expected by upstream plugins."""
    parser.addini("asyncio_mode", "asyncio mode compatibility shim", default="auto")


def pytest_configure(config: pytest.Config) -> None:
    """Register markers used in the suite."""
    config.addinivalue_line("markers", "asyncio: mark test as async to be run with asyncio")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run async test functions without requiring external plugins.

    This mirrors the minimal behavior of pytest-asyncio's default mode by
    detecting coroutine functions and running them in a fresh event loop.
    """
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func):
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(test_func(**kwargs))
        return True
    return None
