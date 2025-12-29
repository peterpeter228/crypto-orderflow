import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini("asyncio_mode", "compatibility shim for asyncio-marked tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: alias to run coroutine tests using pytest-anyio",
    )


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Map legacy asyncio marker to anyio so async tests run without pytest-asyncio."""
    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.anyio)
