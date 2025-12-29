import asyncio
import inspect
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as requiring an event loop")


def pytest_addoption(parser):
    parser.addini("asyncio_mode", "asyncio execution mode compatibility")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    test_function = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_function):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            signature = inspect.signature(test_function)
            filtered_args = {
                name: value
                for name, value in pyfuncitem.funcargs.items()
                if name in signature.parameters
            }
            loop.run_until_complete(test_function(**filtered_args))
        finally:
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
            asyncio.set_event_loop(None)
        return True
    return None
