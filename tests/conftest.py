import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --runslow command line option."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked with @pytest.mark.slow unless --runslow is specified."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
