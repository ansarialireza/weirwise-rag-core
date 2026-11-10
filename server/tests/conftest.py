import os

import pytest

# asyncpg connections belong to the event loop that opened them and
# pytest-asyncio gives every test a fresh loop. A pooled engine would hand
# dead connections to the next test, so disable pooling for the suite.
os.environ.setdefault("WEIRWISE_DB_POOL_ENABLED", "false")


def pytest_collection_modifyitems(items):
    if os.getenv("RUN_POSTGRES_TESTS") == "1":
        return

    skip_postgres = pytest.mark.skip(
        reason="PostgreSQL tests are opt-in; use a disposable database and set RUN_POSTGRES_TESTS=1"
    )
    for item in items:
        if item.get_closest_marker("postgres"):
            item.add_marker(skip_postgres)
