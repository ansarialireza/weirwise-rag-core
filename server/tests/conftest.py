import os

# asyncpg connections belong to the event loop that opened them and
# pytest-asyncio gives every test a fresh loop. A pooled engine would hand
# dead connections to the next test, so disable pooling for the suite.
os.environ.setdefault("WEIRWISE_DB_POOL_ENABLED", "false")
