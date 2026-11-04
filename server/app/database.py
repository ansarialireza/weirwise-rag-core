from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from server.app.config import get_settings

settings = get_settings()

is_postgres = "postgresql" in settings.database_url

connect_args: dict = {}
engine_kwargs: dict = {
    "echo": (settings.log_level.upper() == "DEBUG"),
    "future": True,
}

if is_postgres:
    # Bound every statement so a pathological vector query cannot exhaust the
    # database; the previous audit campaign took the host down with one.
    connect_args["server_settings"] = {
        "application_name": "weirwise-core",
        "statement_timeout": str(settings.db_statement_timeout_ms),
    }
    if settings.db_pool_enabled:
        engine_kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
            pool_recycle=settings.db_pool_recycle_seconds,
            pool_pre_ping=True,
        )
    else:
        engine_kwargs["poolclass"] = NullPool
else:
    connect_args["check_same_thread"] = False
    engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(
    settings.database_url,
    connect_args=connect_args,
    **engine_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
