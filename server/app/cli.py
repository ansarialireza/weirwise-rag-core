import sys
import asyncio
import argparse
from sqlalchemy import select, text
from server.app.database import AsyncSessionLocal, engine, Base
from server.app.models import Tenant, APIKey
from server.app.auth import generate_api_key, hash_api_key
from server.app.config import get_settings


async def check_db_cmd() -> int:
    settings = get_settings()
    masked_url = settings.database_url
    if "@" in masked_url:
        prefix = masked_url.split("@")[0]
        suffix = masked_url.split("@")[1]
        scheme = prefix.split("://")[0] if "://" in prefix else ""
        masked_url = f"{scheme}://***:***@{suffix}"

    print("=" * 60)
    print(" Weirwise System & Database Diagnostics")
    print("=" * 60)
    print(f" Database URL : {masked_url}")
    print(f" Ingestion Mode: {settings.ingestion_mode}")
    print(f" Chunk Size   : {settings.chunk_size} (overlap: {settings.chunk_overlap})")
    print("-" * 60)

    try:
        async with engine.connect() as conn:
            # 1. Check basic connection
            res = await conn.execute(text("SELECT 1"))
            res.scalar()
            print(" [PASS] PostgreSQL connection successful.")

            # 2. Check dialect and vector extension
            if conn.dialect.name == "postgresql":
                ver_res = await conn.execute(text("SHOW server_version;"))
                pg_version = ver_res.scalar()
                print(f" [PASS] PostgreSQL server version: {pg_version}")

                ext_res = await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                )
                ext_version = ext_res.scalar()
                if ext_version:
                    print(f" [PASS] pgvector extension installed: v{ext_version}")
                else:
                    print(
                        " [WARN] pgvector extension is NOT installed in current database.\n"
                        "        Run 'CREATE EXTENSION IF NOT EXISTS vector;' or check Docker image."
                    )
            else:
                print(f" [INFO] Dialect is '{conn.dialect.name}', vector extension check skipped.")

        print("=" * 60)
        print(" All diagnostic checks passed.")
        print("=" * 60)
        return 0
    except Exception as e:
        print(" [FAIL] Database connectivity check failed:")
        print(f"        {type(e).__name__}: {e}")
        print("=" * 60)
        return 1


async def create_tenant_cmd(
    tenant_name: str, key_name: str = "default", raw_api_key: str | None = None
):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # Check or Create Tenant
        tenant_res = await session.execute(select(Tenant).where(Tenant.name == tenant_name))
        tenant = tenant_res.scalars().first()
        if not tenant:
            tenant = Tenant(name=tenant_name, status="active")
            session.add(tenant)
            await session.flush()

        # Generate or use provided API Key
        if raw_api_key:
            raw_key = raw_api_key
            key_hash = hash_api_key(raw_key)
        else:
            raw_key, key_hash = generate_api_key(prefix="ww_")

        key_res = await session.execute(select(APIKey).where(APIKey.key_hash == key_hash))
        api_key = key_res.scalars().first()
        if not api_key:
            api_key = APIKey(
                tenant_id=tenant.id,
                key_hash=key_hash,
                name=key_name,
                is_active=True,
            )
            session.add(api_key)
            await session.commit()
            print(" [PASS] Created new API key.")
        else:
            print(" [INFO] API key already registered for this tenant.")

        print("=" * 60)
        print(" Weirwise Tenant & API Key Configured Successfully")
        print("=" * 60)
        print(f" Tenant ID : {tenant.id}")
        print(f" Tenant Name: {tenant.name}")
        print(f" Key Name   : {key_name}")
        print("=" * 60)


def cli():
    parser = argparse.ArgumentParser(description="Weirwise Admin CLI")
    subparsers = parser.add_subparsers(dest="command")

    admin_parser = subparsers.add_parser("admin")
    admin_sub = admin_parser.add_subparsers(dest="subcommand")

    create_tenant_p = admin_sub.add_parser("create-tenant")
    create_tenant_p.add_argument("--name", required=True, help="Tenant display name")
    create_tenant_p.add_argument(
        "--key-name", default="default", help="Descriptive name for the API key"
    )
    create_tenant_p.add_argument(
        "--key", default=None, help="Explicit API key string to register (optional)"
    )

    admin_sub.add_parser("check-db", help="Diagnose database connection and pgvector extension")

    # Also support top-level 'check-db' alias for convenience
    subparsers.add_parser("check-db", help="Diagnose database connection and pgvector extension")

    args = parser.parse_args()

    if args.command == "admin" and args.subcommand == "create-tenant":
        asyncio.run(create_tenant_cmd(args.name, args.key_name, args.key))
    elif (args.command == "admin" and args.subcommand == "check-db") or args.command == "check-db":
        exit_code = asyncio.run(check_db_cmd())
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
