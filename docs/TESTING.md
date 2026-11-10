# Core Testing Strategy

Core tests are split between `server/tests/` and `packages/python-sdk/tests/`. Keep unit tests
independent of live databases/providers where practical, and use explicit integration tests for the
real PostgreSQL/pgvector boundary.

| Level | Examples and expectations |
|---|---|
| Unit | Chunking, provider retries/cache, schema validation, auth helpers, ranking, and failure handling. |
| Integration | Database constraints, PostgreSQL/pgvector search, ingestion jobs, and Core REST behavior against isolated services or controlled fakes. |
| Contract | Route/schema behavior, tenant authorization, error/status compatibility, and SDK serialization against the public REST surface. |
| End-to-end | A test-client flow through upload, indexing, search, assistant chat, and SDK operations. |
| Regression | Focused coverage for fixed issues such as duplicate uploads, lease loss, atomic chunk replacement, or cross-tenant access. |

Run the configured checks from the Core repository root:

```text
ruff check .
ruff format --check .
pytest server/tests packages/python-sdk/tests
```

CI is the source for external-service setup and supported Python versions. Tests must use a
disposable database, isolated storage directory, and fake or explicitly configured provider
credentials. Never run migrations against a shared/production database or use real customer files.
Report skipped PostgreSQL/provider integration tests clearly; a SQLite-only pass does not prove
PostgreSQL/pgvector behavior.
