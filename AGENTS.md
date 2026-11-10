# Agent Engineering Instructions

## Purpose and architecture

Weirwise Core is independently reusable RAG infrastructure: a FastAPI REST service, asynchronous ingestion worker, PostgreSQL/pgvector persistence, retrieval and provider adapters, and a separately distributable typed Python SDK under `packages/python-sdk/`. This repository and its software are licensed under MIT; preserve the license and copyright notices. Keep Core independent from Platform and other product repositories. Preserve the public REST API and SDK contracts described in `CONTRACT.md` and the SDK package documentation. Treat changes to routes, schemas, authentication, errors, and SDK behavior as compatibility-sensitive; document and test any intentional contract change.

Keep ingestion, embedding, provider, retrieval, persistence, API, and worker responsibilities at their existing boundaries. Avoid new dependencies unless the task requires them and a smaller existing-library solution is insufficient. The server declares Python >=3.11; the separately packaged SDK declares Python >=3.9. Preserve both supported ranges and the configured Ruff/Pytest conventions. Consult `README.md`, `CONTRIBUTING.md`, `CONTRACT.md`, `SECURITY.md`, `docs/ARCHITECTURE.md`, the focused `docs/` engineering guides, `pyproject.toml`, and `.github/workflows/ci.yml` for canonical details.

The local instructions and documentation in this repository define the minimum workflow; Brand is
an optional shared reference, not a prerequisite for working on Core.

Core tenants are a Core-owned API-key isolation mechanism, not Platform customer accounts or
workspaces. Keep Platform account ownership, authentication, and authorization outside Core; any
Platform integration must use the public REST contract and explicitly authorized Core credentials.

## Working practices

Inspect relevant implementation, tests, docs, configuration, and Git status before editing. Plan multi-file, architectural, security-sensitive, API/SDK, and migration work before implementation. Keep changes focused. Add or update tests for meaningful behavior changes, including API and SDK compatibility where relevant. Follow existing Python style and configuration; use type annotations and clear async/resource lifecycle handling consistent with neighboring code.

Codex is explicitly authorized to commit and push changes to `develop` without further approval, including pushes that trigger automatic development deployment. Create and use feature branches and worktrees when useful. Merging into `main`, pushing directly to `main`, or triggering production deployment always requires explicit user approval. Never perform production deployment without that approval; do not assume repository/environment approval gates exist. Do not change deployment scripts without a dedicated implementation plan and review.

Keep changes focused and independently reviewable; tests do not replace inspection of the actual
diff. Pull requests must include the local CI/check evidence and review artifacts required by this
repository. Cross-repository governance in Brand may be consulted when available, but this
repository's instructions, tests, and local docs define the minimum workflow and do not require
Brand to be present.

## Verification

Run relevant checks when authorized and feasible, and report the exact commands and actual results. Never imply a test, migration, build, or deployment was run unless there is evidence. The documented checks are:

- `ruff check .`
- `ruff format --check .`
- `pytest server/tests packages/python-sdk/tests`
- PostgreSQL/pgvector tests require a disposable database and `RUN_POSTGRES_TESTS=1`; provider
  integrations require explicit `RUN_LIVE_CORE_TESTS=1` and credentials supplied through the
  environment. Never point either test mode at a shared or production database.
- CI also runs Alembic migrations against its disposable PostgreSQL/pgvector service and builds the SDK and Core distributions. Do not run migrations against an unspecified or shared database.

Pull requests must include the independent QA report and actual check evidence required by `.github/PULL_REQUEST_TEMPLATE.md`.

## Security and data handling

Never expose secrets or inspect secret values unnecessarily. Never commit credentials, private keys, tokens, customer documents, or customer data. Use `.env.example` as the configuration reference; keep actual environment files private. Preserve tenant scoping and API-key hashing. Treat uploaded documents, local databases, storage, `data/`, `var/`, and generated reports/artifacts as mutable or sensitive runtime state. Keep runtime data, uploads, generated databases, caches, and mutable artifacts outside tracked source directories; preserve the repository's ignore rules. Do not repurpose real customer data as test fixtures.

Inspect before modifying. Preserve existing functionality. Never shut down Windows; if a shutdown timer is pending, cancel it with `shutdown /a`.
