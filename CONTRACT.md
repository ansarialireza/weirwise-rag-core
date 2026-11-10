# Weirwise Core REST API Contract (1.0.0)

This document summarizes the public REST contract implemented by `server/app/`. Exact generated
schemas are available from the FastAPI OpenAPI document in a running development/test service;
route declarations and Pydantic models are authoritative if this summary differs.

## Base path and authentication

- The service mounts the versioned API under `/api/v1`.
- Tenant-scoped API requests use the `X-API-Key` header. The server stores a SHA-256 hash of a
  generated key; raw keys are shown only when created by the administrative CLI.
- Health endpoints are unauthenticated.
- Core tenants and keys are owned and managed by Core. Platform identities/credentials are not
  interchangeable with Core tenant keys.

## Endpoints

### Health

- `GET /health/live` — process liveness; returns `status: alive`, `database: unverified`, and the
  configured service version.
- `GET /health/ready` — checks the database. A healthy result has `status: ready` and
  `database: healthy`; a failed check returns `status: degraded` and an `unhealthy: <exception
  type>` database value.

### Knowledge bases and documents

- `POST`, `GET /api/v1/knowledge-bases`; `GET`, `DELETE /api/v1/knowledge-bases/{kb_id}`.
- `POST /api/v1/documents/upload` accepts multipart `knowledge_base_id` and `file`, validates
  supported file extensions and configured size, deduplicates by knowledge-base/content hash, and
  returns `202 Accepted` with a document/job response.
- `GET /api/v1/documents/{doc_id}` and `GET /api/v1/knowledge-bases/{kb_id}/documents` expose
  tenant-scoped document state.
- `DELETE /api/v1/documents/{doc_id}` removes a tenant-owned document.
- Ingestion states include `pending`, `processing`, `completed`, and `failed`.

### Search

- `POST /api/v1/search` performs tenant-scoped semantic search.
- The request accepts a non-empty `query`, `top_k`, optional `score_threshold`, optional
  `metadata_filters`, and either a `knowledge_base_id` or multiple `knowledge_base_ids`.
- `score_threshold` is a minimum cosine-similarity value in the supported schema range. Results
  include chunk/document identifiers, content, metadata, similarity score, and rank.

### Assistants and conversations

- `POST`, `GET /api/v1/assistants`; `GET`, `DELETE /api/v1/assistants/{assistant_id}`.
- `POST /api/v1/assistants/{assistant_id}/chat` accepts a message and optional conversation ID and
  returns an assistant message with source attribution.
- `GET /api/v1/conversations/{conversation_id}/messages` returns tenant-scoped conversation
  history.

## Compatibility rules

The REST API is Core's public service boundary and is consumed independently of the Python SDK.
Treat route paths, authentication, required/optional fields, defaults, status codes, error behavior,
tenant scoping, ingestion states, similarity-score semantics, and source fields as compatibility
surface.

For a contract change:

1. Inspect both route and Pydantic schema implementations and all SDK/Platform callers.
2. Decide whether the change is backward-compatible. For a breaking change, document the new
   contract and migration/deprecation strategy before release.
3. Add route/contract tests and corresponding SDK tests; update this document and SDK docs.
4. Preserve tenant scoping and Core's independence from Platform.

Core package releases (for example, `weirwise-core` 1.0.0) are independent from Weirwise product
roadmap versions such as V1.0. Do not use product version names for Core API or package releases.
