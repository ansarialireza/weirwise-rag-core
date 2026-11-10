# Core Providers and Storage

## Provider boundary

Provider interfaces and their current OpenAI-compatible implementations live in
`server/app/providers.py`:

- `EmbeddingProvider` generates vectors; `OpenAIEmbeddingProvider` calls the configured
  OpenAI-compatible `/embeddings` endpoint.
- `ChatProvider` generates assistant text; `OpenAIChatProvider` calls the configured
  OpenAI-compatible `/chat/completions` endpoint.
- Retrieval and worker code depend on provider interfaces rather than owning HTTP transport.

Provider URLs, model identifiers, vector dimensions, timeouts, retry settings, and query-embedding
cache bounds are configured through the `WEIRWISE_` settings in `server/app/config.py`. Check
`.env.example` for names and defaults. Do not commit keys or expose them through API errors.

Transient upstream status/transport errors are retried and ultimately surface as
`ProviderUnavailableError`; non-transient errors fail explicitly. When a provider key is absent,
the current provider classes generate deterministic/simulated responses. This code path is not
environment-gated; it must not be mistaken for output from an external provider or presented as
production generation.

## Storage boundary

Core stores uploaded source files on the local filesystem under `WEIRWISE_STORAGE_DIR`. SQLAlchemy
stores document metadata, chunks, and vectors; PostgreSQL with pgvector is the primary vector-search
backend. SQLite-compatible paths are used for isolated tests, not as evidence of PostgreSQL
equivalence.

The storage setting currently defaults to a path under `./var/`; deployments and local developers
must choose a private writable location outside tracked source data, set it consistently for API
and worker processes, and back it up according to their own retention requirements. Core does not
provide an S3/object-storage adapter in the inspected implementation.

The ingestion worker stages large embedding matrices as temporary memory-mapped files under the
configured storage directory and removes them in cleanup paths. Do not put runtime uploads,
generated databases, or these staging artifacts into source control.
