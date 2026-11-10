# Core Performance and Reliability

This guide records mechanisms present in source; it does not promise a latency, throughput, or
availability target. Benchmarks and reports under `benchmarks/` and `docs/engineering/` are
environment-specific evidence and should be dated and reproduced before being used as current
claims.

## Current mechanisms

- Uploads are streamed to disk with a configured hard cap rather than buffered as a full request.
- PostgreSQL retrieval uses pgvector cosine search and a configurable HNSW `ef_search`; tests can
  use an in-memory cosine fallback.
- Provider HTTP clients are pooled per event loop. Provider calls use bounded retry/backoff for
  transient failures; query embeddings use a bounded TTL LRU cache.
- Ingestion uses database job claims, leases, heartbeats, and fencing tokens. Embeddings are batched
  into a disk-backed matrix; chunk replacement is performed atomically after staging succeeds.
- Worker failures and stale leases are recorded/reclaimed according to configured attempt limits.

## Engineering expectations

- Measure representative workloads with the existing benchmark scripts before changing batch size,
  pool limits, indexes, or concurrency.
- Track p50/p95/p99 latency, throughput, memory, provider usage, and failures together; do not
  optimize one metric by hiding errors or weakening tenant isolation.
- Test large uploads/documents, provider timeouts/rate limits, worker interruption, duplicate jobs,
  and PostgreSQL query plans in isolated environments.
- Keep timeouts, retries, pool bounds, and concurrency explicit. Avoid unbounded in-memory
  accumulation and unbounded retries.
- Do not state measured values without naming the benchmark, hardware/service configuration, date,
  and test result.
