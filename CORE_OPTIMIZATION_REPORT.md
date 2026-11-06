# Weirwise Core Optimization Report

Date: 2026-11-05
Repository: `weirwise-rag-core`, branch `develop`
Commits covered: `68e5a26` (resource bounds and retrieval latency), `f600dea` (multi-KB search tests), `fad5aff` (repository-relative `.env`), plus the earlier ingestion reliability fix `6ae346a`.
Environment: Windows host, Python 3.11, Core services on `127.0.0.1:8002/8003/8004`, WSL PostgreSQL 16 + pgvector, isolated QA databases.

No production system was touched. No embedding model was changed, no existing user data was re-embedded, and no schema migration was introduced. All work remains on `develop`.

---

## 1. Scope

The Core service is the retrieval and ingestion boundary. Every answer depends on two costs: embedding the query (remote provider) and scanning vectors (PostgreSQL). The optimization work targeted four risks identified in the architecture audit:

| Audit item | Risk | Resolution |
|---|---|---|
| B2 | Retrieval p50 ~1.2 s dominated by a per-query embedding round trip | Per-loop keep-alive HTTP pool + bounded TTL embedding cache + one embedding per multi-KB query |
| B8 | HNSW under-returns rows with selective knowledge-base filters | `SET LOCAL hnsw.ef_search` inside the retrieval transaction |
| B9 | Large documents could exhaust worker RSS | Disk-backed `numpy` memmap staging and batched inserts |
| B10 | An unbounded query could exhaust the host | `statement_timeout`, bounded pool, upload size cap, streamed upload |

## 2. What changed

### 2.1 Resource bounds

| Control | Value | Location |
|---|---|---|
| PostgreSQL statement timeout | 30 s (`db_statement_timeout_ms=30000`) | `server/app/config.py`, applied per statement/transaction |
| Connection pool | 10 connections + 20 overflow | `server/app/config.py`, `server/app/database.py` |
| HNSW search breadth | `hnsw.ef_search=100`, floor 40 | `server/app/config.py`, `server/app/retrieval.py` |
| Upload size cap | 50 MB, streamed in 1 MB chunks | `server/app/config.py`, `server/app/api.py` |
| Embedding batch size | 64 texts per provider call | `server/app/config.py`, `server/app/worker.py` |
| Embedding connection pool | 20 keep-alive connections per loop | `server/app/config.py`, `server/app/providers.py` |
| Embedding cache | 512 entries, 300 s TTL, keyed by base URL + model + dimensions + SHA-256 of text | `server/app/providers.py` |
| Chat connection pool | 10 connections | `server/app/providers.py` |

Uploads are spooled to a `.upload-*.part` file while hashing and size-checking, so the process never buffers a 50 MB file in memory. Partial files are removed on any failure path.

### 2.2 Latency work

1. **One embedding for multi-KB retrieval.** A search that targets several knowledge bases embeds the query once and reuses the vector, instead of paying the provider round trip per KB.
2. **Per-event-loop pooled HTTP clients.** Embedding and chat clients are cached per loop with `keepalive_expiry=120 s`, so consecutive requests skip DNS/TCP/TLS setup. A fresh client per call was the previous behavior.
3. **Bounded TTL embedding cache.** Identical query text within five minutes is served from a per-process LRU-style cache. The cache is bounded in entries and time, and keys include the provider base URL, model name and dimensions so a configuration change cannot serve a stale vector from another model.
4. **HNSW breadth inside the transaction.** `SET LOCAL hnsw.ef_search` is issued in the same transaction as the vector query. With a selective knowledge-base filter, the approximate index previously returned fewer rows than requested; raising `ef_search` to at least 40 (100 by default) trades a small amount of scan time for complete candidate sets.
5. **Batched ingestion.** Chunks are embedded 64 at a time and staged into a disk memmap, then inserted in batches instead of materializing the full matrix in RAM.

### 2.3 Correctness and reliability

- **Transient provider retry (`6ae346a`).** Embedding calls retry transient 429/5xx responses; permanent failures terminate the job with an explicit failure state instead of hanging.
- **Claim-token fencing.** A restarted worker cannot complete a job claimed by another worker; inline and worker modes share terminal-state handling.
- **Dimension and shape validation.** Every provider response is checked against the configured dimension count and the batch length before it is written; a malformed response fails the document rather than storing corrupt vectors.
- **Deterministic chunking and deduplication.** Chunk boundaries are deterministic for a given document; duplicate uploads reuse the same content hash.
- **Signature and text validation on upload.** PDF requires `%PDF-`, DOCX requires `PK\x03\x04`, and text formats are validated as UTF-8 chunk-by-chunk. Binary payloads no longer reach the database as replacement characters.
- **Cascade deletion with file cleanup.** Deleting a knowledge base removes its storage files, documents, and vectors (previously files were orphaned).
- **Repository-relative `.env` (`fad5aff`).** Configuration loading no longer depends on the process working directory.

## 3. Verification

### 3.1 Automated tests

```
cd weirwise-rag-core
python -m pytest server/tests -q
51 passed in 43.89s
```

The suite includes the new ingestion hardening tests (144 lines) and multi-KB scoping/request-validation tests (115 lines). Test runs use deterministic pseudo-embeddings when no provider key is configured; they do not call a remote provider.

The Python SDK tests under `packages/python-sdk/tests` are not installed as a package in this environment (`ModuleNotFoundError: weirwise` when collected from the repository root). They were not part of this result and are not counted as passing.

### 3.2 Retrieval quality baseline (unchanged by this work)

The recorded 16-query synthetic evaluation remains the quality reference:

| Metric | Value |
|---|---|
| Recall@1 / @3 / @5 | 1.000 / 1.000 / 1.000 |
| Precision@1 | 1.000 |
| MRR | 1.000 |
| p50 latency | 1,204 ms |
| p95 latency | 1,575 ms |
| 10 concurrent p95 | 1,672 ms |

These numbers were measured with real Gemini embeddings before the provider quota was exhausted. The optimization does not change the embedding model or the stored vectors, so they remain the valid quality reference for this dataset.

### 3.3 Post-optimization measurement status

| Measurement | Status | Reason |
|---|---|---|
| Retrieval latency after pooling/cache | **BLOCKED** | Gemini embedding quota exhausted; a live search cannot obtain a fresh query vector |
| Embedding cache hit rate | **NOT MEASURED** | Requires live provider calls to populate and replay queries |
| Ingestion throughput at 180 KB | **BLOCKED** | Quota exhausted; 43 KB completed in 7.4 s before the limit |
| Resource bounds under load | Verified by configuration and tests, not by a Core load campaign | Deliberately avoided unbounded vector queries after the prior WSL exhaustion incident |

No latency improvement is claimed for the post-optimization build. The changes are verified at the code and test level; the user-visible effect must be re-measured when provider quota is restored.

## 4. Behavior changes worth knowing

- **Cache staleness window.** A repeated query within 300 s can return a vector cached from the same text. This is safe for the fixed embedding model but must be invalidated by deployment if the model or dimensions ever change. The key includes both, so cross-model contamination cannot occur; a model change in configuration naturally misses the old entries.
- **Cache is per process.** Multiple workers do not share it. Warm-up after a restart is cold.
- **`ef_search` trade-off.** Higher values improve recall under selective filters and cost additional scan time. The default of 100 is a deliberate recall-first choice; it is configurable.
- **Inline ingestion still blocks the upload response.** Worker mode remains the production path for large documents.
- **Memmap staging needs disk space** roughly equal to `chunks × dimensions × 4 bytes` per in-flight document; it does not need the equivalent RAM.

## 5. Remaining Core work

1. Re-run the 16-query evaluation and a warm/cold cache latency comparison once embedding quota is available; record cache hit rate.
2. Add a knowledge-base update endpoint (`PATCH`/`PUT`); currently absent (404/405).
3. Add a relevance gate and answer-attribution grounding so refusals do not carry citations.
4. Evaluate a hybrid lexical signal (or reranker) for exact identifiers, only after the latency budget is re-measured.
5. Add conversation-aware query rewriting for follow-up turns; retrieval currently embeds the raw follow-up.
6. Provision a small, curated evaluation set from realistic documents rather than the synthetic corpus.

## 6. Conclusion

The Core now has explicit resource ceilings, pooled provider connections, a bounded query cache, complete multi-KB retrieval, and file-hygiene fixes. These are structurally sound changes covered by 51 passing tests. The measurable performance benefit is **not yet proven** because the embedding provider quota is exhausted; this report therefore does not claim a latency improvement. The next evidence step is a quota-restored warm/cold measurement of the same 16-query workload.
