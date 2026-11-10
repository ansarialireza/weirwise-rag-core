# Core Retrieval and Indexing

## Indexing path

1. `POST /api/v1/documents/upload` verifies tenant ownership, streams the file to disk under the
   configured size cap, validates supported file extensions/text encoding and relevant signatures,
   and deduplicates documents by knowledge base and content hash.
2. The API creates an ingestion job. Inline mode dispatches work through FastAPI background tasks;
   worker mode leaves jobs for the separate `server.app.worker` polling loop.
3. `server/app/ingestion.py` extracts text from PDF, DOCX, CSV, and UTF-8 text/Markdown and chunks
   content using configured size/overlap settings.
4. `server/app/worker.py` claims jobs with leases and fencing tokens, generates embeddings in
   configured batches, stages vectors on disk, then replaces chunks transactionally only while the
   worker still owns the job.
5. Document/job state moves to a terminal completed or failed state. Provider outages may be
   retried according to worker settings.

## Search path

`server/app/retrieval.py` embeds a query once and searches only the authenticated tenant's requested
knowledge base(s). PostgreSQL uses pgvector cosine distance and optional metadata filtering; the
non-PostgreSQL path computes cosine similarity in memory for tests. Results are ordered
deterministically and include chunk/document references, content, metadata, similarity, and rank.

Search supports optional `score_threshold`, `metadata_filters`, and multiple knowledge base IDs.
The query embedding cache is bounded and time-limited. Similarity scores are part of the public
contract; do not change their range, threshold meaning, or ordering without updating `CONTRACT.md`
and tests.

## Boundaries

- Extraction/chunking stay separate from provider I/O and database writes.
- Retrieval does not own document extraction or indexing.
- The worker does not serve HTTP.
- Tenant and knowledge-base ownership are checked before retrieval.
- Platform synchronizes through Core's public REST API; no Platform-specific rules belong in these
  indexing/retrieval modules.
