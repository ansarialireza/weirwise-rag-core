# Core Functional Test Report

Date: 2026-11-05
Repository: `weirwise-rag-core` @ `develop` (commit `6ae346a`)
Service under test: FastAPI Core API on `127.0.0.1:8003`
Database: PostgreSQL 16 + pgvector, database `weirwise_audit_qa`
Harness: `artifacts/audit/core_audit_main.py` (gitignored)
Raw evidence: `artifacts/audit/core_results.json`, `artifacts/audit/core_recheck.json`

## 1. Scope and method

Every test below was executed against the real running Core service, the real PostgreSQL/pgvector
database, and real provider APIs. No mocks were used for the functional paths.

| Component | Configuration |
|---|---|
| Embedding provider | Google Gemini `gemini-embedding-001` via OpenAI-compatible endpoint, 1536 dimensions, real network calls |
| Chat provider | OpenRouter `openai/gpt-4o-mini`, real network calls |
| Auth | Real X-API-Key secrets hashed with SHA-256; a revoked key and a suspended tenant were created for negative tests |
| Data | Dedicated QA tenant and knowledge bases; synthetic documents only; no customer data |
| Corpus | 16-query evaluation corpus (15 scored queries + 1 rejection probe), one document per expected answer, plus distractor documents |
| Harness resilience | Client retry on transport errors (`RetryClient`), server-side embedding retries, incremental evidence flushing |

Test identifiers used below (`A01` … `J04`) map 1:1 to rows in `core_results.json`.

## 2. Result summary

| Classification | Count | Notes |
|---|---:|---|
| PASS | 103 | Includes `C26` and `I06`, reclassified after focused recheck (harness timing/expectation artifacts, see §4) |
| FAIL | 5 | `B05`, `C22`, `I05`, `J02`, `G04` |
| PARTIAL | 1 | `C18` (PDF chunk metadata) |
| BLOCKED | 1 | `J03` (~180 KB document, Gemini quota exhaustion) |
| Total | 110 | |

The full harness initially recorded 101 PASS / 8 FAIL / 1 PARTIAL. Two of the eight failures were
harness artifacts, verified independently; one is environment-bound (quota). Five are genuine product
defects.

### Verified code changes shipped with this audit

Commit `6ae346a` on `develop` (`fix(ingestion): retry transient embedding failures and recover inline jobs`):

| File | Change | Verification |
|---|---|---|
| `server/app/providers.py` | Retry transient embedding failures (429/500/502/503/504, transport errors) up to 4 attempts with 4/12/25 s backoff, honour `Retry-After` (cap 30 s); non-transient 4xx still fail immediately; raise `ProviderUnavailableError` after exhaustion | Unit tests pass; live ingestion recovered from provider 503/429 during the audit |
| `server/app/worker.py` | Only retry transient provider failures; permanently failed jobs mark the document `failed`; inline jobs are re-dispatched via `_retry_inline_job` so they reach a terminal state instead of remaining `pending` | `C27` passes; focused recheck shows re-upload reaches `failed`; 43,200-byte document completed in 7.4 s |

Test suite: `python -m pytest server/tests -q --ignore=server/tests/test_jina_embeddings_integration.py`
→ **40 passed in 33.23 s**. The excluded file contains two live Jina integration tests that fail on
this network because Jina returns HTTP 451 (environmental, unchanged from the previous session).
The Python SDK tests were not collected because the `weirwise` SDK package is not installed in this
environment (not a Core defect; out of scope for this audit).

## 3. Detailed results

### 3.1 Authentication, authorization, error handling (`A`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| A01 | Liveness probe | 200 alive | 200 `{"status":"alive","database":"unverified","version":"1.0.0"}` | PASS |
| A02 | Readiness probe | 200 DB healthy | 200 `{"status":"ready","database":"healthy"}` | PASS |
| A03 | Missing API key | 401 | 401 `Missing API key in X-API-Key header` | PASS |
| A04 | Invalid API key | 401 | 401 `Invalid or revoked API key` | PASS |
| A05 | Revoked API key | 401 | 401 `Invalid or revoked API key` | PASS |
| A06 | Suspended tenant, valid key | 403 | 403 `Tenant account is inactive or disabled` | PASS |
| A07 | Unknown route | 404 | 404 | PASS |
| A08 | Empty KB name | 422 | 422 `string_too_short` | PASS |
| A09 | Malformed UUID | 422 | 422 `uuid_parsing` | PASS |
| A10 | Unknown KB | 404 | 404 | PASS |
| A11 | `limit=101` | 422 | 422 `limit must be between 1 and 100` | PASS |
| A12 | Negative offset | 422 | 422 `offset must be >= 0` | PASS |

### 3.2 Knowledge bases and tenancy (`B`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| B01 | Create KB | 201 + id | 201, id `981cf8da-…`, name/description persisted (96 ms) | PASS |
| B02 | Get KB | 200 | 200 matching record | PASS |
| B03 | List KBs | contains new KB | 200, 7 KBs including new | PASS |
| B04 | Pagination | 1 item with `limit=1` | 200, len 1 | PASS |
| B05 | **Update KB (rename/description)** | 200 updated | **PATCH 405, PUT 405; no update route in OpenAPI** | **FAIL** |
| B06 | Cross-tenant KB read | 404 | 404 | PASS |
| B07 | Cross-tenant search | 404 | 404 | PASS |
| B08 | Cross-tenant upload | 404 | 404 | PASS |

`B05` is a genuine contract gap: the only KB mutations are create, get/list, and delete. A caller
cannot rename a knowledge base or correct its description without deleting and recreating it, which
destroys all documents and vectors.

### 3.3 Ingestion, deduplication, formats (`C`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| C01 | Single text upload | 202 + doc id | 202 (251 ms) | PASS |
| C02 | Ingestion completes | completed | completed in 1.7 s, no error | PASS |
| C03 | Chunks persisted | >0 | 1 chunk | PASS |
| C04 | Embedding dimension | 1536 | 1536 | PASS |
| C05 | Listing includes upload | yes | 200, 1 doc | PASS |
| C06 | Upload to unknown KB | 404 | 404 | PASS |
| C07 | Upload without file | 422 | 422 | PASS |
| C08 | Duplicate content | 1 doc, already-indexed, no job | 202 `Document already indexed`, job `null`, rows 1, same id | PASS (API returns 202 for a no-op — cosmetic) |
| C09 | 5 concurrent identical uploads | 1 document, no 5xx | 1 row, codes `[202×5]`, losing requests report "already in progress" | PASS |
| C10 | Bulk 16-doc ingestion | 16 completed | 16/16 in 4.38 s (accept), slowest doc 0.9 s | PASS |
| C11 | Corpus chunk count | ≥16 | 16 | PASS |
| C12 | 10 concurrent uploads | 10 completed | 10/10 in 1.3 s | PASS |
| C13 | Markdown | completed | completed, 1 chunk, 2.5 s | PASS |
| C14 | CSV | completed | completed, 1 chunk, 2.5 s | PASS |
| C15 | JSON | completed | completed, 1 chunk, 1.7 s | PASS |
| C16 | Valid PDF | completed | completed, 1 chunk, 1.6 s | PASS |
| C17 | PDF text preserved | "24 months" present | 1 matching chunk | PASS |
| C18 | **PDF page metadata** | page number metadata | only `filename` + `chunk_index`; page number dropped | **PARTIAL** |
| C19 | Corrupted PDF | failed | failed in 0.9 s, 0 chunks, `Stream has ended unexpectedly` | PASS |
| C20 | Blank PDF (no text) | completed | completed, 0 chunks | PASS (see defect C-04) |
| C21 | Empty text file | completed | completed, 0 chunks | PASS (see defect C-04) |
| C22 | **Binary PNG rejected at upload** | 415/422 at upload | **202 accepted, then failed in chunk INSERT** | **FAIL** |
| C23 | DOCX binary accepted as text | completed | completed, 2 chunks, 1.8 s | PASS (content is binary noise; see defect C-03) |
| C24 | Text mislabeled as PDF | failed | failed in 0.9 s | PASS |
| C25 | Path traversal filename | stored under storage dir | stored as `var/qa_audit_storage\<uuid>.txt`, traversal neutralized | PASS |
| C26 | Re-upload invalid content reaches terminal state | terminal | focused recheck: `png_final=failed` (recorded initial harness read saw `pending` transiently) | PASS (harness timing artifact) |
| C27 | Delete then re-upload invalid content | terminal | 204 delete, 202 upload, final `failed` | PASS |

Additional ingestion defect observed: uploads are size-checked **after** the whole file is read
(`api.py:186` awaits `file.read()`), so a request can force full-file buffering before rejection
(see `J01`/`J02`).

### 3.4 Chunking (`D`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| D01 | Controlled document chunked | multiple chunks | 60 chunks | PASS |
| D02 | Sequential indices | 0…59 | 0…59 | PASS |
| D03 | Size limit | ≤500 chars | max 500 | PASS |
| D04 | Overlap | ~50 chars shared | overlap observed | PASS |
| D05 | Per-chunk metadata | filename + chunk_index | present | PASS |
| D06 | Oversized paragraph | split ≤500 | 5 chunks, max 499 | PASS |

Chunk sizes follow a 499/500/351 repeating pattern consistent with a 500-character window and
~50-character overlap.

### 3.5 Vector persistence (`E`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| E01 | Non-zero 1536-d vectors | no NULL embeddings | dims 1536, norm 0.697, null_count 0 | PASS |
| E02 | HNSW cosine index | present | `ix_chunks_embedding_hnsw` | PASS |
| E03 | Cosine scores sane | 0–1 | 0.844 | PASS |

### 3.6 Retrieval quality (`F`)

15 scored queries: 12 semantic, 1 exact-phrase, 1 near-distractor, 1 Persian. Each expected document
was returned at rank 1.

| Metric | Value |
|---|---:|
| Recall@1 | 1.000 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| Precision@1 | 1.000 |
| Precision@3 | 0.333 |
| Precision@5 | 0.200 |
| MRR | 1.000 |
| Latency avg | 1254.7 ms |
| Latency p50 | 1204.0 ms |
| Latency p95 | 1574.6 ms |
| Latency max | 1716.3 ms |

These are real semantic retrieval numbers from Gemini embeddings. Latency is dominated by the
outbound query-embedding call, not pgvector.

| ID | Test | Actual | Verdict |
|---|---|---|---|
| F01–F12 | Semantic queries | rank 1 for expected doc, top score 0.647–0.794 | PASS |
| F13 | Exact phrase (`blinking amber light…`) | rank 1, score 0.773 | PASS |
| F14 | Near-distractor (`extended warranty`) | rank 1 | PASS |
| F15 | Persian query | rank 1 | PASS |
| F16 | Aggregate metrics | table above | PASS |
| F17 | `top_k` honored | k=1 → 1; k=4 → 4 | PASS |
| F18 | Score threshold | threshold 0.55 → 2 hits; 0.80 → 0 hits | PASS |
| F19 | No-result | 200 `[]` | PASS |
| F20 | Filename filter | only `returns_shipping.md` | PASS |
| F21 | Filter with no match | empty | PASS |
| F22 | Numeric filter | `chunk_index=0` only | PASS |
| F23 | Cross-collection isolation | doc in B only | PASS |
| F24 | Same content in two KBs | scoped copies | PASS |
| F25 | Empty KB search | 200 `[]` | PASS |
| F26 | Unknown KB search | 404 | PASS |

### 3.7 Chat / grounded answers (`I`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| I01 | Create assistant bound to KB | 201 | 201, id `d8ff80e5-…` | PASS |
| I02 | Grounded answer | 3–5 business days + sources | "standard shipping takes 3 to 5 business days…", 4 sources, 2889 ms | PASS |
| I03 | Citation payload | chunk/document/score/metadata | all fields present, score 0.6988 | PASS |
| I04 | Unsupported question | explicit refusal | "i could not find it.", `refusal=True`; **but 4 sources still returned** | PASS with defect C-05 |
| I05 | **Multi-turn context** | follow-up references shipping timeframe | **"i could not find it."** — retrieval uses the raw follow-up only, no conversation-aware query rewrite | **FAIL** |
| I06 | History persisted/ordered | user/assistant ×2 | 6 entries ordered `u,a,u,a,u,a` (conversation was reused by I04; harness expected 4) | PASS (harness expectation artifact) |
| I07 | Unknown assistant | 404 | 404 | PASS |
| I08 | Cross-tenant chat | 404 | 404 | PASS |
| I09 | Assistant without KB | 200, no citations | 200, 0 sources | PASS |
| I10 | Assistant deletion | 204 then 404 | 204 then 404 | PASS |
| I11 | Cross-tenant history | 404 | 404 | PASS |

### 3.8 Concurrency and validation (`H`)

| ID | Test | Actual | Verdict |
|---|---|---|---|
| H01 | 10 concurrent searches | 0 errors, avg 1547 ms, p95 1672 ms | PASS |
| H02 | 20 concurrent metadata reads | 0 errors | PASS |
| H03 | Empty query | 422 | PASS |
| H04 | `top_k=0` | 422 | PASS |
| H05 | `top_k=51` | 422 (`≤50`) | PASS |
| H06 | Threshold >1 | 422 | PASS |
| H07 | Threshold <0 | 422 | PASS |

### 3.9 Memory, large files, deletion, cleanup (`J`, `G`)

| ID | Test | Expected | Actual | Verdict |
|---|---|---|---|---|
| J01 | Oversized upload | 413 | 413 at 52,428,800-byte cap (1.36 s) | PASS |
| J02 | **Bounded upload memory** | ≪ file size | **RSS +53.6 MB for a 51 MB file** (109.2 → 162.8 MB); full-file `read()` before size check | **FAIL** |
| J03 | ~180 KB document with real embeddings | completed | failed after 180.8 s, 0 chunks (Gemini quota exhaustion during run) | **BLOCKED** (environment/quota); a 43 KB document completed in 7.4 s in the focused recheck |
| J04 | File persisted to disk | exists | present under storage dir | PASS |
| G01 | Delete document | 204, 0 chunks, file removed | as expected | PASS |
| G02 | Bulk delete 10 docs | 0 docs/chunks | 10×204, 0/0, 419 ms | PASS |
| G03 | Delete KB cascades DB | 204, 0 docs/chunks | 204, 0/0 | PASS |
| G04 | **KB delete removes stored files** | 0 orphan files | **11/11 files remain on disk** | **FAIL** |
| G05 | Orphan ingestion jobs | 0 | 0 | PASS |
| G06 | Orphan chunks | 0 | 0 | PASS |

## 4. Reclassified results (transparency)

- **C26** — the main harness polled the document immediately after re-upload and observed `pending`,
  then recorded FAIL. The focused recheck (`core_recheck.json`) confirmed the document reaches
  terminal `failed` with a persisted error; the retry path reprocesses and fails deterministically.
  Product behavior is correct; the result is reclassified PASS and the initial reading was a
  read-before-terminal timing artifact.
- **I06** — the harness expected 4 history entries but the conversation had been reused by I04, so 6
  correctly ordered entries existed. Recorded FAIL was a harness expectation bug; reclassified PASS.

## 5. Defects found (Core)

| ID | Severity | Defect | Evidence | Recommendation |
|---|---|---|---|---|
| CORE-01 | Medium | No KB update endpoint (`PATCH`/`PUT` 405) | B05 | Add update route with name/description validation |
| CORE-02 | Medium | KB deletion leaves all uploaded files on disk | G04 (11/11 orphans) | Delete storage objects when a KB or document is removed |
| CORE-03 | Medium | Binary files accepted and mis-decoded as UTF-8 text; PNG/DOCX ingested as replacement-character noise and only fail incidentally at the DB | C22, C23 | Reject non-text MIME/binary content at upload (415) before enqueueing |
| CORE-04 | Low | Empty/blank documents report `completed` with 0 chunks | C20, C21 | Report a distinct `empty`/`no_text` status or warn instead of success |
| CORE-05 | Medium | Chat returns citations with a refusal, and has no relevance gate before grounding | I04 | Apply a minimum similarity threshold; return no sources when refusing |
| CORE-06 | Medium | Multi-turn retrieval ignores conversation context | I05 | Rewrite the follow-up query using recent history before embedding |
| CORE-07 | Low | PDF page numbers dropped from chunk metadata | C18 | Persist page numbers during extraction |
| CORE-08 | Medium | Upload path buffers whole file in RAM before size validation | J02 | Stream to disk with a hard byte cap; check size during transfer |
| CORE-09 | Low | Duplicate upload returns 202 though nothing is queued | C08 | Return 200/409 with an explicit "already indexed" body |

## 6. Limitations

- `J03` (large-document ingestion) could not be validated at ~180 KB because the authorized Gemini key
  hit its quota during the run; the same path completed a 43 KB document in 7.4 s. This is an
  environment limitation, not a demonstrated product failure.
- The Jina embedding integration tests remain blocked by HTTP 451 from this network; they were
  excluded from the suite run rather than counted as failures.
- Chat fluency was generated by a real LLM (`gpt-4o-mini`) for I02 but the audit did not measure
  hallucination rate at scale.
- Retrieval metrics use a 16-document synthetic corpus; they demonstrate correctness of the retrieval
  plumbing and real embedding quality on that corpus, not production-scale quality.

## 7. Conclusion

Core is functionally sound on the paths that matter most: authenticated tenancy, ingestion,
chunking, vector persistence, real semantic retrieval with perfect Recall@1/MRR on the audit corpus,
metadata filtering, isolation, concurrency, deletion cascades, and error handling. It is **not**
production-clean: binary/empty inputs are handled sloppily, KB metadata cannot be edited, deleting a
KB leaks files, chat can cite sources while refusing to answer, follow-up questions lose context, and
large uploads are fully buffered in memory. The two ingestion reliability fixes in `6ae346a` were
verified with the real provider and the full non-network test suite passes.
