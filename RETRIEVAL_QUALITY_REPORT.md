# Weirwise Retrieval Quality Report

Date: 2026-11-05
Scope: Core vector retrieval (`weirwise-rag-core`) and the Platform's retrieval-to-answer path (`weirwise-rag-platform`).
Environment: isolated QA databases, PostgreSQL 16 + pgvector, 1536-dimension vectors. Provider calls were real when measured; the provider quota is now exhausted for live re-measurement.

This report separates what was measured from what is assumed. No accuracy claim is made beyond the recorded dataset.

---

## 1. Measured result (reference dataset)

The recorded 16-query synthetic corpus evaluation used real Gemini embeddings and the production retrieval path:

| Metric | Value | Interpretation |
|---|---:|---|
| Recall@1 | 1.000 | The correct chunk was always first |
| Recall@3 | 1.000 | The correct chunk was always in the top three |
| Recall@5 | 1.000 | The correct chunk was always in the top five |
| Precision@1 | 1.000 | The first result was always relevant |
| MRR | 1.000 | The first relevant result was always rank 1 |
| p50 latency | 1,204 ms | Dominated by the remote embedding round trip |
| p95 latency | 1,575 ms | Same |
| 10 concurrent p95 | 1,672 ms | No collapse under ten concurrent searches |

A separate end-to-end check produced a grounded answer with the correct citation at retrieval score 0.6097, and the answer text quoted the source ("Standard shipping takes 2 to 3 business days inside the EU [1]").

**These results validate that the pipeline retrieves and cites the right passage on a small, curated corpus. They do not establish broad-domain or production accuracy.** A perfect score on sixteen synthetic queries is a smoke test, not a benchmark.

## 2. What changed in retrieval

| Change | Semantic effect | Evidence |
|---|---|---|
| One embedding per multi-KB search | None; the same query vector is reused across knowledge bases | `f600dea` tests scoping and request validation |
| Bounded TTL embedding cache | None within the TTL for identical text; keys include model, dimensions and base URL | Code inspection; hit rate unmeasured |
| Per-loop connection pooling | None | Code inspection |
| `SET LOCAL hnsw.ef_search` (default 100, floor 40) | Can only increase candidate recall under selective filters; the ordering metric is unchanged | Code inspection and retrieval tests |
| Batched ingestion with memmap staging | None on stored vectors; reduces memory | Worker tests, 51-pass suite |
| Upload signature/UTF-8 validation | Rejects payloads that were previously indexed as garbage | Ingestion hardening tests |

No embedding model change and no re-embedding of existing data occurred. The quality reference therefore remains comparable.

## 3. Multi-knowledge-base behavior

Multi-KB retrieval embeds the query once, applies the knowledge-base filter in SQL, orders by cosine distance, and caps results at `top_k`. The added tests (`server/tests/test_multi_kb_search.py`) cover:

- results are confined to the requested knowledge bases;
- an empty or malformed request is rejected;
- one embedding call serves the whole multi-KB request.

Cross-tenant isolation was verified in the functional audit: a foreign project or knowledge base returns not-found rather than data.

## 4. Quality gaps (real, unresolved)

| ID | Gap | Impact | Status |
|---|---|---|---|
| Q1 | Synthetic 16-query corpus | Saturated metrics; no evidence of broad-domain quality | Open |
| Q2 | No reranker and no lexical/hybrid signal | Exact identifiers, product codes and rare tokens rely entirely on embedding similarity | Open |
| Q3 | Follow-up questions are embedded raw | A pronoun-only follow-up can retrieve the wrong passage; one drift was observed before a passing re-run | Open |
| Q4 | Citations are returned on refusal | A refusal can look grounded and carry sources | Open |
| Q5 | `is_grounded` reflects "citations attached", not answer attribution | Downstream analytics can over-report grounding | Open |
| Q6 | No Persian-language retrieval evaluation | Multilingual quality is unverified | Open |
| Q7 | No evaluation on customer-like documents | Domain shift is unmeasured | Open |

## 5. Measurement status after the optimization commit

A post-optimization quality and latency re-run was attempted in planning but is **BLOCKED**:

- the Gemini embedding quota is exhausted, so a fresh query vector cannot be generated;
- re-embedding the corpus with a different provider is prohibited without an explicit migration, backup and approval;
- the Jina adapter is also unavailable from this network (HTTP 451), so it is not a substitute.

Consequently this report does not claim that retrieval became faster or slower. The structural changes either leave relevance unchanged (cache, pooling, batching) or can only add candidates (higher `ef_search`).

## 6. Recommended evaluation plan

1. When quota is restored, replay the same 16 queries plus a curated set of at least 100 realistic queries with graded relevance judgments.
2. Record Recall@k, MRR, nDCG@10, and answer-attribution accuracy separately from retrieval relevance.
3. Measure warm-cache and cold-cache p50/p95, and the cache hit rate for a realistic repeat-query mix.
4. Add a refusal relevance gate: no sources below a calibrated score threshold.
5. Evaluate a hybrid lexical signal only after the latency budget is re-established; do not stack a reranker on an unmeasured baseline.
6. Add conversation-aware query rewriting and test follow-up turns explicitly.

## 7. Conclusion

Retrieval is correct on the small reference dataset and the pipeline is structurally sound, but quality is **not proven at production breadth**. The honest summary is: 1.000 on sixteen curated queries, no reranking, no hybrid search, no multilingual evidence, and no post-quota re-measurement. Any customer-facing accuracy claim beyond this dataset would be unsupported.
