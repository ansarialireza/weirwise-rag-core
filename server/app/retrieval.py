import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from server.app.models import DocumentChunk
from server.app.providers import EmbeddingProvider

logger = logging.getLogger(__name__)


async def retrieve_relevant_chunks(
    db: AsyncSession,
    tenant_id: UUID,
    knowledge_base_id: Optional[UUID],
    query: str,
    embedding_provider: EmbeddingProvider,
    top_k: int = 4,
    score_threshold: Optional[float] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    knowledge_base_ids: Optional[List[UUID]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks for query using pgvector cosine distance, scoped to tenant & KB(s).

    score_threshold (optional): filters out chunks whose similarity_score is strictly less than this threshold.
    knowledge_base_ids (optional): search several KBs with a single query embedding.
    """
    # 1. Generate query embedding (cached for identical text by the provider)
    query_embeddings = await embedding_provider.get_embeddings([query])
    query_vector = query_embeddings[0]

    # Check dialect for vector distance vs in-memory fallback
    bind = db.bind
    dialect_name = bind.dialect.name if bind else "postgresql"

    kb_ids = list(knowledge_base_ids) if knowledge_base_ids else (
        [knowledge_base_id] if knowledge_base_id else []
    )
    if not kb_ids:
        return []

    results = []

    if dialect_name == "postgresql":
        from sqlalchemy import text as sa_text

        # HNSW is approximate: a small ef_search plus a selective KB filter can
        # return fewer rows than requested. Raise it per transaction; exact plans
        # ignore the hint.
        try:
            from server.app.config import get_settings

            ef_search = max(40, int(get_settings().retrieval_hnsw_ef_search))
            async with db.begin_nested():
                await db.execute(sa_text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
        except Exception:
            pass

        # Native pgvector cosine distance: embedding.cosine_distance(query_vector)
        # Cosine distance ranges from 0 (identical) to 2 (opposite). Similarity = 1 - distance.
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")

        # Base filter conditions
        conditions = [
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.knowledge_base_id.in_(kb_ids),
            DocumentChunk.embedding.isnot(None),
        ]
        # In cosine distance: similarity = 1 - distance >= score_threshold  <=>  distance <= 1 - score_threshold
        if score_threshold is not None:
            max_distance = 1.0 - float(score_threshold)
            conditions.append(DocumentChunk.embedding.cosine_distance(query_vector) <= max_distance)

        # Apply metadata filtering (JSON containment / exact match)
        if metadata_filters:
            from sqlalchemy import cast
            from sqlalchemy.dialects.postgresql import JSONB

            for key, val in metadata_filters.items():
                conditions.append(cast(DocumentChunk.chunk_metadata, JSONB).contains({key: val}))

        stmt = (
            select(DocumentChunk, distance_col)
            .where(*conditions)
            .order_by(distance_col.asc(), DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
            .limit(top_k)
        )

        res = await db.execute(stmt)
        rows = res.all()

        for rank, (chunk, distance) in enumerate(rows, start=1):
            similarity_score = max(0.0, 1.0 - float(distance))
            if score_threshold is not None and similarity_score < score_threshold:
                continue
            results.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.chunk_metadata or {},
                    "similarity_score": round(similarity_score, 4),
                    "rank": rank,
                }
            )
    else:
        # Fallback for testing environments without pgvector (e.g. SQLite test engine)
        import numpy as np

        stmt = select(DocumentChunk).where(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.knowledge_base_id.in_(kb_ids),
        )
        res = await db.execute(stmt)
        chunks = res.scalars().all()

        scored_chunks = []
        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)

        for chunk in chunks:
            # Metadata filter check for fallback
            if metadata_filters:
                meta = chunk.chunk_metadata or {}
                if not all(meta.get(k) == v for k, v in metadata_filters.items()):
                    continue

            if chunk.embedding is not None:
                c_vec = np.array(chunk.embedding)
                c_norm = np.linalg.norm(c_vec)
                # Compute raw cosine similarity; if either vector has zero norm, raw cosine is 0.0
                raw_sim = (
                    float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                    if (q_norm * c_norm) > 0
                    else 0.0
                )
                # Filter against raw cosine similarity: identical to PostgreSQL's distance <= (1.0 - threshold)
                # because distance = 1.0 - raw_sim <=> 1.0 - distance = raw_sim >= threshold
                if score_threshold is not None and raw_sim < score_threshold:
                    continue
                # Public similarity score contract clamps non-positive similarity to 0.0 (domain [0.0, 1.0])
                sim = max(0.0, raw_sim)
                scored_chunks.append((chunk, sim))

        scored_chunks.sort(key=lambda x: (-x[1], x[0].chunk_index, str(x[0].id)))
        top_chunks = scored_chunks[:top_k]

        for rank, (chunk, score) in enumerate(top_chunks, start=1):
            results.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.chunk_metadata or {},
                    "similarity_score": round(score, 4),
                    "rank": rank,
                }
            )

    return results
