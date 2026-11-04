import asyncio
import datetime
import logging
import uuid
from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from server.app.models import IngestionJob, Document, DocumentChunk
from server.app.ingestion import extract_text_from_file, chunk_text
from server.app.providers import EmbeddingProvider, ProviderUnavailableError
from server.app.config import Settings
from server.app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def _heartbeat_loop(
    job_id: uuid.UUID, claim_token: uuid.UUID, settings: Settings, stop_event: asyncio.Event
):
    """Periodically renews the worker lease for long-running jobs."""
    interval = settings.worker_heartbeat_interval
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        try:
            async with AsyncSessionLocal() as session:
                stmt = text(
                    "UPDATE ingestion_jobs "
                    "SET locked_at = NOW() "
                    "WHERE id = :job_id AND claim_token = :token AND status = 'processing'"
                )
                res = await session.execute(stmt, {"job_id": job_id, "token": claim_token})
                await session.commit()
                if res.rowcount == 0:
                    logger.warning(f"Heartbeat: Worker lost ownership of job {job_id}")
                    break
        except Exception as e:
            logger.error(f"Heartbeat failed for job {job_id}: {e}")


async def process_document_job(
    job_id,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    db: Optional[AsyncSession] = None,
    db_session_factory=None,
    worker_id: Optional[str] = None,
    claim_token: Optional[uuid.UUID] = None,
):
    """Executes the ingestion pipeline with single-ownership fencing and atomic chunk swap."""
    close_db = False
    if db is None:
        factory = db_session_factory or AsyncSessionLocal
        db = factory()
        close_db = True

    worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    stop_heartbeat = asyncio.Event()
    heartbeat_task = None

    try:
        # 1. Fetch and atomically claim / lock Job if not already claimed with token
        job_res = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
        job = job_res.scalars().first()
        if not job or job.status == "completed":
            return

        if not claim_token:
            # Inline invocation or initial claim
            claim_token = uuid.uuid4()
            job.status = "processing"
            job.locked_at = datetime.datetime.utcnow()
            job.locked_by = worker_id
            job.claim_token = claim_token
            job.generation = (job.generation or 0) + 1
            job.attempts += 1
            await db.commit()

        # Start heartbeat renewal in background
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(job.id, claim_token, settings, stop_heartbeat)
        )

        # 2. Fetch Document
        doc_res = await db.execute(select(Document).where(Document.id == job.document_id))
        doc = doc_res.scalars().first()
        if not doc:
            job.status = "failed"
            job.error_message = "Document record not found"
            job.locked_at = None
            job.locked_by = None
            await db.commit()
            return

        doc.status = "processing"
        await db.commit()

        # 3. Text extraction & chunking staged entirely in memory before DB modification
        try:
            raw_text = extract_text_from_file(doc.storage_path, doc.mime_type)
        except Exception as e:
            logger.error(f"Failed to extract text from {doc.storage_path}: {e}")
            doc.status = "failed"
            doc.error_message = f"Text extraction failed: {e!s}"
            job.status = "failed"
            job.error_message = doc.error_message
            job.locked_at = None
            job.locked_by = None
            await db.commit()
            return

        chunks = chunk_text(
            raw_text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # 4. Generate embeddings in batches (staged in memory; previous DB chunks remain untouched!)
        all_embeddings = []
        if chunks:
            batch_size = settings.embedding_batch_size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                embeddings = await embedding_provider.get_embeddings(batch)
                all_embeddings.extend(embeddings)

        # 5. ATOMIC CHUNK REPLACEMENT WITH FENCING TOKEN GUARD
        # Verify ownership token is still valid. If another worker stole the lease, rows_affected will be 0.
        check_ownership_stmt = (
            select(IngestionJob)
            .where(
                IngestionJob.id == job.id,
                IngestionJob.claim_token == claim_token,
                IngestionJob.status == "processing",
            )
            .with_for_update()
        )
        owned_job_res = await db.execute(check_ownership_stmt)
        owned_job = owned_job_res.scalars().first()

        if not owned_job:
            # Fencing check failed! Lost ownership to a newer claim token. Abort without committing chunks.
            logger.warning(
                f"Fencing guard rejected commit: Worker {worker_id} lost ownership of job {job.id}."
            )
            await db.rollback()
            return

        # Atomically replace chunks in the same transaction
        # Delete old chunks only now that new ones are completely validated
        await db.execute(
            text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
            {"doc_id": str(doc.id)},
        )

        for idx, (chunk_content, emb) in enumerate(zip(chunks, all_embeddings)):
            chunk_record = DocumentChunk(
                tenant_id=doc.tenant_id,
                knowledge_base_id=doc.knowledge_base_id,
                document_id=doc.id,
                chunk_index=idx,
                content=chunk_content,
                chunk_metadata={"filename": doc.filename, "chunk_index": idx},
                embedding=emb,
            )
            db.add(chunk_record)

        # Update document and job completion within the same transaction
        doc.status = "completed"
        doc.error_message = None
        owned_job.status = "completed"
        owned_job.error_message = None
        owned_job.locked_at = None
        owned_job.locked_by = None

        await db.commit()
        logger.info(
            f"Atomic ingestion completed for doc {doc.id} ({len(chunks)} chunks, token {claim_token})"
        )

    except Exception as e:
        logger.exception(f"Unhandled error during ingestion job {job_id}: {e}")
        await db.rollback()
        should_retry_inline = False
        try:
            job_res = await db.execute(
                select(IngestionJob).where(
                    IngestionJob.id == job_id,
                    IngestionJob.claim_token == claim_token,
                )
            )
            job = job_res.scalars().first()
            if job:
                job.error_message = str(e)
                job.locked_at = None
                job.locked_by = None
                # Only transient provider failures are worth retrying; malformed input
                # (for example binary bytes rejected by the UTF-8 chunk insert) would
                # fail identically on every attempt.
                retryable = isinstance(e, ProviderUnavailableError) and job.attempts < job.max_attempts
                if retryable:
                    job.status = "pending"  # eligible for retry
                    should_retry_inline = settings.ingestion_mode == "inline"
                else:
                    job.status = "failed"
            doc = None
            if job:
                doc_res = await db.execute(select(Document).where(Document.id == job.document_id))
                doc = doc_res.scalars().first()
            if doc and job and job.status == "failed":
                doc.status = "failed"
                doc.error_message = str(e)
            await db.commit()
        except Exception:
            pass
        if should_retry_inline:
            # Inline mode has no polling worker, so a job left pending would never
            # be claimed again. Re-dispatch it here to reach a terminal state.
            asyncio.create_task(_retry_inline_job(job_id, settings, embedding_provider))
    finally:
        stop_heartbeat.set()
        if heartbeat_task:
            heartbeat_task.cancel()
        if close_db:
            await db.close()


async def _retry_inline_job(job_id, settings: Settings, embedding_provider: EmbeddingProvider, delay: float = 2.0):
    """Re-runs a transiently failed inline ingestion after a backoff."""
    await asyncio.sleep(delay)
    try:
        await process_document_job(job_id, settings, embedding_provider)
    except Exception as e:
        logger.exception(f"Inline retry for job {job_id} failed: {e}")


async def reclaim_stale_jobs(settings: Settings, db: AsyncSession):
    """Reclaims stale processing jobs where worker heartbeat expired."""
    lease_seconds = settings.worker_lease_seconds
    threshold = datetime.datetime.utcnow() - datetime.timedelta(seconds=lease_seconds)

    # Find jobs in processing past lease duration
    stmt = (
        select(IngestionJob)
        .where(
            IngestionJob.status == "processing",
            IngestionJob.locked_at < threshold,
        )
        .with_for_update(skip_locked=True)
    )
    res = await db.execute(stmt)
    stale_jobs = res.scalars().all()

    for job in stale_jobs:
        logger.warning(
            f"Reclaiming stale job {job.id} (locked by {job.locked_by} at {job.locked_at})"
        )
        job.locked_at = None
        job.locked_by = None
        job.claim_token = None
        if job.attempts >= job.max_attempts:
            job.status = "failed"
            job.error_message = "Job exceeded maximum attempts after lease expiration."
        else:
            job.status = "pending"  # return to pool
    if stale_jobs:
        await db.commit()


async def run_worker_loop():
    """Autonomous background worker polling with SKIP LOCKED and lease reclamation."""
    from server.app.config import get_settings
    from server.app.providers import OpenAIEmbeddingProvider

    settings = get_settings()
    provider = OpenAIEmbeddingProvider(settings)
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting Weirwise worker {worker_id}...")

    while True:
        try:
            async with AsyncSessionLocal() as session:
                # 1. Periodically reclaim stale abandoned jobs
                await reclaim_stale_jobs(settings, session)

                # 2. Atomically claim next pending job
                stmt = (
                    select(IngestionJob)
                    .where(IngestionJob.status == "pending")
                    .order_by(IngestionJob.created_at.asc())
                    .limit(1)
                )
                if session.bind and session.bind.dialect.name == "postgresql":
                    stmt = stmt.with_for_update(skip_locked=True)

                res = await session.execute(stmt)
                job = res.scalars().first()

                if job:
                    job_id = job.id
                    claim_token = uuid.uuid4()
                    job.status = "processing"
                    job.locked_at = datetime.datetime.utcnow()
                    job.locked_by = worker_id
                    job.claim_token = claim_token
                    job.generation = (job.generation or 0) + 1
                    job.attempts += 1
                    await session.commit()

                    logger.info(f"Worker {worker_id} claimed job {job_id} with token {claim_token}")
                    await process_document_job(
                        job_id=job_id,
                        settings=settings,
                        embedding_provider=provider,
                        db=session,
                        worker_id=worker_id,
                        claim_token=claim_token,
                    )
                else:
                    await asyncio.sleep(settings.worker_poll_interval)
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            await asyncio.sleep(settings.worker_poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker_loop())
