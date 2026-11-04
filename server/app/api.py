import os
import uuid
import hashlib
import codecs
from typing import List
from uuid import UUID
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from server.app.database import get_db, AsyncSessionLocal
from server.app.dependencies import get_current_tenant
from server.app.config import get_settings, Settings
from server.app.models import (
    Tenant,
    KnowledgeBase,
    Document,
    IngestionJob,
    Assistant,
    Conversation,
    Message,
)
from server.app.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    DocumentResponse,
    DocumentUploadResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    AssistantCreate,
    AssistantResponse,
    MessageResponse,
    ChatRequest,
    ChatResponse,
)
from server.app.retrieval import retrieve_relevant_chunks
from server.app.providers import (
    OpenAIEmbeddingProvider,
    OpenAIChatProvider,
    ProviderUnavailableError,
)
from server.app.worker import process_document_job

router = APIRouter(prefix="/v1")

_TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown"}
_UPLOAD_EXTENSIONS = _TEXT_EXTENSIONS | {".pdf", ".docx"}


async def _spool_upload(file: UploadFile, settings: Settings) -> tuple[str, str, int, str]:
    """Stream an upload to a temp file with a hard size cap; return (path, ext, size, sha256).

    Reading the whole body into memory before checking its size let one request
    hold up to the full 50MB limit (and more) in RAM.
    """
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{ext or 'none'}'. "
                f"Supported: {', '.join(sorted(_UPLOAD_EXTENSIONS))}"
            ),
        )
    kind = "pdf" if ext == ".pdf" else "docx" if ext == ".docx" else "text"

    storage_dir = settings.storage_dir
    os.makedirs(storage_dir, exist_ok=True)
    part_path = os.path.join(storage_dir, f".upload-{uuid.uuid4().hex}.part")

    hasher = hashlib.sha256()
    total = 0
    head = b""
    decoder = codecs.getincrementaldecoder("utf-8")() if kind == "text" else None

    try:
        with open(part_path, "wb") as out:
            while True:
                chunk = await file.read(settings.upload_read_chunk_bytes)
                if not chunk:
                    break
                if not head:
                    head = chunk[:8]
                total += len(chunk)
                if total > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            "File exceeds maximum allowed size of "
                            f"{settings.max_upload_size_bytes} bytes"
                        ),
                    )
                if kind == "text":
                    if b"\x00" in chunk:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="File appears to be binary, not UTF-8 text.",
                        )
                    try:
                        decoder.decode(chunk)
                    except UnicodeDecodeError:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="File is not valid UTF-8 text.",
                        )
                hasher.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())

        if total == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")
        if kind == "pdf" and not head.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File does not have a valid PDF signature.",
            )
        if kind == "docx" and not head.startswith(b"PK\x03\x04"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File does not have a valid DOCX signature.",
            )
        return part_path, ext, total, hasher.hexdigest()
    except BaseException:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except OSError:
            pass
        raise


# --- Knowledge Bases ---
@router.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Knowledge Bases"],
    summary="Create Knowledge Base",
    description="Creates a new isolated partition for storing and indexing related documents.",
    operation_id="create_knowledge_base",
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    kb = KnowledgeBase(
        tenant_id=tenant.id,
        name=payload.name,
        description=payload.description,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


@router.get(
    "/knowledge-bases",
    response_model=List[KnowledgeBaseResponse],
    tags=["Knowledge Bases"],
    summary="List Knowledge Bases",
    description="Retrieves a paginated list of knowledge bases owned by the authenticated tenant.",
    operation_id="list_knowledge_bases",
)
async def list_knowledge_bases(
    limit: int = 50,
    offset: int = 0,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be >= 0")

    stmt = (
        select(KnowledgeBase)
        .where(KnowledgeBase.tenant_id == tenant.id)
        .order_by(KnowledgeBase.created_at.desc(), KnowledgeBase.id.desc())
        .limit(limit)
        .offset(offset)
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@router.get(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseResponse,
    tags=["Knowledge Bases"],
    summary="Get Knowledge Base by ID",
    description="Retrieves detailed metadata for a specific knowledge base owned by the authenticated tenant.",
    operation_id="get_knowledge_base",
)
async def get_knowledge_base(
    kb_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    kb = res.scalars().first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.delete(
    "/knowledge-bases/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Knowledge Bases"],
    summary="Delete Knowledge Base",
    description="Permanently deletes a knowledge base and all associated documents, chunks, and vector embeddings.",
    operation_id="delete_knowledge_base",
)
async def delete_knowledge_base(
    kb_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    kb = res.scalars().first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Remove physical files before the cascade drops the document rows.
    docs_res = await db.execute(select(Document).where(Document.knowledge_base_id == kb.id))
    for doc in docs_res.scalars().all():
        if doc.storage_path and os.path.exists(doc.storage_path):
            try:
                os.remove(doc.storage_path)
            except OSError:
                pass

    await db.delete(kb)
    await db.commit()
    return None


# --- Documents ---
@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Documents & Ingestion"],
    summary="Upload & Ingest Document",
    description="Uploads a file to a designated knowledge base. Automatically performs SHA-256 deduplication and initiates asynchronous chunking and vector embedding generation.",
    operation_id="upload_document",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    knowledge_base_id: UUID = Form(...),
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    # Verify Knowledge Base ownership
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == knowledge_base_id,
        KnowledgeBase.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    kb = res.scalars().first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Stream to disk under a hard cap instead of buffering the whole file in RAM.
    part_path, file_ext, file_size, content_hash = await _spool_upload(file, settings)
    storage_path = None

    try:
        # ATOMIC DEDUPLICATION & CONCURRENCY CONTROL
        # Lock document record by (knowledge_base_id, content_hash) if exists
        existing_doc_stmt = (
            select(Document)
            .where(
                Document.knowledge_base_id == kb.id,
                Document.content_hash == content_hash,
            )
            .with_for_update()
        )
        existing_res = await db.execute(existing_doc_stmt)
        existing_doc = existing_res.scalars().first()

        if existing_doc:
            if existing_doc.status == "completed":
                return DocumentUploadResponse(
                    document=DocumentResponse.model_validate(existing_doc),
                    job_id=None,
                    message="Document already indexed.",
                )
            elif existing_doc.status in ("pending", "processing"):
                return DocumentUploadResponse(
                    document=DocumentResponse.model_validate(existing_doc),
                    job_id=None,
                    message="Document ingestion already in progress.",
                )
            elif existing_doc.status == "failed":
                # Failed document retry: check if an active job already exists
                active_job_stmt = (
                    select(IngestionJob)
                    .where(
                        IngestionJob.document_id == existing_doc.id,
                        IngestionJob.status.in_(["pending", "processing"]),
                    )
                    .with_for_update()
                )
                job_res = await db.execute(active_job_stmt)
                active_job = job_res.scalars().first()

                if active_job:
                    return DocumentUploadResponse(
                        document=DocumentResponse.model_validate(existing_doc),
                        job_id=active_job.id,
                        message="Document ingestion already in progress.",
                    )

                # Atomically reset failed document to pending and enqueue single job
                existing_doc.status = "pending"
                existing_doc.error_message = None
                retry_job = IngestionJob(
                    tenant_id=tenant.id,
                    document_id=existing_doc.id,
                    status="pending",
                )
                db.add(retry_job)
                await db.commit()
                await db.refresh(existing_doc)
                await db.refresh(retry_job)

                if settings.ingestion_mode == "inline":
                    provider = OpenAIEmbeddingProvider(settings)
                    session_factory = getattr(db, "_session_factory", AsyncSessionLocal)
                    background_tasks.add_task(
                        process_document_job,
                        retry_job.id,
                        settings,
                        provider,
                        db_session_factory=session_factory,
                    )

                return DocumentUploadResponse(
                    document=DocumentResponse.model_validate(existing_doc),
                    job_id=retry_job.id,
                    message="Failed document retrying. Ingestion started.",
                )

        # The content is new: promote the spooled file to its durable name.
        os.makedirs(settings.storage_dir, exist_ok=True)
        storage_path = os.path.join(settings.storage_dir, f"{uuid.uuid4()}{file_ext}")
        os.replace(part_path, storage_path)

        doc = Document(
            tenant_id=tenant.id,
            knowledge_base_id=kb.id,
            filename=file.filename or "unknown",
            storage_path=storage_path,
            mime_type=file.content_type or "application/octet-stream",
            file_size=file_size,
            content_hash=content_hash,
            status="pending",
        )
        try:
            async with db.begin_nested():
                db.add(doc)
                await db.flush()
        except Exception as e:
            # Race condition: another concurrent upload inserted the same (knowledge_base_id, content_hash)
            race_res = await db.execute(
                select(Document).where(
                    Document.knowledge_base_id == kb.id,
                    Document.content_hash == content_hash,
                )
            )
            existing_doc = race_res.scalars().first()
            if existing_doc:
                try:
                    os.remove(storage_path)
                except OSError:
                    pass
                storage_path = None
                if existing_doc.status == "completed":
                    return DocumentUploadResponse(
                        document=DocumentResponse.model_validate(existing_doc),
                        job_id=None,
                        message="Document already indexed.",
                    )
                return DocumentUploadResponse(
                    document=DocumentResponse.model_validate(existing_doc),
                    job_id=None,
                    message="Document ingestion already in progress.",
                )
            raise e

    except BaseException:
        if storage_path and os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except OSError:
                pass
        raise
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass

    job = IngestionJob(
        tenant_id=tenant.id,
        document_id=doc.id,
        status="pending",
    )
    db.add(job)
    await db.commit()
    await db.refresh(doc)
    await db.refresh(job)

    # If ingestion_mode is inline, execute background task; otherwise worker handles it via SKIP LOCKED
    if settings.ingestion_mode == "inline":
        provider = OpenAIEmbeddingProvider(settings)

        session_factory = getattr(db, "_session_factory", AsyncSessionLocal)
        background_tasks.add_task(
            process_document_job,
            job.id,
            settings,
            provider,
            db_session_factory=session_factory,
        )

    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(doc),
        job_id=job.id,
        message="Document uploaded successfully. Processing started.",
    )


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    tags=["Documents & Ingestion"],
    summary="Get Document Status",
    description="Retrieves the current ingestion status, metadata, content hash, and error details of a specific document.",
    operation_id="get_document",
)
async def get_document(
    doc_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Document).where(
        Document.id == doc_id,
        Document.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    doc = res.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get(
    "/knowledge-bases/{kb_id}/documents",
    response_model=List[DocumentResponse],
    tags=["Documents & Ingestion"],
    summary="List Documents in Knowledge Base",
    description="Lists documents belonging to a specific knowledge base with pagination support.",
    operation_id="list_documents_in_knowledge_base",
)
async def list_documents(
    kb_id: UUID,
    limit: int = 50,
    offset: int = 0,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be >= 0")

    # Check KB ownership
    kb_res = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.tenant_id == tenant.id,
        )
    )
    if not kb_res.scalars().first():
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    stmt = (
        select(Document)
        .where(
            Document.knowledge_base_id == kb_id,
            Document.tenant_id == tenant.id,
        )
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents & Ingestion"],
    summary="Delete Document",
    description="Deletes a document record, physical file, and all associated embedding chunks from vector storage.",
    operation_id="delete_document",
)
async def delete_document(
    doc_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Document).where(
        Document.id == doc_id,
        Document.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    doc = res.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Clean up physical file if exists
    if os.path.exists(doc.storage_path):
        try:
            os.remove(doc.storage_path)
        except Exception:
            pass

    await db.delete(doc)
    await db.commit()
    return None


# --- Search ---
@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["Retrieval & Search"],
    summary="Semantic Vector Search",
    description="Performs semantic cosine similarity search across indexed document chunks with optional minimum score threshold and metadata filtering.",
    operation_id="search_knowledge",
)
async def search_knowledge(
    payload: SearchRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    # Verify every requested Knowledge Base belongs to this tenant
    kb_ids = payload.knowledge_base_ids or [payload.knowledge_base_id]
    unique_kb_ids = list(dict.fromkeys(kb_ids))
    kb_res = await db.execute(
        select(KnowledgeBase.id).where(
            KnowledgeBase.id.in_(unique_kb_ids),
            KnowledgeBase.tenant_id == tenant.id,
        )
    )
    found_ids = set(kb_res.scalars().all())
    if len(found_ids) != len(unique_kb_ids):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    provider = OpenAIEmbeddingProvider(settings)
    try:
        results = await retrieve_relevant_chunks(
            db=db,
            tenant_id=tenant.id,
            knowledge_base_id=payload.knowledge_base_id,
            knowledge_base_ids=payload.knowledge_base_ids,
            query=payload.query,
            embedding_provider=provider,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            metadata_filters=payload.metadata_filters,
        )
    except ProviderUnavailableError as e:
        # The provider is upstream and transient; callers should retry rather
        # than treat this as a defect in their request.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "5"},
        )

    items = [SearchResultItem(**r) for r in results]
    return SearchResponse(results=items, total=len(items))


# --- Assistants ---
@router.post(
    "/assistants",
    response_model=AssistantResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Assistants & Chat"],
    summary="Create Assistant",
    description="Creates a conversational AI assistant bound to an optional knowledge base for grounded generation.",
    operation_id="create_assistant",
)
async def create_assistant(
    payload: AssistantCreate,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    if payload.knowledge_base_id:
        kb_res = await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == payload.knowledge_base_id,
                KnowledgeBase.tenant_id == tenant.id,
            )
        )
        if not kb_res.scalars().first():
            raise HTTPException(status_code=404, detail="Knowledge base not found")

    assistant = Assistant(
        tenant_id=tenant.id,
        name=payload.name,
        knowledge_base_id=payload.knowledge_base_id,
        system_instructions=payload.system_instructions,
        model=payload.model,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )
    db.add(assistant)
    await db.commit()
    await db.refresh(assistant)
    return assistant


@router.get(
    "/assistants",
    response_model=List[AssistantResponse],
    tags=["Assistants & Chat"],
    summary="List Assistants",
    description="Retrieves a paginated list of AI assistants configured by the authenticated tenant.",
    operation_id="list_assistants",
)
async def list_assistants(
    limit: int = 50,
    offset: int = 0,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be >= 0")

    stmt = (
        select(Assistant)
        .where(Assistant.tenant_id == tenant.id)
        .order_by(Assistant.created_at.desc(), Assistant.id.desc())
        .limit(limit)
        .offset(offset)
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@router.get(
    "/assistants/{assistant_id}",
    response_model=AssistantResponse,
    tags=["Assistants & Chat"],
    summary="Get Assistant by ID",
    description="Retrieves configuration parameters and grounding status of a specific assistant.",
    operation_id="get_assistant",
)
async def get_assistant(
    assistant_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Assistant).where(
        Assistant.id == assistant_id,
        Assistant.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    ast = res.scalars().first()
    if not ast:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return ast


@router.delete(
    "/assistants/{assistant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Assistants & Chat"],
    summary="Delete Assistant",
    description="Permanently deletes an assistant and all associated conversational histories.",
    operation_id="delete_assistant",
)
async def delete_assistant(
    assistant_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Assistant).where(
        Assistant.id == assistant_id,
        Assistant.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    ast = res.scalars().first()
    if not ast:
        raise HTTPException(status_code=404, detail="Assistant not found")
    await db.delete(ast)
    await db.commit()
    return None


# --- Conversations & Chat ---
@router.post(
    "/assistants/{assistant_id}/chat",
    response_model=ChatResponse,
    tags=["Assistants & Chat"],
    summary="Grounded Chat Completion",
    description="Sends a user inquiry to an assistant. Retrieves relevant knowledge chunks, injects context into the prompt, invokes the LLM, and returns the response with source citation metadata.",
    operation_id="chat_with_assistant",
)
async def chat_with_assistant(
    assistant_id: UUID,
    payload: ChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    # 1. Fetch and verify Assistant
    stmt = select(Assistant).where(
        Assistant.id == assistant_id,
        Assistant.tenant_id == tenant.id,
    )
    res = await db.execute(stmt)
    assistant = res.scalars().first()
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # 2. Get or create Conversation
    conversation_id = payload.conversation_id
    if conversation_id:
        conv_res = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == tenant.id,
                Conversation.assistant_id == assistant.id,
            )
        )
        conversation = conv_res.scalars().first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            tenant_id=tenant.id,
            assistant_id=assistant.id,
            title=payload.message[:40],
        )
        db.add(conversation)
        await db.flush()
        conversation_id = conversation.id

    # 3. Retrieve grounding sources from KB if attached
    sources = []
    context_str = ""
    if assistant.knowledge_base_id:
        embed_provider = OpenAIEmbeddingProvider(settings)
        sources = await retrieve_relevant_chunks(
            db=db,
            tenant_id=tenant.id,
            knowledge_base_id=assistant.knowledge_base_id,
            query=payload.message,
            embedding_provider=embed_provider,
            top_k=4,
        )
        if sources:
            context_blocks = [f"[Source chunk {s['chunk_id']}]:\n{s['content']}" for s in sources]
            context_str = "\n\n".join(context_blocks)

    # 4. Construct prompt messages
    sys_instruction = (
        assistant.system_instructions or "You are a helpful and knowledgeable AI assistant."
    )
    messages_payload = [{"role": "system", "content": sys_instruction}]

    if context_str:
        user_prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n"
            f"Given the context above, answer the following question. If the context does not contain enough information to answer, state that clearly.\n\n"
            f"Question: {payload.message}"
        )
    else:
        user_prompt = payload.message

    # Fetch past conversation messages (bounded history: last 6)
    past_msgs_res = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.tenant_id == tenant.id,
        )
        .order_by(Message.created_at.desc())
        .limit(6)
    )
    past_msgs = list(reversed(past_msgs_res.scalars().all()))
    for m in past_msgs:
        messages_payload.append({"role": m.role, "content": m.content})

    messages_payload.append({"role": "user", "content": user_prompt})

    # 5. Generate Response via LLM
    chat_provider = OpenAIChatProvider(settings)
    answer = await chat_provider.generate_response(
        messages=messages_payload,
        temperature=assistant.temperature,
        max_tokens=assistant.max_tokens,
    )

    # 6. Persist Messages
    user_msg = Message(
        tenant_id=tenant.id,
        conversation_id=conversation_id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)

    # Format source references for response
    formatted_sources = [
        {
            "chunk_id": str(s["chunk_id"]),
            "document_id": str(s["document_id"]),
            "similarity_score": s["similarity_score"],
            "metadata": s.get("metadata", {}),
        }
        for s in sources
    ]

    assistant_msg = Message(
        tenant_id=tenant.id,
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=formatted_sources,
    )
    db.add(assistant_msg)
    await db.commit()
    await db.refresh(assistant_msg)

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=assistant_msg.id,
        role="assistant",
        content=answer,
        sources=formatted_sources,
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[MessageResponse],
    tags=["Assistants & Chat"],
    summary="Get Conversation Messages",
    description="Retrieves ordered chronological message history and grounding sources for a specific conversation thread.",
    operation_id="get_conversation_messages",
)
async def get_conversation_messages(
    conversation_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    # Verify ownership
    conv_res = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == tenant.id,
        )
    )
    if not conv_res.scalars().first():
        raise HTTPException(status_code=404, detail="Conversation not found")

    stmt = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.tenant_id == tenant.id,
        )
        .order_by(Message.created_at.asc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()
