import logging
import uuid
import time
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from server.app.config import get_settings
from server.app.database import get_db, engine, Base
from server.app.api import router as v1_router
from server.app.schemas import HealthResponse

logger = logging.getLogger(__name__)
settings = get_settings()

_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.\:\+]{1,128}$")


def _sanitize_request_id(req_id: str | None) -> str:
    """Validate and sanitize incoming correlation ID to prevent header injection or unbounded lengths."""
    if req_id and _REQUEST_ID_REGEX.match(req_id):
        return req_id
    return str(uuid.uuid4())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage directory exists
    import os

    os.makedirs(settings.storage_dir, exist_ok=True)
    # Setup tables if needed
    async with engine.begin() as conn:
        # If postgresql, ensure pgvector extension exists
        if conn.dialect.name == "postgresql":
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as e:
                logger.warning(f"Could not automatically create extension 'vector': {e}")
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


tags_metadata = [
    {
        "name": "Health & Diagnostics",
        "description": "Liveness and readiness probes for service orchestration and database monitoring.",
    },
    {
        "name": "Knowledge Bases",
        "description": "Manage multi-tenant isolated knowledge bases for document partitioning.",
    },
    {
        "name": "Documents & Ingestion",
        "description": "Upload, deduplicate, chunk, and index documents using vector embeddings.",
    },
    {
        "name": "Retrieval & Search",
        "description": "Perform semantic similarity search with cosine distance, threshold filtering, and metadata constraints.",
    },
    {
        "name": "Assistants & Chat",
        "description": "Configure AI assistants grounded in knowledge bases and execute conversational chat.",
    },
]

app = FastAPI(
    title="Weirwise Core API",
    description="""
**Weirwise Core** is an open-source, lightweight, enterprise-grade RAG (Retrieval-Augmented Generation) infrastructure engine.

### Key Capabilities:
- **Tenant Isolation**: Multi-tenant database partitioning secured by SHA-256 hashed API keys.
- **Asynchronous Ingestion**: Automatic chunking, deduplication, and atomic zero-downtime vector replacement.
- **pgvector & Fallback**: High-performance semantic vector search with cosine similarity and metadata filtering.
- **Assistant Grounding**: Grounded question-answering with citation attribution and source verification.

### Authentication:
Supply your API key in the `X-API-Key` HTTP request header:
```http
X-API-Key: ww_live_xxxxxxxxxxxxxxxxxxxxxxxx
```
""",
    version=settings.app_version,
    openapi_tags=tags_metadata,
    contact={
        "name": "Weirwise Open-Source Contributors",
        "url": "https://github.com/ansarialireza/weirwise-rag-core",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    # Core authenticates with an X-API-Key header, not cookies. Keeping
    # credentials off prevents a wildcard origin from exposing tenant data
    # to arbitrary browsers.
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    # Validate and sanitize incoming X-Request-ID or generate new UUID
    req_id = _sanitize_request_id(request.headers.get("X-Request-ID"))
    start_time = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    finally:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        if response is not None:
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
        else:
            logger.error(f"Request {req_id} failed after {duration_ms} ms")
    return response


@app.get(
    "/health/live",
    tags=["Health & Diagnostics"],
    summary="Liveness Probe",
    description="Check whether the application process is running and responding to HTTP requests.",
    response_model=HealthResponse,
    operation_id="get_health_live",
)
async def health_live():
    return HealthResponse(status="alive", database="unverified", version=settings.app_version)


@app.get(
    "/health/ready",
    tags=["Health & Diagnostics"],
    summary="Readiness Probe",
    description="Check whether the application is fully initialized and can successfully execute queries against the primary database.",
    response_model=HealthResponse,
    operation_id="get_health_ready",
)
async def health_ready(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Readiness check database failure: {e}")
        db_status = f"unhealthy: {str(e)}"
    return HealthResponse(
        status="ready" if db_status == "healthy" else "degraded",
        database=db_status,
        version=settings.app_version,
    )


app.include_router(v1_router, prefix="/api")
