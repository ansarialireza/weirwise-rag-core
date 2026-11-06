from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, model_validator


# --- Health ---
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", examples=["ready", "alive"])
    database: str = Field(
        ..., description="Database connectivity status", examples=["healthy", "unverified"]
    )
    version: str = Field(
        default="1.0.0", description="Application semantic version", examples=["1.0.0"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "database": "healthy",
                "version": "1.0.0",
            }
        }
    )


# --- Tenant ---
class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class TenantResponse(BaseModel):
    id: UUID
    name: str
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- API Key ---
class APIKeyCreate(BaseModel):
    name: str = Field(default="default", max_length=100)


class APIKeyCreatedResponse(BaseModel):
    id: UUID
    name: str
    raw_key: str
    tenant_id: UUID
    created_at: datetime


class APIKeyResponse(BaseModel):
    id: UUID
    name: str
    tenant_id: UUID
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Knowledge Base ---
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name of the knowledge base",
        examples=["Engineering Architecture Docs"],
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional description of the knowledge domain or documents indexed",
        examples=["Internal developer documentation and RFCs"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Engineering Architecture Docs",
                "description": "Internal developer documentation and RFCs",
            }
        }
    )


class KnowledgeBaseResponse(BaseModel):
    id: UUID = Field(..., description="Unique knowledge base identifier")
    tenant_id: UUID = Field(..., description="Tenant owner ID")
    name: str = Field(..., description="Knowledge base name")
    description: Optional[str] = Field(None, description="Knowledge base description")
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC update timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "11111111-2222-3333-4444-555555555555",
                "tenant_id": "99999999-8888-7777-6666-555555555555",
                "name": "Engineering Architecture Docs",
                "description": "Internal developer documentation and RFCs",
                "created_at": "2026-10-30T09:00:00Z",
                "updated_at": "2026-10-30T09:00:00Z",
            }
        },
    )


# --- Document ---
class DocumentResponse(BaseModel):
    id: UUID = Field(..., description="Unique document identifier")
    tenant_id: UUID = Field(..., description="Tenant owner ID")
    knowledge_base_id: UUID = Field(..., description="Associated knowledge base ID")
    filename: str = Field(
        ..., description="Original uploaded filename", examples=["architecture_spec.pdf"]
    )
    mime_type: str = Field(..., description="Detected MIME type", examples=["application/pdf"])
    file_size: int = Field(..., description="File size in bytes", examples=[1048576])
    content_hash: str = Field(
        ...,
        description="SHA-256 content hash for deduplication",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    )
    status: str = Field(
        ...,
        description="Processing status: pending, processing, completed, or failed",
        examples=["completed"],
    )
    error_message: Optional[str] = Field(
        None, description="Detailed failure message if status is failed"
    )
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC update timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "22222222-3333-4444-5555-666666666666",
                "tenant_id": "99999999-8888-7777-6666-555555555555",
                "knowledge_base_id": "11111111-2222-3333-4444-555555555555",
                "filename": "architecture_spec.pdf",
                "mime_type": "application/pdf",
                "file_size": 1048576,
                "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "status": "completed",
                "error_message": None,
                "created_at": "2026-10-30T09:01:00Z",
                "updated_at": "2026-10-30T09:02:00Z",
            }
        },
    )


class DocumentUploadResponse(BaseModel):
    document: DocumentResponse
    job_id: Optional[UUID] = Field(
        None,
        description="Asynchronous ingestion job ID if queued",
        examples=["33333333-4444-5555-6666-777777777777"],
    )
    message: str = Field(
        ...,
        description="Human-readable status summary",
        examples=["Document uploaded successfully. Processing started."],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": {
                    "id": "22222222-3333-4444-5555-666666666666",
                    "tenant_id": "99999999-8888-7777-6666-555555555555",
                    "knowledge_base_id": "11111111-2222-3333-4444-555555555555",
                    "filename": "architecture_spec.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1048576,
                    "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "status": "pending",
                    "error_message": None,
                    "created_at": "2026-10-30T09:01:00Z",
                    "updated_at": "2026-10-30T09:01:00Z",
                },
                "job_id": "33333333-4444-5555-6666-777777777777",
                "message": "Document uploaded successfully. Processing started.",
            }
        }
    )


# --- Search & Retrieval ---
class SearchRequest(BaseModel):
    knowledge_base_id: Optional[UUID] = Field(
        None, description="Knowledge base to search against (single-KB shorthand)"
    )
    knowledge_base_ids: Optional[List[UUID]] = Field(
        None,
        max_length=50,
        description="Search several knowledge bases with one query embedding",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language semantic search query",
        examples=["What is the zero-downtime deployment strategy?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve",
        examples=[4],
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold (0.0 to 1.0).",
        examples=[0.65],
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional key-value pairs to filter chunk metadata.",
        examples=[{"environment": "production"}],
    )

    @model_validator(mode="after")
    def validate_targets(self) -> "SearchRequest":
        if not self.knowledge_base_id and not self.knowledge_base_ids:
            raise ValueError("Provide knowledge_base_id or knowledge_base_ids")
        if self.knowledge_base_id and self.knowledge_base_ids:
            raise ValueError("Provide either knowledge_base_id or knowledge_base_ids, not both")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "knowledge_base_id": "11111111-2222-3333-4444-555555555555",
                "query": "What is the zero-downtime deployment strategy?",
                "top_k": 4,
                "score_threshold": 0.65,
                "metadata_filters": {"environment": "production"},
            }
        }
    )


class SearchResultItem(BaseModel):
    chunk_id: UUID = Field(..., description="Unique chunk identifier")
    document_id: UUID = Field(..., description="Source document identifier")
    content: str = Field(..., description="Retrieved textual chunk content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata dictionary associated with chunk"
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity score (0.0 to 1.0)", examples=[0.892]
    )
    rank: int = Field(..., description="Result relevance ranking index starting at 1", examples=[1])


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total: int = Field(
        ..., description="Total count of retrieved chunks matching constraints", examples=[1]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "chunk_id": "44444444-5555-6666-7777-888888888888",
                        "document_id": "22222222-3333-4444-5555-666666666666",
                        "content": "Zero-downtime deployments are executed via blue-green rollover...",
                        "metadata": {"environment": "production", "section": "deployments"},
                        "similarity_score": 0.892,
                        "rank": 1,
                    }
                ],
                "total": 1,
            }
        }
    )


# --- Assistant ---
class AssistantCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the assistant",
        examples=["Architecture Advisor"],
    )
    knowledge_base_id: Optional[UUID] = Field(
        None, description="Optional grounded knowledge base ID"
    )
    system_instructions: Optional[str] = Field(
        None,
        description="System personality and grounding instructions",
        examples=["You are an expert enterprise architecture advisor."],
    )
    model: str = Field(
        default="gpt-4o-mini", description="LLM model identifier", examples=["gpt-4o-mini"]
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 and 2.0",
        examples=[0.2],
    )
    max_tokens: int = Field(
        default=1024, ge=1, le=4096, description="Maximum generation token ceiling", examples=[1024]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Architecture Advisor",
                "knowledge_base_id": "11111111-2222-3333-4444-555555555555",
                "system_instructions": "You are an expert enterprise architecture advisor.",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        }
    )


class AssistantResponse(BaseModel):
    id: UUID = Field(..., description="Assistant identifier")
    tenant_id: UUID = Field(..., description="Tenant owner identifier")
    knowledge_base_id: Optional[UUID] = Field(
        None, description="Associated knowledge base identifier"
    )
    name: str = Field(..., description="Assistant display name")
    system_instructions: Optional[str] = Field(None, description="System instructions")
    model: str = Field(..., description="LLM model identifier")
    temperature: float = Field(..., description="Model temperature")
    max_tokens: int = Field(..., description="Max tokens")
    is_active: bool = Field(..., description="Active operational status")
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC update timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "55555555-6666-7777-8888-999999999999",
                "tenant_id": "99999999-8888-7777-6666-555555555555",
                "knowledge_base_id": "11111111-2222-3333-4444-555555555555",
                "name": "Architecture Advisor",
                "system_instructions": "You are an expert enterprise architecture advisor.",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 1024,
                "is_active": True,
                "created_at": "2026-10-30T09:05:00Z",
                "updated_at": "2026-10-30T09:05:00Z",
            }
        },
    )


# --- Conversation & Chat ---
class ConversationCreate(BaseModel):
    assistant_id: UUID = Field(..., description="Assistant to converse with")
    title: Optional[str] = Field(None, max_length=255, description="Optional conversation title")


class ConversationResponse(BaseModel):
    id: UUID = Field(..., description="Conversation identifier")
    tenant_id: UUID = Field(..., description="Tenant owner identifier")
    assistant_id: UUID = Field(..., description="Assistant identifier")
    title: Optional[str] = Field(None, description="Conversation title")
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC update timestamp")

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    id: UUID = Field(..., description="Message identifier")
    tenant_id: UUID = Field(..., description="Tenant owner identifier")
    conversation_id: UUID = Field(..., description="Parent conversation identifier")
    role: str = Field(
        ..., description="Message role (user, assistant, or system)", examples=["assistant"]
    )
    content: str = Field(..., description="Text content of the message")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="Array of grounding source citations"
    )
    created_at: datetime = Field(..., description="UTC creation timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "77777777-8888-9999-0000-111111111111",
                "tenant_id": "99999999-8888-7777-6666-555555555555",
                "conversation_id": "66666666-7777-8888-9999-000000000000",
                "role": "assistant",
                "content": "Our zero-downtime deployment strategy employs a blue-green rollover...",
                "sources": [
                    {
                        "chunk_id": "44444444-5555-6666-7777-888888888888",
                        "document_id": "22222222-3333-4444-5555-666666666666",
                        "similarity_score": 0.892,
                    }
                ],
                "created_at": "2026-10-30T09:06:00Z",
            }
        },
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User prompt message to send to the grounded assistant",
        examples=["Explain the blue-green deployment rollover process."],
    )
    conversation_id: Optional[UUID] = Field(
        None, description="Optional existing conversation ID to continue a thread"
    )
    stream: bool = Field(
        default=False, description="Whether to stream response tokens (if supported)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Explain the blue-green deployment rollover process.",
                "conversation_id": "66666666-7777-8888-9999-000000000000",
                "stream": False,
            }
        }
    )


class ChatResponse(BaseModel):
    conversation_id: UUID = Field(..., description="Conversation identifier")
    message_id: UUID = Field(..., description="Generated assistant message identifier")
    role: str = Field(default="assistant", description="Message role")
    content: str = Field(
        ..., description="Generated assistant answer grounded on knowledge base sources"
    )
    sources: List[Dict[str, Any]] = Field(
        default=[], description="List of retrieved source citations and similarity scores"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "66666666-7777-8888-9999-000000000000",
                "message_id": "77777777-8888-9999-0000-111111111111",
                "role": "assistant",
                "content": "Based on the internal documentation, zero-downtime deployment relies on blue-green rollover...",
                "sources": [
                    {
                        "chunk_id": "44444444-5555-6666-7777-888888888888",
                        "document_id": "22222222-3333-4444-5555-666666666666",
                        "similarity_score": 0.892,
                    }
                ],
            }
        }
    )


# --- Error Schemas ---
class ErrorDetail(BaseModel):
    code: str = Field(
        ..., description="Standardized application error code", examples=["NOT_FOUND"]
    )
    message: str = Field(
        ..., description="Human-readable error description", examples=["Knowledge base not found"]
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Optional diagnostic details or validation errors"
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "NOT_FOUND",
                    "message": "The requested resource was not found.",
                    "details": {"resource_id": "11111111-2222-3333-4444-555555555555"},
                }
            }
        }
    )
