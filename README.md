# Weirwise Core

> Lightweight, reliable, modular open-source RAG infrastructure for developers.

[![License: MIT](https://img.shields.io/badge/License-MIT-6B7078.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-6B7078.svg)](https://www.python.org/downloads/)

Weirwise Core provides a self-hostable, production-ready Retrieval-Augmented Generation (RAG) backend. It handles multi-tenant document ingestion, chunking, vector embeddings, similarity search, and conversation management so you can integrate AI knowledge retrieval into your applications in minutes.

---

## Key Capabilities

- **Strict Multi-Tenant Isolation**: Every document, chunk, assistant, and query is scoped strictly to authenticated tenants.
- **Fast PostgreSQL / pgvector Retrieval**: Native vector similarity operations without external vector databases or in-memory bloat.
- **Asynchronous Ingestion**: Heavy text extraction (PDF, TXT, Markdown) and embedding generation are processed asynchronously with background worker queues.
- **Secure Authentication**: SHA-256 hashed API keys; secrets are never stored in plaintext.
- **Grounded AI Assistants**: Context construction with untrusted data boundaries and precise source attribution.
- **Independent Python SDK**: Clean, typed client for rapid integration into Python backends.

---

## Architecture Overview

```text
[ Developer App / Python SDK ]
              │
              ▼ (HTTP REST / X-API-Key)
    ┌───────────────────┐
    │  Weirwise Server  │  <───> [ PostgreSQL + pgvector ]
    │     (FastAPI)     │                   ▲
    └─────────┬─────────┘                   │
              │ (Queued Jobs)               │
              ▼                             │
    ┌───────────────────┐                   │
    │  Ingestion Worker │───────────────────┘
    └───────────────────┘
```

---

## Quickstart (Local Docker)

### 1. Start Services
```bash
docker compose up -d
```

### 2. Verify Database & Extension Diagnostics
```bash
docker compose exec server python -m server.app.cli check-db
```

### 3. Create Tenant & API Key
```bash
docker compose exec server python -m server.app.cli admin create-tenant --name "My Workspace"
```
Output:
```text
============================================================
 Weirwise Tenant & API Key Created Successfully
============================================================
 Tenant ID : c40d6c8e-5b12-4c22-92fe-5c5cb29135a1
 Tenant Name: My Workspace
 API Key    : ww_kL29c...
============================================================
```

---

## Python SDK Usage

Install the SDK:
```bash
pip install ./packages/python-sdk
```

Run a complete RAG workflow:
```python
from weirwise import Weirwise

client = Weirwise(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY",
)

# 1. Create a Knowledge Base
kb = client.knowledge_bases.create(name="Product Docs")

# 2. Upload Document (PDF, Markdown, TXT)
doc = client.documents.upload(
    knowledge_base_id=kb.id,
    path="./docs/handbook.pdf",
)

# Wait for ingestion to finish processing
client.documents.wait_for_completion(doc.id, timeout=30.0)

# 3. Search Indexed Knowledge (with optional score threshold)
results = client.search(
    knowledge_base_id=kb.id,
    query="How does authentication work?",
    top_k=4,
    score_threshold=0.7,
)
for r in results:
    print(r.similarity_score, r.content)

# 4. Create an Assistant & Chat
assistant = client.assistants.create(
    name="Docs Bot",
    knowledge_base_id=kb.id,
)

response = client.chat(
    assistant_id=assistant.id,
    message="Explain the authentication flow.",
)

print(response.content)
print("Attributed sources:", response.sources)
```

---

## Configuration

Weirwise is configured via environment variables or a `.env` file. See [.env.example](.env.example) for all available options:

| Setting | Default | Description |
|---|---|---|
| `WEIRWISE_DATABASE_URL` | `postgresql+asyncpg://...` | Asynchronous PostgreSQL connection string |
| `WEIRWISE_DATABASE_SYNC_URL` | `postgresql://...` | Sync connection string for Alembic migrations |
| `WEIRWISE_STORAGE_DIR` | `./data/storage` | Local directory for raw uploaded files |
| `WEIRWISE_EMBEDDING_API_KEY` | None | OpenAI API key for embeddings |
| `WEIRWISE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `WEIRWISE_CHAT_API_KEY` | None | OpenAI API key for assistant chat |
| `WEIRWISE_CHAT_MODEL` | `gpt-4o-mini` | Chat model name |

---

## Running Tests

```bash
# Run linting
ruff check .
ruff format --check .

# Run test suite
pytest server/tests packages/python-sdk/tests
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
