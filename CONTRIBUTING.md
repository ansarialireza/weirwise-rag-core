# Contributing to Weirwise Core

Thank you for your interest in contributing to Weirwise Core!

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ansarialireza/weirwise-rag-core.git
   cd weirwise-rag-core
   ```

2. **Create Python virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   pip install -e packages/python-sdk
   ```

4. **Run PostgreSQL with pgvector**:
   ```bash
   docker compose up -d db
   alembic -c server/alembic.ini upgrade head
   ```

## Running Tests

Run linting and test suites:
```bash
ruff check .
ruff format --check .
pytest server/tests packages/python-sdk/tests
```

## Submitting Pull Requests
- Keep PRs focused on a single responsibility.
- Include unit and integration tests for new functionality.
- Ensure all CI tests and ruff lint checks pass cleanly.
