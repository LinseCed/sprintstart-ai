# Developer Setup

## Prerequisites

Before you can run Codebridge locally you need the following tools installed:

- **Docker Desktop** 4.x or later (used to run Postgres and Redis)
- **Node.js** 20 LTS (frontend)
- **Python** 3.11 or later (backend API)
- **pnpm** 8.x (`npm install -g pnpm`)

## Clone and Bootstrap

```bash
git clone git@github.com:codebridge-io/codebridge.git
cd codebridge
cp .env.example .env          # fill in secrets (see .env.example comments)
docker compose up -d          # starts postgres:14 and redis:7
```

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head          # run all database migrations
uvicorn app.main:app --reload  # starts on http://localhost:8000
```

## Frontend

```bash
cd frontend
pnpm install
pnpm dev                      # starts on http://localhost:5173
```

## Running Tests

```bash
# backend unit + integration tests
cd backend && pytest

# frontend component tests
cd frontend && pnpm test

# end-to-end tests (requires both servers running)
pnpm --filter e2e run test
```

## Common Issues

**Postgres connection refused**: make sure `docker compose up -d` has finished and
the `postgres` container is healthy (`docker ps`).

**Migration errors**: if you switch branches with schema changes, run
`alembic downgrade base && alembic upgrade head` to get a clean state.
