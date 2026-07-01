FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_CACHE_DIR=/app/.cache/uv

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY src ./src

RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]