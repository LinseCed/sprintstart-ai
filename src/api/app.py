import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

from api.dependencies import get_llm
from api.routes import chat, health, ingest
from llm.errors import LLMUnavailableError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        get_llm().embed("ping")
    except LLMUnavailableError as exc:
        logger.warning("LLM backend unreachable at startup: %s", exc)
    yield


app = FastAPI(title="sprintstart-ai", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(ingest.router)

app.include_router(api_router)
