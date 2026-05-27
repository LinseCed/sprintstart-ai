from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI

load_dotenv()

from api.routes import chat, health, ingest

app = FastAPI(title="sprintstart-ai", version="0.1.0")

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(ingest.router)

app.include_router(api_router)
