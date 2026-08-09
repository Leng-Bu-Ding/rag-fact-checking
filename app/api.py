from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.services.rag import RAGService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

_rag_service: RAGService | None = None
_rag_service_lock = threading.Lock()


def get_rag_service() -> RAGService:
    """Load the model stack exactly once, including under concurrent requests."""
    global _rag_service
    if _rag_service is None:
        with _rag_service_lock:
            if _rag_service is None:
                _rag_service = RAGService(PROJECT_ROOT)
    return _rag_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Uvicorn is not ready until the indexes and all local models can load.
    await asyncio.to_thread(get_rag_service)
    yield


app = FastAPI(title="Trustworthy RAG API", version="0.5.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, object]:
    return get_rag_service().status


@app.get("/api/samples")
def samples(
    limit: int = Query(default=4, ge=1, le=20),
) -> dict[str, object]:
    return {"samples": get_rag_service().samples[:limit]}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, object]:
    try:
        return get_rag_service().ask(request.question, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
