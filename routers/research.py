import re
import time
import json
import asyncio
import logging
import queue as queue_mod

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from models.requests import ResearchRequest
from models.responses import ResearchResponse, ErrorResponse
from services.orchestrator import run_research
from services.youtube_transcript import (
    fallback_transcribe,
    fallback_transcribe_with_progress_sync,
)
from utils.text import clean_transcript_text
from config import settings
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["research"])


@router.post(
    "/research",
    response_model=ResearchResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Run deep sales research",
    description=(
        "Accepts a person+company pair and returns synthesized sales prep "
        "material from YouTube videos, articles, and Gemini analysis."
    ),
)
async def research(request: ResearchRequest, user: dict = Depends(get_current_user)) -> ResearchResponse:
    request.user_id = user.get("email")
    start_time = time.time()
    try:
        result = await run_research(request)
        elapsed = time.time() - start_time
        logger.info(
            f"Research completed for {request.target_name} @ "
            f"{request.target_company} in {elapsed:.1f}s"
        )
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Research failed for {request.target_name} @ "
            f"{request.target_company} after {elapsed:.1f}s: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Research pipeline failed",
                "detail": str(e),
                "step_reached": "unknown",
            },
        )


@router.post("/research-stream", summary="Run research with SSE progress")
async def research_stream(request: ResearchRequest, user: dict = Depends(get_current_user)):
    request.user_id = user.get("email")
    progress_q: asyncio.Queue = asyncio.Queue()

    async def on_progress(event_type: str, data: dict):
        await progress_q.put({"type": event_type, **data})

    async def generate():
        start_time = time.time()
        try:
            task = asyncio.create_task(run_research(request, on_progress=on_progress))

            while not task.done():
                try:
                    event = await asyncio.wait_for(progress_q.get(), timeout=0.5)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    pass

            # Drain remaining events
            while not progress_q.empty():
                event = progress_q.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"

            result = task.result()
            elapsed = round(time.time() - start_time, 1)
            yield f"data: {json.dumps({'type': 'result', 'elapsed': elapsed, 'data': result.model_dump()})}\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 1)
            logger.error(f"Research stream failed after {elapsed}s: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong. Please try again.'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Test endpoint: transcription only ----------

class TranscribeRequest(BaseModel):
    youtube_url: str


_YT_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)"
    r"([a-zA-Z0-9_-]{11})"
)


@router.get("/test-transcribe-stream", summary="SSE transcription with progress (temporary)")
async def test_transcribe_stream(youtube_url: str):
    m = _YT_ID_RE.search(youtube_url)
    if not m:
        raise HTTPException(status_code=422, detail="Invalid YouTube URL")
    video_id = m.group(1)

    progress_q: queue_mod.Queue = queue_mod.Queue()

    def on_progress(stage: str, percent: int, detail: str = ""):
        progress_q.put({"stage": stage, "percent": percent, "detail": detail})

    async def generate():
        start = time.time()
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            None, fallback_transcribe_with_progress_sync, video_id, on_progress
        )

        finished = False
        while not finished:
            await asyncio.sleep(0.3)
            while True:
                try:
                    event = progress_q.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event["stage"] in ("done", "error"):
                        finished = True
                except queue_mod.Empty:
                    break

        text = await task
        elapsed = round(time.time() - start, 1)
        if text:
            text = clean_transcript_text(text)
            if len(text) > settings.max_transcript_chars:
                text = text[: settings.max_transcript_chars] + "... [truncated]"

        result = {
            "stage": "result",
            "video_id": video_id,
            "transcript_available": text is not None,
            "transcript_length": len(text) if text else 0,
            "transcript": text,
            "elapsed_seconds": elapsed,
        }
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
