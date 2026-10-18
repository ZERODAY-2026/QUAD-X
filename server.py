"""API for the Private Legal Triage Agent. Run: uvicorn server:app --port 8000"""

import logging
import threading
import uuid

import ollama
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pdf_utils
from legal_analyzer import (
    SINGLE_PASS_TOKENS, ReportError, create_conversation, estimate_tokens, follow_up,
)

logger = logging.getLogger(__name__)
app = FastAPI()
SESSIONS: dict[str, dict] = {}  # replaces st.session_state


def friendly(error: Exception) -> str:
    if isinstance(error, (ReportError, ValueError)):
        return str(error)
    if isinstance(error, ConnectionError):
        return "Couldn't reach Ollama. Make sure it is running (`ollama serve`) and try again."
    if isinstance(error, ollama.ResponseError):
        if getattr(error, "status_code", None) == 404:
            return "The model isn't installed. Pull it with `ollama pull <model>`."
        return f"Ollama reported an error: {getattr(error, 'error', error)}"
    logger.error("Unexpected error", exc_info=error)
    return f"Something went wrong: {error}"


def get_session(sid: str) -> dict:
    if sid not in SESSIONS:
        raise HTTPException(404, "Session expired. Please upload the document again.")
    return SESSIONS[sid]


def run_analysis(sid: str, prompt: str) -> None:
    s = SESSIONS[sid]
    try:
        s["messages"], s["report"] = create_conversation(
            s["text"], prompt,
            progress=lambda done, total: s.update(done=done, total=total),
        )
        s["status"] = "done"
    except Exception as error:
        s["status"], s["error"] = "error", friendly(error)


@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...), prompt: str = Form(...)):
    if not prompt.strip():
        raise HTTPException(400, "Please enter an analysis prompt.")
    pdf = await file.read()
    try:
        pages, skipped, ocr_pages = pdf_utils.extract_pages(pdf)
    except ValueError as error:
        raise HTTPException(400, str(error))
    if not pages:
        raise HTTPException(400, (
            "No text could be read from this PDF, even with OCR." if pdf_utils.tesseract_available()
            else "No selectable text found. It looks scanned, and OCR (Tesseract) isn't installed or on your PATH."
        ))

    text = pdf_utils.build_document_text(pages)
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "status": "running", "done": 0, "total": 1, "text": text, "pdf": pdf,
        "name": file.filename, "messages": None, "report": None, "error": None,
    }
    threading.Thread(target=run_analysis, args=(sid, prompt.strip()), daemon=True).start()
    return {
        "id": sid, "pages": len(pages), "skipped": skipped, "ocr_pages": ocr_pages,
        "long": estimate_tokens(text) > SINGLE_PASS_TOKENS,
    }


@app.get("/api/status/{sid}")
def status(sid: str):
    s = get_session(sid)
    return {k: s[k] for k in ("status", "done", "total", "error", "report", "name")}


class FollowUp(BaseModel):
    prompt: str


@app.post("/api/follow-up/{sid}")
def follow_up_route(sid: str, body: FollowUp):
    s = get_session(sid)
    if not body.prompt.strip():
        raise HTTPException(400, "Please enter a follow-up prompt.")
    try:
        s["messages"], s["report"] = follow_up(
            s["messages"], body.prompt.strip(), document_text=s["text"]
        )
    except Exception as error:
        raise HTTPException(500, friendly(error))
    return {"report": s["report"]}


@app.get("/api/source/{sid}")
def source(sid: str, quote: str, page: int | None = None):
    result = pdf_utils.render_source(get_session(sid)["pdf"], page, quote)
    if not result:
        raise HTTPException(404, "Couldn't locate this passage in the PDF layout.")
    return Response(result[1], media_type="image/png", headers={"X-Page": str(result[0])})


@app.delete("/api/session/{sid}")
def reset(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
