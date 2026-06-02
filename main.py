import os
import uuid
import asyncio
import json
import httpx
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingestion import parse_10k_html
from chunker import chunk_sections
from embedding import store_chunks, list_doc_ids, delete_doc_chunks, get_collection
from rag import answer_question

app = FastAPI(
    title="SEC 10-K Document Analysis API",
    description="RAG-powered Q&A over SEC 10-K filings",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

DOCUMENTS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}

REGISTRY_PATH = Path("./documents_registry.json")


def _save_registry() -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(DOCUMENTS, f)


def _load_registry() -> None:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                data = json.load(f)
            DOCUMENTS.update(data)
        except Exception:
            pass


@app.on_event("startup")
async def startup() -> None:
    _load_registry()


class IngestRequest(BaseModel):
    url: Optional[str] = None
    html: Optional[str] = None
    company_name: str
    doc_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
                "company_name": "Apple",
                "doc_id": "aapl-2025",
            }
        }
    }


class QuestionRequest(BaseModel):
    question: str
    company_filter: Optional[str] = None
    section_filter: Optional[str] = None
    doc_id_filter: Optional[str] = None
    top_k: int = 5

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the top business risks described by the company?",
                "top_k": 5,
            }
        }
    }


class AnalysisJobRequest(BaseModel):
    questions: list[str]
    company_filter: Optional[str] = None
    doc_id_filter: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "questions": [
                    "What are the top business risks described by the company?",
                    "What themes appear repeatedly in management discussion?",
                ]
            }
        }
    }


async def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "FairPlay-Assignment contact@example.com",
        "Accept-Encoding": "gzip, deflate",
    }
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


def _ingest_pipeline(html: str, company_name: str, doc_id: str) -> dict:
    sections = parse_10k_html(html, company_name)

    if not any(sections.values()):
        raise ValueError("No sections could be extracted from the provided document.")

    chunks = chunk_sections(sections, company=company_name, doc_id=doc_id)
    stored = store_chunks(chunks)

    return {
        "doc_id": doc_id,
        "company": company_name,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "sections_found": list(sections.keys()),
        "chunk_count": stored,
        "sections": {k: v[:500] + "..." if len(v) > 500 else v for k, v in sections.items()},
        "_full_sections": sections,
    }


@app.post("/documents/ingest", status_code=201)
async def ingest_document(req: IngestRequest):
    if not req.url and not req.html:
        raise HTTPException(status_code=400, detail="Provide either `url` or `html`.")

    doc_id = req.doc_id or str(uuid.uuid4())

    if doc_id in DOCUMENTS:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{doc_id}' already exists. Use a different doc_id to re-ingest.",
        )

    html = req.html
    if req.url:
        try:
            html = await _fetch_html(req.url)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")

    try:
        doc_record = _ingest_pipeline(html, req.company_name, doc_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    DOCUMENTS[doc_id] = doc_record
    _save_registry()

    return {k: v for k, v in doc_record.items() if k != "_full_sections"}


@app.get("/documents")
def list_documents():
    return {
        "count": len(DOCUMENTS),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "company": d["company"],
                "ingested_at": d["ingested_at"],
                "sections_found": d["sections_found"],
                "chunk_count": d["chunk_count"],
            }
            for d in DOCUMENTS.values()
        ],
    }


@app.get("/documents/{doc_id}/sections")
def get_sections(doc_id: str, preview_chars: int = 500):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    full_sections = DOCUMENTS[doc_id]["_full_sections"]

    sections_out = {}
    for name, text in full_sections.items():
        if preview_chars and len(text) > preview_chars:
            sections_out[name] = {
                "preview": text[:preview_chars] + "...",
                "total_chars": len(text),
                "truncated": True,
            }
        else:
            sections_out[name] = {
                "preview": text,
                "total_chars": len(text),
                "truncated": False,
            }

    return {
        "doc_id": doc_id,
        "company": DOCUMENTS[doc_id]["company"],
        "sections": sections_out,
    }


@app.post("/questions/ask")
def ask_question(req: QuestionRequest):
    if not DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. POST to /documents/ingest first.",
        )

    result = answer_question(
        question=req.question,
        top_k=req.top_k,
        company_filter=req.company_filter,
        section_filter=req.section_filter,
        doc_id_filter=req.doc_id_filter,
    )
    return result


async def _run_analysis_job(job_id: str, req: AnalysisJobRequest):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    results = []
    try:
        for question in req.questions:
            result = answer_question(
                question=question,
                company_filter=req.company_filter,
                doc_id_filter=req.doc_id_filter,
            )
            results.append(result)
            await asyncio.sleep(0)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        JOBS[job_id]["results"] = results

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@app.post("/analysis-jobs", status_code=202)
async def create_analysis_job(req: AnalysisJobRequest, background_tasks: BackgroundTasks):
    if not DOCUMENTS:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")
    if not req.questions:
        raise HTTPException(status_code=400, detail="Provide at least one question.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "question_count": len(req.questions),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "results": None,
        "error": None,
    }

    background_tasks.add_task(_run_analysis_job, job_id, req)
    return {"job_id": job_id, "status": "queued", "question_count": len(req.questions)}


@app.get("/analysis-jobs/{job_id}")
def get_analysis_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOBS[job_id]


@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len(DOCUMENTS)}
