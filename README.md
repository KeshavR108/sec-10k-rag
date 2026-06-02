# SEC 10-K Document Analysis API

A RAG-powered Q&A system over public SEC 10-K filings. Ask natural language questions and get answers cited directly from the source documents.

## Stack

- Python / FastAPI
- ChromaDB (vector store)
- SentenceTransformers — `all-MiniLM-L6-v2` (local embeddings)
- Claude (Anthropic) for answer generation
- BeautifulSoup for HTML parsing

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx beautifulsoup4 chromadb sentence-transformers anthropic python-dotenv
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` for the chat UI or `http://127.0.0.1:8000/docs` for the API explorer.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/ingest` | Ingest a 10-K filing from a SEC EDGAR URL |
| GET | `/documents` | List all ingested documents |
| GET | `/documents/{id}/sections` | Get extracted sections for a document |
| POST | `/questions/ask` | Ask a question — returns answer + citations |
| POST | `/analysis-jobs` | Submit a batch of questions asynchronously |
| GET | `/analysis-jobs/{id}` | Poll async job status and results |

## Architecture

Each filing goes through four stages:

1. **Ingestion** — HTML is stripped with BeautifulSoup, then the four major 10-K sections (Business, Risk Factors, MD&A, Financial Statements) are extracted using regex anchors on Item headings. The last occurrence of each heading is used to skip the Table of Contents.
2. **Chunking** — Each section is split into 500-word sliding windows with a 50-word overlap to preserve context at boundaries.
3. **Embedding** — Chunks are embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB with company/section metadata.
4. **Retrieval + Generation** — At query time the question is embedded, the top-5 closest chunks are retrieved by cosine similarity, and Claude generates a grounded answer using only the retrieved text.

## Tradeoffs

- Local embeddings over API embeddings — free, no latency, negligible quality difference for financial text
- ChromaDB over a managed vector DB — zero setup, right for this scale, not horizontally scalable
- Fixed-size chunking over semantic chunking — simple and predictable; a recursive splitter would better respect paragraph boundaries
- In-memory document registry with JSON file persistence — easy to reason about; production would use PostgreSQL

## What I would improve for production

- Hybrid search (BM25 + vector) for better recall on exact financial figures
- A reranking step (retrieve top-20, rerank with a cross-encoder, pass best 5 to Claude)
- Replace JSON registry with a proper database
- Streaming responses
- Authentication and rate limiting on all endpoints
- A proper job queue (Celery + Redis) instead of FastAPI BackgroundTasks
