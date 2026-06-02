# SEC 10-K Document Analysis API

A RAG-powered (Retrieval-Augmented Generation) system for analysing public SEC 10-K annual filings. Ask natural-language questions and get grounded, cited answers drawn directly from the source documents.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd sec-10k-rag

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install fastapi uvicorn httpx beautifulsoup4 chromadb \
            sentence-transformers anthropic python-dotenv

# 4. Add your Anthropic API key
cp .env.example .env
# Open .env and replace the placeholder with your real key

# 5. Start the server
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** for the chat UI, or **http://127.0.0.1:8000/docs** for the interactive API explorer.

---

## Architecture Overview

```
SEC .htm URL
    │
    ▼  ingestion.py       Strip HTML tags, extract 4 major 10-K sections
    ▼  chunker.py         Split each section into overlapping 500-word chunks
    ▼  embedding.py       Embed chunks with a local model, store in ChromaDB
    ────────────── ingestion complete ──────────────
    ▼  (question arrives)
    ▼  embedding.py       Embed question, find top-5 closest chunks
    ▼  rag.py             Build grounded prompt, call Claude, return answer + citations
    ▼  main.py            FastAPI web layer — HTTP endpoints + chat UI
```

The system has two phases:

- **Offline (ingestion):** A 10-K filing is downloaded, parsed into its major sections, chunked, and stored as vectors in a local ChromaDB database. This happens once per document.
- **Online (query):** A question is embedded and compared against all stored chunks. The most relevant passages are retrieved and sent to Claude with strict instructions to answer only from the provided text.

---

## File Structure

```
.
├── ingestion.py        HTML parsing and section extraction
├── chunker.py          Text chunking with sliding window overlap
├── embedding.py        Vector storage and retrieval (ChromaDB)
├── rag.py              RAG pipeline — retrieval + Claude generation
├── main.py             FastAPI app — all HTTP endpoints
├── static/
│   └── index.html      Chat UI served at /
├── .env.example        API key template
├── .gitignore
└── README.md
```

---

## Chunking Approach

Each extracted section is split into **500-word sliding windows with a 50-word overlap**.

**Why 500 words?** Large enough to contain a complete idea or argument, small enough that the resulting vector captures a specific topic rather than a broad average across many topics.

**Why overlap?** If a key sentence falls at the boundary between two chunks, a hard split would lose its context. The 50-word overlap ensures no important idea is orphaned at a chunk edge.

Each chunk is stored with metadata: `doc_id`, `company`, `section`, and a unique `id`. This metadata powers both citations ("this came from Apple's Risk Factors section") and optional filters when querying.

---

## Retrieval Approach

Retrieval is **dense vector search** using cosine similarity:

1. Every chunk is embedded at ingestion time using `all-MiniLM-L6-v2` (SentenceTransformers), producing a 384-dimensional vector.
2. At query time, the question is embedded with the same model.
3. ChromaDB finds the top-5 chunks whose vectors are closest to the question vector (lowest cosine distance).
4. Those chunks are formatted as numbered `[SOURCE N]` blocks and injected into the Claude prompt.

The embedding model runs **entirely locally** — no API key, no cost, no network latency. ChromaDB persists vectors to disk in `chroma_db/` so they survive server restarts without re-embedding.

---

## API Endpoints

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `POST` | `/documents/ingest` | 201 | Ingest a 10-K filing from a SEC EDGAR URL |
| `GET` | `/documents` | 200 | List all ingested documents |
| `GET` | `/documents/{id}/sections` | 200 | Get extracted sections for a document |
| `POST` | `/questions/ask` | 200 | Ask a question — returns answer + citations |
| `POST` | `/analysis-jobs` | 202 | Submit a batch of questions asynchronously |
| `GET` | `/analysis-jobs/{id}` | 200 | Poll async job status and results |
| `GET` | `/health` | 200 | Liveness check |
| `GET` | `/` | 200 | Chat UI |

### Sample Requests

**Ingest a filing:**
```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
    "company_name": "Apple",
    "doc_id": "aapl-2025"
  }'
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/questions/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top business risks described by the company?", "top_k": 5}'
```

**Sample response:**
```json
{
  "question": "What are the top business risks described by the company?",
  "answer": "Apple identifies several key business risks [SOURCE 1]. The company faces intense competition across all product categories and geographic markets [SOURCE 2]. Additional risks include dependence on key personnel, supply chain concentration in Asia, and exposure to global macroeconomic conditions [SOURCE 3].\n\nCitations:\nSource 1 → Apple | Risk Factors\nSource 2 → Apple | Risk Factors\nSource 3 → Apple | Risk Factors",
  "sources": [
    {
      "source_index": 1,
      "company": "Apple",
      "section": "Risk Factors",
      "doc_id": "aapl-2025",
      "excerpt": "The following summarizes factors that could have a material adverse effect on the Company's business...",
      "relevance_distance": 0.4864
    }
  ],
  "truncated": false,
  "error": null
}
```

---

## Tradeoffs Made

**Local embeddings over API embeddings.** `all-MiniLM-L6-v2` runs on-device at no cost with no rate limits. For a retrieval task over structured financial text, the quality gap versus frontier embedding models (e.g. OpenAI `text-embedding-ada-002`) is negligible.

**ChromaDB over a managed vector database.** Zero setup, Python-native, persists to a local folder. The right choice for 2–3 documents and thousands of chunks. Would not scale horizontally — swap for Pinecone or pgvector at production scale.

**Fixed-size word chunking over semantic chunking.** Simple, predictable, and document-structure-agnostic. The downside is occasional mid-sentence splits; the overlap mitigates this. A recursive text splitter that respects paragraph and sentence boundaries would improve precision.

**In-memory document registry with JSON persistence.** `DOCUMENTS` is a Python dict that loads from `documents_registry.json` on startup. Fast and dependency-free, but not thread-safe for concurrent writes. A proper database (SQLite minimum, PostgreSQL for multi-server) is the production path.

**`top_k = 5` for retrieval.** 500 words × 5 chunks ≈ 2,500 words of context per question — enough for nuanced answers without bloating the Claude prompt. Cross-company comparison questions benefit from a higher `top_k` (8–10).

---

## Assumptions

- Input documents are SEC EDGAR 10-K filings in `.htm` format following the standard Item numbering convention (Item 1, 1A, 7, 8).
- The Anthropic API key is valid and has sufficient credits. The system degrades gracefully without one — retrieval still works, generation is skipped.
- The server runs as a single process. The in-memory document registry and ChromaDB are not safe for concurrent multi-process deployments without additional coordination.
- Section extraction assumes filings follow standard SEC formatting. Atypical filings (amended, foreign private issuers) may extract fewer sections.

---

## Limitations

- **Section extraction is regex-based.** It works reliably on standard 10-K filings but can miss sections in non-standard formats or when the document uses unusual heading styles.
- **No re-ingestion support.** Submitting the same `doc_id` twice returns a 409 Conflict. Delete the `chroma_db/` folder and `documents_registry.json` to start fresh.
- **Retrieval is semantic only.** There is no keyword (BM25) search. Exact-match queries on specific figures or ticker symbols may retrieve suboptimal results.
- **No streaming.** Answers are returned all at once after Claude finishes generating. Long answers have noticeable latency.
- **Single-process only.** Restarting the server mid-job will lose any in-flight analysis jobs (though document data persists via the registry file).

---

## How to Improve for Production

**Short-term:**
- Add hybrid search (BM25 + vector) for better recall on specific financial figures and terminology
- Add a reranking step — retrieve top-20, rerank with a cross-encoder, pass best 5 to Claude
- Replace JSON registry with SQLite
- Add streaming responses so answers render word-by-word
- Validate that submitted URLs are genuine SEC EDGAR URLs before fetching

**Medium-term:**
- Move to PostgreSQL + pgvector for multi-server deployments
- Add authentication (API keys or OAuth) to all endpoints
- Add rate limiting on the question and ingestion endpoints
- Replace `BackgroundTasks` with a proper job queue (Celery + Redis)

**Long-term:**
- Switch to a managed vector database (Pinecone, Weaviate) for horizontal scaling
- Build an evaluation pipeline — track retrieval recall and answer faithfulness against a held-out question set
- Add document versioning to detect when a new 10-K supersedes an old one
- Add multi-modal extraction to index financial tables and charts

---

## What Worked Well

- The section extraction approach (anchoring on the last occurrence of each Item heading) reliably avoids the Table of Contents trap present in all standard 10-K filings.
- ChromaDB's local persistence means the vector index survives restarts with no re-embedding cost.
- FastAPI's Pydantic validation catches malformed requests before they reach business logic, making error messages clear and consistent.
- The strict system prompt (answer only from provided sources) keeps Claude grounded and makes every answer auditable against the source text.

## What I Would Improve Next

- Smarter chunking: use a recursive splitter that respects paragraph and sentence boundaries instead of a fixed word count.
- Hybrid retrieval: add BM25 alongside dense vectors so exact financial terminology is matched precisely.
- Evaluation harness: define a set of ground-truth question/answer pairs and measure retrieval recall and answer correctness automatically.
- Streaming: pipe Claude's response token-by-token to the UI so long answers don't feel slow.

---

## What I Would Do Differently for a Production-Scale System

- **Storage:** Replace the JSON file and in-memory dict with PostgreSQL for document metadata and pgvector for embeddings. This enables concurrent writes, proper transactions, and horizontal scaling.
- **Job queue:** Replace FastAPI `BackgroundTasks` with Celery + Redis for durable, retryable async jobs with visibility into failures.
- **Observability:** Add structured logging (every request, every retrieval, every Claude call), distributed tracing (OpenTelemetry), and a dashboard for RAG quality metrics (context recall, answer faithfulness, latency percentiles).
- **Security:** Add API key authentication, rate limiting per key, and input sanitisation on document URLs to prevent SSRF attacks.
- **Chunking:** Move to a domain-specific chunking strategy for 10-Ks — split on financial statement boundaries, preserve table structure, and keep related disclosures together rather than splitting on word count.
