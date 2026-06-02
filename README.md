RAG powered AI anlysis system for public SEC 10-K filings. 

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

Open http://127.0.0.1:8000 for the chat UI 

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/ingest` | Ingest a 10-K filing from a SEC EDGAR URL |
| GET | `/documents` | List all ingested documents |
| GET | `/documents/{id}/sections` | Get extracted sections for a document |
| POST | `/questions/ask` | Ask a question — returns answer + citations |


## Architecture

Each filing goes through four stages:

1. Ingestion - BeautifulSoup strips HTML, then four sections (business, risk factors, MD&A, financial statements) are extracted with regex anchors on item hadings. Table of contents is skipped since last occurence of each heading is used.
2. Chunking - Each section is split into 500 word windonws with 50 word overlap to preserve context at boundareis
3. Embedding - chunks are embedded with all-MiniLM-L6-v2 locally, which is stored in ChromaDB with metadata (company/section)
4. Retrieval and generation - At query time question gets embeded, 5 closest chunks retrieved by cosine similarity, and claude generates answer using only retrieved text (no outside knowledge is used)

## Tradeoffs

- local embeddings over api - free, no latency, doesn't have much quality difference for finance-related texts
- using chromaDB over managed vector db - no set up, works fine for this scale
- fixed-size chunking over semantic - more simple and predictable, recursive splitter would be better for paragraph boundaries
- in memory document registry with json file persistence - very logical and predictable; if production-scale then postrgreSQL would be better

## What I would improve for production

- Hybrid search (vector + bm25) which would have better recall on exact financial figures
- reranking step (get top 20, rerank with cross encoder, continue with 5 to claude)
- replace json with proper databse
- streaming responses