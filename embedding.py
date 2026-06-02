import chromadb
from chromadb.utils import embedding_functions

_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_COLLECTION_NAME = "sec_10k_chunks"


def get_collection(persist_dir: str = "./chroma_db") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(chunks: list[dict], persist_dir: str = "./chroma_db") -> int:
    collection = get_collection(persist_dir)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "company": c["company"],
            "section": c["section"],
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query_chunks(
    question: str,
    top_k: int = 5,
    doc_id: str | None = None,
    company: str | None = None,
    section: str | None = None,
    persist_dir: str = "./chroma_db",
) -> list[dict]:
    collection = get_collection(persist_dir)

    filters = []
    if doc_id:
        filters.append({"doc_id": {"$eq": doc_id}})
    if company:
        filters.append({"company": {"$eq": company}})
    if section:
        filters.append({"section": {"$eq": section}})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": text,
                "doc_id": meta["doc_id"],
                "company": meta["company"],
                "section": meta["section"],
                "distance": round(dist, 4),
            }
        )

    return chunks


def delete_doc_chunks(doc_id: str, persist_dir: str = "./chroma_db") -> None:
    collection = get_collection(persist_dir)
    collection.delete(where={"doc_id": {"$eq": doc_id}})


def list_doc_ids(persist_dir: str = "./chroma_db") -> list[str]:
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    seen = set()
    doc_ids = []
    for meta in results["metadatas"]:
        if meta["doc_id"] not in seen:
            seen.add(meta["doc_id"])
            doc_ids.append(meta["doc_id"])
    return doc_ids
