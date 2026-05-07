"""
Pinecone vector store operations.
Handles index creation, upserting, and querying with hybrid search.
"""

import logging
import hashlib
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons
_pinecone_client: Optional[Pinecone] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_keyword_db_lock = Lock()


def _get_keyword_db_path() -> Path:
    settings = get_settings()
    db_path = Path(settings.KEYWORD_INDEX_SQLITE_PATH)
    if not db_path.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        db_path = backend_root / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _ensure_keyword_index() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_keyword_db_path(), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            video_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            video_title TEXT,
            channel TEXT,
            start_time REAL,
            end_time REAL,
            start_display TEXT,
            end_display TEXT,
            PRIMARY KEY (video_id, chunk_index)
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            video_id UNINDEXED,
            chunk_index UNINDEXED,
            text
        )
    """)
    conn.commit()
    return conn


def get_pinecone_client() -> Pinecone:
    """Get or create Pinecone client singleton."""
    global _pinecone_client
    if _pinecone_client is None:
        settings = get_settings()
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pinecone_client


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create embeddings model singleton."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
    return _embeddings


def ensure_index_exists() -> None:
    """Create Pinecone index if it doesn't exist."""
    settings = get_settings()
    pc = get_pinecone_client()

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if settings.PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSIONS,
            metric="dotproduct",  # Required for hybrid search
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )
        logger.info(f"Index {settings.PINECONE_INDEX_NAME} created successfully")
    else:
        logger.info(f"Index {settings.PINECONE_INDEX_NAME} already exists")


def get_index():
    """Get the Pinecone index."""
    settings = get_settings()
    pc = get_pinecone_client()
    return pc.Index(settings.PINECONE_INDEX_NAME)


def generate_vector_id(video_id: str, chunk_index: int) -> str:
    """Generate a deterministic vector ID for a chunk."""
    raw = f"{video_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


async def upsert_documents(
    documents: list[Document],
    video_id: str,
) -> int:
    """
    Embed and upsert documents into Pinecone.
    Uses the video_id as the namespace for isolation.

    Returns the number of vectors upserted.
    """
    if not documents:
        return 0

    ensure_index_exists()
    index = get_index()
    embeddings = get_embeddings()

    # Delete existing vectors for this video (re-index support)
    try:
        index.delete(namespace=video_id, delete_all=True)
        logger.info(f"Cleared existing vectors for namespace: {video_id}")
    except Exception as e:
        logger.debug(f"No existing vectors to clear for {video_id}: {e}")

    # Reset keyword index namespace for this video
    with _keyword_db_lock:
        conn = _ensure_keyword_index()
        conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM chunks_fts WHERE video_id = ?", (video_id,))
        conn.commit()
        conn.close()

    # Batch embed and upsert
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]

        # Generate dense embeddings
        dense_vectors = embeddings.embed_documents(texts)

        # Prepare upsert data
        vectors = []
        keyword_rows = []
        for j, (doc, dense_vec) in enumerate(zip(batch, dense_vectors)):
            vec_id = generate_vector_id(video_id, i + j)

            # Build metadata (Pinecone has metadata size limits)
            metadata = {
                "text": doc.page_content[:1000],  # Truncate for metadata storage
                "video_id": doc.metadata.get("video_id", video_id),
                "video_title": doc.metadata.get("video_title", "")[:200],
                "channel": doc.metadata.get("channel", "")[:100],
                "start_time": doc.metadata.get("start_time", 0.0),
                "end_time": doc.metadata.get("end_time", 0.0),
                "start_display": doc.metadata.get("start_display", "00:00"),
                "end_display": doc.metadata.get("end_display", "00:00"),
                "chunk_index": doc.metadata.get("chunk_index", i + j),
            }

            vectors.append({
                "id": vec_id,
                "values": dense_vec,
                "metadata": metadata,
            })
            keyword_rows.append(metadata)

        index.upsert(vectors=vectors, namespace=video_id)
        with _keyword_db_lock:
            conn = _ensure_keyword_index()
            for metadata in keyword_rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        video_id, chunk_index, text, video_title, channel,
                        start_time, end_time, start_display, end_display
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata["video_id"],
                        metadata["chunk_index"],
                        metadata["text"],
                        metadata["video_title"],
                        metadata["channel"],
                        metadata["start_time"],
                        metadata["end_time"],
                        metadata["start_display"],
                        metadata["end_display"],
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunks_fts (video_id, chunk_index, text)
                    VALUES (?, ?, ?)
                    """,
                    (metadata["video_id"], metadata["chunk_index"], metadata["text"]),
                )
            conn.commit()
            conn.close()
        total_upserted += len(vectors)

    logger.info(f"Upserted {total_upserted} vectors for video {video_id}")
    return total_upserted


async def query_vectors(
    query: str,
    video_id: str,
    top_k: int = 20,
) -> list[Document]:
    """
    Query Pinecone for similar documents.
    Returns LangChain Documents with metadata.
    """
    embeddings = get_embeddings()
    index = get_index()

    # Generate query embedding
    query_vector = embeddings.embed_query(query)

    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=video_id,
        include_metadata=True,
    )

    # Convert to LangChain Documents
    documents = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        doc = Document(
            page_content=metadata.get("text", ""),
            metadata={
                "score": match.get("score", 0.0),
                "video_id": metadata.get("video_id", video_id),
                "video_title": metadata.get("video_title", ""),
                "channel": metadata.get("channel", ""),
                "start_time": metadata.get("start_time", 0.0),
                "end_time": metadata.get("end_time", 0.0),
                "start_display": metadata.get("start_display", "00:00"),
                "end_display": metadata.get("end_display", "00:00"),
                "chunk_index": metadata.get("chunk_index", 0),
            },
        )
        documents.append(doc)

    return documents


async def query_keyword_documents(
    query: str,
    video_id: str,
    top_k: int = 20,
) -> list[Document]:
    """Query local SQLite FTS keyword index for sparse retrieval."""
    try:
        with _keyword_db_lock:
            conn = _ensure_keyword_index()
            rows = conn.execute(
                """
                SELECT
                    c.text,
                    c.video_id,
                    c.video_title,
                    c.channel,
                    c.start_time,
                    c.end_time,
                    c.start_display,
                    c.end_display,
                    c.chunk_index,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c
                    ON c.video_id = chunks_fts.video_id
                    AND c.chunk_index = chunks_fts.chunk_index
                WHERE chunks_fts MATCH ? AND chunks_fts.video_id = ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (query, video_id, top_k),
            ).fetchall()
            conn.close()
    except sqlite3.OperationalError:
        # Invalid FTS query terms should degrade gracefully.
        return []

    docs: list[Document] = []
    for row in rows:
        bm25_score = row[9] if row[9] is not None else 10.0
        sparse_score = 1.0 / (1.0 + max(0.0, bm25_score))
        docs.append(
            Document(
                page_content=row[0],
                metadata={
                    "score": sparse_score,
                    "sparse_score": sparse_score,
                    "retrieval_source": "sparse",
                    "video_id": row[1],
                    "video_title": row[2] or "",
                    "channel": row[3] or "",
                    "start_time": row[4] or 0.0,
                    "end_time": row[5] or 0.0,
                    "start_display": row[6] or "00:00",
                    "end_display": row[7] or "00:00",
                    "chunk_index": row[8] or 0,
                },
            )
        )
    return docs


def check_video_indexed(video_id: str) -> dict:
    """
    Check if a video has been indexed in Pinecone.
    Returns stats about the namespace.
    """
    try:
        index = get_index()
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if video_id in namespaces:
            count = namespaces[video_id].get("vector_count", 0)
            return {"indexed": True, "chunk_count": count}
        return {"indexed": False, "chunk_count": 0}
    except Exception as e:
        logger.error(f"Error checking index status for {video_id}: {e}")
        return {"indexed": False, "chunk_count": 0}
