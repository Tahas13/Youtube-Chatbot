"""
Qdrant vector store operations.
Handles collection creation, upserting, and querying with hybrid search.
"""

import logging
import sqlite3
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons
_qdrant_client: Optional[QdrantClient] = None
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


def _get_collection_name() -> str:
    settings = get_settings()
    return settings.QDRANT_COLLECTION_NAME.strip() or "yt-chatbot"


def _get_video_filter(video_id: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="video_id",
                match=qmodels.MatchValue(value=video_id),
            )
        ]
    )


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        qdrant_url = settings.QDRANT_URL.strip()
        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is required for vector storage")

        client_kwargs = {"url": qdrant_url}
        qdrant_api_key = settings.QDRANT_API_KEY.strip()
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key

        _qdrant_client = QdrantClient(**client_kwargs)
    return _qdrant_client


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
    """Create Qdrant collection and payload indices if they don't exist."""
    from qdrant_client.http import models as http_models
    
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = _get_collection_name()

    if not client.collection_exists(collection_name):
        logger.info(f"Creating Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(f"Collection {collection_name} created successfully")
    else:
        logger.info(f"Collection {collection_name} already exists")
    
    # Ensure payload index exists for video_id filtering
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="video_id",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Ensured keyword index on video_id for {collection_name}")
    except Exception as e:
        # Index might already exist, which is fine
        logger.debug(f"Payload index creation: {e}")
    # Ensure numeric indices for time range filtering (start_time, end_time)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="start_time",
            field_schema=http_models.PayloadSchemaType.FLOAT,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="end_time",
            field_schema=http_models.PayloadSchemaType.FLOAT,
        )
        logger.info(f"Ensured numeric payload indices for start_time/end_time on {collection_name}")
    except Exception as e:
        logger.debug(f"Payload numeric index creation: {e}")


def generate_vector_id(video_id: str, chunk_index: int) -> str:
    """Generate a deterministic vector ID for a chunk."""
    raw = f"{video_id}_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


async def upsert_documents(
    documents: list[Document],
    video_id: str,
) -> int:
    """
    Embed and upsert documents into Qdrant.
    Uses video_id as a payload filter for isolation.

    Returns the number of vectors upserted.
    """
    if not documents:
        return 0

    ensure_index_exists()
    client = get_qdrant_client()
    collection_name = _get_collection_name()
    embeddings = get_embeddings()

    # Delete existing vectors for this video (re-index support)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(filter=_get_video_filter(video_id)),
            wait=True,
        )
        logger.info(f"Cleared existing vectors for video_id: {video_id}")
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
        points = []
        keyword_rows = []
        for j, (doc, dense_vec) in enumerate(zip(batch, dense_vectors)):
            vec_id = generate_vector_id(video_id, i + j)

            # Build payload metadata (kept small for vector DB storage).
            payload = {
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

            points.append(
                qmodels.PointStruct(
                    id=vec_id,
                    vector=dense_vec,
                    payload=payload,
                )
            )
            keyword_rows.append(payload)

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
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
        total_upserted += len(points)

    logger.info(f"Upserted {total_upserted} vectors for video {video_id}")
    return total_upserted


async def query_vectors(
    query: str,
    video_id: str,
    top_k: int = 20,
    time_window: tuple[float, float] | None = None,
) -> list[Document]:
    """
    Query Qdrant for similar documents.
    Returns LangChain Documents with metadata.
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()
    collection_name = _get_collection_name()

    if not client.collection_exists(collection_name):
        return []

    # Generate query embedding
    query_vector = embeddings.embed_query(query)

    # Query Qdrant
    # Build filter: base video_id filter plus optional time overlap filter
    q_filter = _get_video_filter(video_id)
    if time_window is not None:
        start, end = time_window
        try:
            time_conditions = [
                qmodels.FieldCondition(
                    key="start_time",
                    range=qmodels.Range(lte=end),
                ),
                qmodels.FieldCondition(
                    key="end_time",
                    range=qmodels.Range(gte=start),
                ),
            ]
            # combine with existing must conditions
            q_filter.must.extend(time_conditions)
        except Exception:
            # If range conditions not supported, ignore and continue
            pass

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
    )

    matches = getattr(results, "points", None)
    if matches is None:
        matches = getattr(results, "result", None)
    if matches is None:
        matches = results if isinstance(results, list) else []

    # Convert to LangChain Documents
    documents = []
    for match in matches:
        metadata = match.payload or {}
        doc = Document(
            page_content=metadata.get("text", ""),
            metadata={
                "score": float(match.score or 0.0),
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
    time_window: tuple[float, float] | None = None,
) -> list[Document]:
    """Query local SQLite FTS keyword index for sparse retrieval."""
    try:
        with _keyword_db_lock:
            conn = _ensure_keyword_index()
            # If a time window is specified, filter for chunks that overlap
            # the requested window: (start_time < end) AND (end_time > start)
            if time_window is not None:
                start, end = time_window
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
                      AND c.start_time < ? AND c.end_time > ?
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    (query, video_id, end, start, top_k),
                ).fetchall()
            else:
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
    Check if a video has been indexed in Qdrant.
    Returns stats about the namespace.
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name()
        if not client.collection_exists(collection_name):
            return {"indexed": False, "chunk_count": 0}

        count_result = client.count(
            collection_name=collection_name,
            count_filter=_get_video_filter(video_id),
            exact=True,
        )
        count = int(getattr(count_result, "count", 0) or 0)
        return {"indexed": count > 0, "chunk_count": count}
    except Exception as e:
        logger.error(f"Error checking index status for {video_id}: {e}")
        return {"indexed": False, "chunk_count": 0}
