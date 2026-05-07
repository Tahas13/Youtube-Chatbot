# Backend Operations Notes

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill required secrets.

Core keys:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

Optional/advanced:
- `COHERE_API_KEY` (reranking; system falls back to score-based ranking when missing)
- `PINECONE_CLOUD`, `PINECONE_REGION`
- `EMBEDDING_DIMENSIONS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SEMANTIC_CHUNKING`
- Retrieval tuning: `RETRIEVAL_K`, `RETRIEVAL_TOP_N`, `MMR_LAMBDA`, `MMR_FETCH_K`
- Guardrail tuning: `EVIDENCE_CONFIDENCE_THRESHOLD`

## Session Memory Backend Selection and Fallback

Session storage is controlled by `SESSION_STORE_BACKEND`:
- `auto` (default): try Redis first, then SQLite, then in-memory.
- `redis`: prefer Redis; falls back to SQLite if Redis is unavailable.
- `sqlite`: use SQLite persistence.
- `memory`: in-memory only (non-persistent, process-local).

Related vars:
- `SESSION_TTL_SECONDS`
- `REDIS_URL`
- `SESSION_SQLITE_PATH`

Operational behavior:
- Redis backend stores each session list with TTL refresh on read/write.
- SQLite backend stores message rows and prunes expired rows by TTL.
- Memory backend is always available as a final fallback.

## Hybrid Retrieval Sparse Index Maintenance

Dense retrieval uses Pinecone. Sparse retrieval uses local SQLite FTS at `KEYWORD_INDEX_SQLITE_PATH`.

During ingest/upsert:
1. Existing dense vectors for the video namespace are deleted from Pinecone.
2. Existing sparse rows for the video are deleted from SQLite (`chunks` and `chunks_fts`).
3. New chunks are inserted into both dense and sparse stores.

This keeps dense and sparse indexes aligned per video re-index, without requiring separate maintenance jobs.
