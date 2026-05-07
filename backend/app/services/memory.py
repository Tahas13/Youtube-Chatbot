"""
Conversation memory management for multi-turn chat sessions.
Uses LangGraph's MemorySaver for persistence within a session.
"""

import logging
import time
from typing import Optional
from threading import Lock
import sqlite3
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SessionStore:
    """
    In-memory session store for conversation history.
    Each session is keyed by session_id and stores message history.
    Sessions are auto-cleaned after TTL expiration.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, dict] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        with self._lock:
            self._cleanup_expired()
            session = self._store.get(session_id)
            if session:
                session["last_access"] = time.time()
                return session.get("messages", [])
            return []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session history."""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {
                    "messages": [],
                    "created_at": time.time(),
                    "last_access": time.time(),
                }

            self._store[session_id]["messages"].append({
                "role": role,
                "content": content,
            })
            self._store[session_id]["last_access"] = time.time()

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session. Returns True if session existed."""
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def get_context_window(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> list[dict]:
        """
        Get the last N messages for context window optimization.
        Always includes the system message if present.
        """
        history = self.get_history(session_id)
        if not history:
            return []

        # Keep system message + last N messages
        system_msgs = [m for m in history if m["role"] == "system"]
        non_system = [m for m in history if m["role"] != "system"]

        # Take last max_messages
        recent = non_system[-max_messages:]

        return system_msgs + recent

    def _cleanup_expired(self) -> None:
        """Remove sessions that have exceeded TTL."""
        now = time.time()
        expired = [
            sid for sid, session in self._store.items()
            if now - session.get("last_access", 0) > self._ttl
        ]
        for sid in expired:
            del self._store[sid]
            logger.debug(f"Cleaned up expired session: {sid}")


class PersistentSessionStore:
    """Persistent session store with Redis-first and SQLite fallback."""

    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._redis = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._backend = "memory"
        self._memory_fallback = SessionStore(ttl_seconds=ttl_seconds)
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        from app.config import get_settings
        settings = get_settings()
        backend_pref = settings.SESSION_STORE_BACKEND.lower()

        if backend_pref in ("auto", "redis"):
            try:
                import redis
                self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self._redis.ping()
                self._backend = "redis"
                logger.info("Session store initialized with Redis backend")
                return
            except Exception as e:
                logger.warning(f"Redis session backend unavailable: {e}")
                if backend_pref == "redis":
                    logger.warning("Falling back to SQLite session backend")

        if backend_pref in ("auto", "sqlite", "redis"):
            try:
                sqlite_path = Path(settings.SESSION_SQLITE_PATH)
                if not sqlite_path.is_absolute():
                    backend_root = Path(__file__).resolve().parents[2]
                    sqlite_path = backend_root / sqlite_path
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                self._sqlite_conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
                self._sqlite_conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_messages (
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                self._sqlite_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_messages_sid_created ON session_messages(session_id, created_at)"
                )
                self._sqlite_conn.commit()
                self._backend = "sqlite"
                logger.info("Session store initialized with SQLite backend")
                return
            except Exception as e:
                logger.warning(f"SQLite session backend unavailable: {e}")

        self._backend = "memory"
        logger.warning("Using in-memory session backend (non-persistent)")

    def _prune_sqlite(self) -> None:
        if not self._sqlite_conn:
            return
        cutoff = time.time() - self._ttl
        self._sqlite_conn.execute("DELETE FROM session_messages WHERE created_at < ?", (cutoff,))
        self._sqlite_conn.commit()

    def get_history(self, session_id: str) -> list[dict]:
        if self._backend == "redis" and self._redis:
            key = f"session:{session_id}"
            data = self._redis.lrange(key, 0, -1)
            self._redis.expire(key, self._ttl)
            return [json.loads(item) for item in data]

        if self._backend == "sqlite" and self._sqlite_conn:
            with self._lock:
                self._prune_sqlite()
                rows = self._sqlite_conn.execute(
                    """
                    SELECT role, content
                    FROM session_messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows]

        return self._memory_fallback.get_history(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        if self._backend == "redis" and self._redis:
            key = f"session:{session_id}"
            self._redis.rpush(key, json.dumps(message))
            self._redis.expire(key, self._ttl)
            return

        if self._backend == "sqlite" and self._sqlite_conn:
            with self._lock:
                self._sqlite_conn.execute(
                    """
                    INSERT INTO session_messages (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, content, time.time()),
                )
                self._sqlite_conn.commit()
            return

        self._memory_fallback.add_message(session_id, role, content)

    def clear_session(self, session_id: str) -> bool:
        if self._backend == "redis" and self._redis:
            deleted = self._redis.delete(f"session:{session_id}")
            return bool(deleted)

        if self._backend == "sqlite" and self._sqlite_conn:
            with self._lock:
                cursor = self._sqlite_conn.execute(
                    "DELETE FROM session_messages WHERE session_id = ?",
                    (session_id,),
                )
                self._sqlite_conn.commit()
                return cursor.rowcount > 0

        return self._memory_fallback.clear_session(session_id)

    def get_context_window(self, session_id: str, max_messages: int = 10) -> list[dict]:
        history = self.get_history(session_id)
        if not history:
            return []

        system_msgs = [m for m in history if m["role"] == "system"]
        non_system = [m for m in history if m["role"] != "system"]
        recent = non_system[-max_messages:]
        return system_msgs + recent


# Global session store singleton
_session_store: Optional[PersistentSessionStore] = None


def get_session_store() -> PersistentSessionStore:
    """Get or create the global session store."""
    global _session_store
    if _session_store is None:
        from app.config import get_settings
        settings = get_settings()
        _session_store = PersistentSessionStore(ttl_seconds=settings.SESSION_TTL_SECONDS)
    return _session_store
