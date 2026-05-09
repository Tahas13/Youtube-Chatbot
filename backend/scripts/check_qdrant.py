"""Quick Qdrant connectivity check."""
from app.config import get_settings
from qdrant_client import QdrantClient

settings = get_settings()
print("QDRANT_URL:", settings.QDRANT_URL)
print("QDRANT_API_KEY set:", bool(settings.QDRANT_API_KEY))
print("GOOGLE_API_KEY set:", bool(settings.GOOGLE_API_KEY))

try:
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    cols = client.get_collections()
    names = [c.name for c in getattr(cols, "collections", [])]
    print("✅ Qdrant connected. Collections:", names)
except Exception as e:
    print("❌ Qdrant connection failed:", e)
