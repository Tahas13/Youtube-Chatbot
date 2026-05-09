"""Diagnostic script to check video indexing status."""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.vectorstore import check_video_indexed

video_id = "wfcWRAxRVBA"
result = check_video_indexed(video_id)
print(f"Video ID: {video_id}")
print(f"Result: {result}")
print(f"Indexed: {result.get('indexed')}")
print(f"Chunk count: {result.get('chunk_count')}")
