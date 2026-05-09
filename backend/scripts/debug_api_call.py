import sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from app.main import app
import uuid

client = TestClient(app)
resp = client.post('/api/chat/extractive', json={
    'video_id':'PY9DcIMGxMs',
    'message':'explain the content explained in first 5 mins in video',
    'session_id': str(uuid.uuid4())
}, timeout=60)
print('STATUS', resp.status_code)
print(resp.text)
