import json
import uuid
import requests

url = 'http://127.0.0.1:8000/api/chat'
payload = {
    'video_id': 'PY9DcIMGxMs',
    'message': 'what are the main points in the first 5 minutes?',
    'session_id': str(uuid.uuid4()),
}

print('Sending request...')
import time
start = time.time()
resp = requests.post(url, json=payload, timeout=120)
elapsed = time.time() - start
print('STATUS', resp.status_code, 'elapsed=', f"{elapsed:.2f}s")
print(json.dumps(resp.json(), indent=2))
