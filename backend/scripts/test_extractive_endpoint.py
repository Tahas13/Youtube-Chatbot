import requests, uuid, json
url = 'http://127.0.0.1:8000/api/chat/extractive'
payload = {
    'video_id': 'PY9DcIMGxMs',
    'message': 'explain the content explained in first 5 mins in video',
    'session_id': str(uuid.uuid4())
}
resp = requests.post(url, json=payload, timeout=30)
print('STATUS', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
