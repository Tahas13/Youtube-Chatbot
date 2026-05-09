import requests
import uuid
import json

url = "http://127.0.0.1:8000/api/chat"
payload = {
    "video_id": "PY9DcIMGxMs",
    "message": "explain the content explained in first 5 mins in video",
    "session_id": str(uuid.uuid4()),
}

try:
    r = requests.post(url, json=payload, timeout=60)
    print("STATUS", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
except Exception as e:
    print("Request failed:", e)
