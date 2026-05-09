import requests
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=5)
    print('STATUS', r.status_code)
    print(r.text)
except Exception as e:
    print('ERR', e)
