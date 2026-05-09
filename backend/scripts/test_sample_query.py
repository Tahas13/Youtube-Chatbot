"""
Sample end-to-end test: ingest a YouTube video and query it.
Requires backend running at http://localhost:8000
"""

import json
import sys
import urllib.request
import urllib.error
import time

BASE_URL = "http://localhost:8000/api"

# Use a Corey Schafer Python tutorial - confirmed has transcripts available
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=wfcWRAxRVBA"  # Python Classes and Objects

def test_ingest():
    """Ingest a YouTube video."""
    print("\n📥 Ingesting YouTube video...")
    print(f"   URL: {TEST_VIDEO_URL}")
    
    try:
        req_data = json.dumps({"video_url": TEST_VIDEO_URL}).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/ingest",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode('utf-8'))
            print(f"✅ Ingest successful!")
            print(f"   Title: {data.get('title', 'N/A')}")
            print(f"   Video ID: {data.get('video_id', 'N/A')}")
            print(f"   Chunks: {data.get('chunk_count', 0)}")
            print(f"   Duration: {data.get('duration_seconds', 0)}s")
            
            return data.get("video_id")
    except urllib.error.URLError as e:
        print(f"❌ Cannot connect to backend at http://localhost:8000")
        print(f"   Error: {e}")
        return None
    except urllib.error.HTTPError as e:
        print(f"❌ Ingest failed: {e.code}")
        try:
            error_body = e.read().decode('utf-8')
            error_data = json.loads(error_body)
            print(f"   Error: {error_data.get('detail', e.reason)}")
        except:
            print(f"   Error body: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"❌ Error during ingest: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_query(video_id):
    """Query the ingested video."""
    print(f"\n💬 Querying video {video_id}...")
    
    test_questions = [
        "What is the main topic of this video?",
    ]
    
    try:
        for question in test_questions:
            print(f"\n   Question: {question}")
            
            req_data = json.dumps({
                "video_id": video_id,
                "message": question,
                "session_id": "test-session"
            }).encode('utf-8')
            req = urllib.request.Request(
                f"{BASE_URL}/chat",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                data = json.loads(response.read().decode('utf-8'))
                print(f"   ✅ Response received!")
                answer = data.get('answer', 'N/A')
                print(f"   Answer: {answer[:200]}...")
                
                citations = data.get("citations", [])
                if citations:
                    print(f"   Citations: {len(citations)} chunks referenced")
                
                return True
    except urllib.error.URLError as e:
        print(f"   ❌ Connection error: {e}")
        return False
    except urllib.error.HTTPError as e:
        print(f"   ❌ Query failed: {e.code}")
        return False
    except Exception as e:
        print(f"   ❌ Error during query: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("YouTube Chatbot — Sample Query Test")
    print("=" * 60)
    
    # Step 1: Ingest
    video_id = test_ingest()
    if not video_id:
        print("\n❌ Test failed: could not ingest video")
        sys.exit(1)
    
    # Wait for indexes to be updated
    print("\n⏳ Waiting for indexes to be updated...")
    time.sleep(2)
    
    # Step 2: Query
    success = test_query(video_id)
    if not success:
        print("\n❌ Test failed: could not query video")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
