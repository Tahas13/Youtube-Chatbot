import sys, asyncio, time
sys.path.insert(0, '.')
from app.rag.retriever import full_retrieval_pipeline

async def main():
    query = 'explain the content explained in first 5 mins in video'
    # Restrict retrieval to first 5 minutes (0 - 300 seconds)
    docs = await full_retrieval_pipeline([query], 'PY9DcIMGxMs', query_type='search', domain='other', time_window=(0.0, 300.0))
    print('Retrieved docs:', len(docs))
    for i,d in enumerate(docs[:10], start=1):
        meta = d.metadata if hasattr(d, 'metadata') else d.get('metadata', {})
        print(
            i,
            'start_time=', meta.get('start_time'),
            'end_time=', meta.get('end_time'),
            'start_display=', meta.get('start_display'),
            'chunk_index=', meta.get('chunk_index'),
            'score=', meta.get('score'),
        )

if __name__ == '__main__':
    asyncio.run(main())
