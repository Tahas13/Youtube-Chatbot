import asyncio, json
import sys
sys.path.insert(0, '.')
from app.rag.retriever import full_retrieval_pipeline
from app.rag.nodes import _extractive_summary_from_docs

async def main():
    query = 'explain the content explained in first 5 mins in video'
    docs = await full_retrieval_pipeline([query], 'PY9DcIMGxMs', query_type='search', domain='other', time_window=(0.0, 300.0))
    print('Retrieved docs:', len(docs))
    for d in docs:
        meta = d.metadata
        print('-', meta.get('start_display'), meta.get('chunk_index'), meta.get('score'))
    fallback = _extractive_summary_from_docs(docs, max_bullets=6, video_title='')
    print('\nExtractive fallback:\n')
    print(fallback)

if __name__ == '__main__':
    asyncio.run(main())
