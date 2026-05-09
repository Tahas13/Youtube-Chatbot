"""
Contextual compression for retrieved documents.
Extracts only the relevant parts of each document before passing to the LLM.
"""

import logging
import asyncio
import re
from langchain_core.documents import Document
from app.rag.llm import get_llm

logger = logging.getLogger(__name__)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc) or ""
    return "quota" in msg.lower() or "exceeded" in msg.lower() or "resourceexhausted" in msg.lower()

async def compress_documents(
    documents: list[Document],
    query: str,
) -> list[Document]:
    """
    Apply batch contextual compression to extract only relevant
    information from each retrieved document.
    """
    import asyncio
    from app.rag.llm import get_fallback_llm

    settings = __import__('app.config', fromlist=['get_settings']).get_settings()
    timeout = float(settings.LLM_STEP_TIMEOUT_SECONDS or 8.0)

    llm = get_llm(temperature=0, max_output_tokens=1500)

    batch_size = 4
    compressed = []

    async def _process_batch(batch: list[Document]) -> list[Document]:
        batch_context = ""
        for idx, doc in enumerate(batch):
            batch_context += f"--- CHUNK {idx} ---\n{doc.page_content}\n\n"

        prompt = f"""
        User Query: {query}
        Below are {len(batch)} segments of a video transcript. 
        For each segment, extract only the sentences directly relevant to the query.
        If a segment is not relevant, return "NOT_RELEVANT" for that chunk.
        
        Format your response exactly like this:
        RESULT 0: [Extracted text or NOT_RELEVANT]
        RESULT 1: [Extracted text or NOT_RELEVANT]
        ...
        
        Segments:
        {batch_context}
        """

        try:
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout)
            content = response.content.strip()

            # Parse results based on "RESULT X:"
            pattern = r"RESULT\s+(\d+):(.*?(?=RESULT\s+\d+:|$))"
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            
            parsed_results = {}
            for match in matches:
                idx = int(match.group(1))
                text = match.group(2).strip()
                parsed_results[idx] = text
            
            processed_batch = []
            for idx, doc in enumerate(batch):
                extracted = parsed_results.get(idx, "")
                if not extracted or "NOT_RELEVANT" in extracted.upper()[:20]:
                    logger.debug(f"Chunk at {doc.metadata.get('start_display')} filtered as not relevant")
                    continue
                    
                processed_batch.append(Document(
                    page_content=extracted,
                    metadata=doc.metadata.copy(),
                ))
            return processed_batch
        except asyncio.TimeoutError:
            logger.warning("Batch compression timed out, attempting fallback or keeping original batch")
            fb = get_fallback_llm(temperature=0, max_output_tokens=800)
            if fb is not None:
                try:
                    response = await asyncio.wait_for(fb.ainvoke(prompt), timeout=4.0)
                    content = response.content.strip()
                    pattern = r"RESULT\s+(\d+):(.*?(?=RESULT\s+\d+:|$))"
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

                    parsed_results = {}
                    for match in matches:
                        idx = int(match.group(1))
                        text = match.group(2).strip()
                        parsed_results[idx] = text

                    processed_batch = []
                    for idx, doc in enumerate(batch):
                        extracted = parsed_results.get(idx, "")
                        if not extracted or "NOT_RELEVANT" in extracted.upper()[:20]:
                            continue
                        processed_batch.append(Document(
                            page_content=extracted,
                            metadata=doc.metadata.copy(),
                        ))
                    return processed_batch
                except Exception as e:
                    logger.warning(f"Fallback compression failed: {e}")
                    return batch
            else:
                return batch
        except Exception as e:
            logger.warning(f"Batch compression failed for chunk: {e}")
            if _is_quota_error(e):
                fb = get_fallback_llm(temperature=0, max_output_tokens=800)
                if fb is not None:
                    try:
                        response = await asyncio.wait_for(fb.ainvoke(prompt), timeout=4.0)
                        content = response.content.strip()
                        pattern = r"RESULT\s+(\d+):(.*?(?=RESULT\s+\d+:|$))"
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

                        parsed_results = {}
                        for match in matches:
                            idx = int(match.group(1))
                            text = match.group(2).strip()
                            parsed_results[idx] = text

                        processed_batch = []
                        for idx, doc in enumerate(batch):
                            extracted = parsed_results.get(idx, "")
                            if not extracted or "NOT_RELEVANT" in extracted.upper()[:20]:
                                continue
                            processed_batch.append(Document(
                                page_content=extracted,
                                metadata=doc.metadata.copy(),
                            ))
                        return processed_batch or batch
                    except Exception as fb_err:
                        logger.warning(f"Fallback compression after quota error failed: {fb_err}")
            return batch

    # Collect batches and run them in parallel
    tasks = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        tasks.append(_process_batch(batch))
        
    results = await asyncio.gather(*tasks)
    for res_batch in results:
        compressed.extend(res_batch)

    logger.info(f"Compressed {len(documents)} → {len(compressed)} documents")
    return compressed


def format_context(documents: list[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    Includes timestamps and clear separation between chunks.
    """
    if not documents:
        return "No relevant transcript excerpts found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        start = doc.metadata.get("start_display", "00:00")
        end = doc.metadata.get("end_display", "00:00")
        text = doc.page_content.strip()

        context_parts.append(
            f"[Excerpt {i} | Timestamp: {start} - {end}]\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)
