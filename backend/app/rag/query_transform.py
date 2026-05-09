"""
Query transformation: rewriting and multi-query generation.
Pre-retrieval step to improve search quality.
"""

import logging
from app.config import get_settings
from app.rag.prompts import QUERY_REWRITE_PROMPT, MULTI_QUERY_PROMPT
from app.rag.llm import get_llm, get_fallback_llm
import asyncio

logger = logging.getLogger(__name__)

# Mock LLM flag
_SETTINGS = get_settings()
_USE_MOCK_LLM = str(_SETTINGS.LLM_MODEL).strip().lower() == "mock"


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc) or ""
    return "quota" in msg.lower() or "exceeded" in msg.lower() or "resourceexhausted" in msg.lower()


async def rewrite_query(
    query: str,
    chat_history: str = "",
) -> str:
    """
    Rewrite the user's query to be more search-friendly.
    Resolves pronouns and references using conversation history.
    """
    # If mock LLM, skip rewriting
    if _USE_MOCK_LLM:
        logger.info("Mock LLM enabled — skipping query rewrite (returning original)")
        return query

    settings = get_settings()
    timeout = float(settings.LLM_STEP_TIMEOUT_SECONDS or 8.0)
    llm = get_llm(temperature=0, max_output_tokens=200)

    try:
        chain = QUERY_REWRITE_PROMPT | llm
        result = await asyncio.wait_for(
            chain.ainvoke({"query": query, "chat_history": chat_history or "No previous conversation."}),
            timeout=timeout,
        )
        rewritten = result.content.strip()
        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
    except asyncio.TimeoutError:
        logger.warning("Query rewrite timed out, attempting fallback or using original")
        fb_llm = get_fallback_llm(temperature=0, max_output_tokens=200)
        if fb_llm is not None:
            try:
                chain = QUERY_REWRITE_PROMPT | fb_llm
                result = await asyncio.wait_for(chain.ainvoke({"query": query, "chat_history": chat_history}), timeout=4.0)
                return result.content.strip()
            except Exception as e:
                logger.warning(f"Fallback rewrite also failed: {e}")
        return query
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
        if _is_quota_error(e):
            fb_llm = get_fallback_llm(temperature=0, max_output_tokens=200)
            if fb_llm is not None:
                try:
                    chain = QUERY_REWRITE_PROMPT | fb_llm
                    result = await asyncio.wait_for(chain.ainvoke({"query": query, "chat_history": chat_history}), timeout=4.0)
                    return result.content.strip()
                except Exception as fb_err:
                    logger.warning(f"Fallback rewrite after quota error failed: {fb_err}")
        return query


async def generate_multi_queries(query: str) -> list[str]:
    """
    Generate multiple query variations to improve retrieval coverage.
    Returns the original query + 3 generated alternatives.
    """
    # If mock LLM, skip multi-query generation
    if _USE_MOCK_LLM:
        logger.info("Mock LLM enabled — skipping multi-query generation (returning original)")
        return [query]

    settings = get_settings()
    timeout = float(settings.LLM_STEP_TIMEOUT_SECONDS or 8.0)
    llm = get_llm(temperature=0.7, max_output_tokens=300)

    try:
        chain = MULTI_QUERY_PROMPT | llm
        result = await asyncio.wait_for(chain.ainvoke({"query": query}), timeout=timeout)

        queries = [
            line.strip()
            for line in result.content.strip().split("\n")
            if line.strip()
        ][:3]
        all_queries = [query] + queries
        logger.info(f"Generated {len(all_queries)} query variations")
        return all_queries

    except asyncio.TimeoutError:
        logger.warning("Multi-query generation timed out, using original")
        fb_llm = get_fallback_llm(temperature=0.7, max_output_tokens=200)
        if fb_llm is not None:
            try:
                chain = MULTI_QUERY_PROMPT | fb_llm
                result = await asyncio.wait_for(chain.ainvoke({"query": query}), timeout=4.0)
                queries = [l.strip() for l in result.content.strip().split("\n") if l.strip()][:3]
                return [query] + queries
            except Exception as e:
                logger.warning(f"Fallback multi-query generation failed: {e}")
        return [query]
    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}")
        if _is_quota_error(e):
            fb_llm = get_fallback_llm(temperature=0.7, max_output_tokens=200)
            if fb_llm is not None:
                try:
                    chain = MULTI_QUERY_PROMPT | fb_llm
                    result = await asyncio.wait_for(chain.ainvoke({"query": query}), timeout=4.0)
                    queries = [l.strip() for l in result.content.strip().split("\n") if l.strip()][:3]
                    return [query] + queries
                except Exception as fb_err:
                    logger.warning(f"Fallback multi-query after quota error failed: {fb_err}")
        return [query]
