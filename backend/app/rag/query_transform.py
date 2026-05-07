"""
Query transformation: rewriting and multi-query generation.
Pre-retrieval step to improve search quality.
"""

import logging
from langchain_openai import ChatOpenAI
from app.config import get_settings
from app.rag.prompts import QUERY_REWRITE_PROMPT, MULTI_QUERY_PROMPT

logger = logging.getLogger(__name__)


async def rewrite_query(
    query: str,
    chat_history: str = "",
) -> str:
    """
    Rewrite the user's query to be more search-friendly.
    Resolves pronouns and references using conversation history.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "yt-chatbot",
        },
        temperature=0,
        max_tokens=200,
    )

    try:
        chain = QUERY_REWRITE_PROMPT | llm
        result = await chain.ainvoke({
            "query": query,
            "chat_history": chat_history or "No previous conversation.",
        })
        rewritten = result.content.strip()
        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return query


async def generate_multi_queries(query: str) -> list[str]:
    """
    Generate multiple query variations to improve retrieval coverage.
    Returns the original query + 3 generated alternatives.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "yt-chatbot",
        },
        temperature=0.7,
        max_tokens=300,
    )

    try:
        chain = MULTI_QUERY_PROMPT | llm
        result = await chain.ainvoke({"query": query})

        # Parse the 3 queries from the response
        queries = [
            line.strip()
            for line in result.content.strip().split("\n")
            if line.strip()
        ][:3]

        # Always include the original query
        all_queries = [query] + queries
        logger.info(f"Generated {len(all_queries)} query variations")
        return all_queries

    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}")
        return [query]
