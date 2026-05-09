"""
Hybrid retrieval with MMR diversity and reranking.
Core retrieval pipeline combining dense vector search,
MMR for diversity, and cross-encoder reranking.
"""

import logging
from dataclasses import dataclass
from langchain_core.documents import Document
from app.config import get_settings
from app.services.vectorstore import query_vectors, query_keyword_documents

logger = logging.getLogger(__name__)

_ranker = None
_ranker_unavailable = False


def _get_result_value(result, key: str):
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


@dataclass
class RetrievalPolicy:
    top_k: int
    top_n: int
    mmr_lambda: float
    strategy: str = "balanced"


def get_retrieval_policy(query_type: str, domain: str) -> RetrievalPolicy:
    """Resolve retrieval settings from query intent + content domain."""
    settings = get_settings()
    policy = RetrievalPolicy(
        top_k=settings.RETRIEVAL_K,
        top_n=settings.RETRIEVAL_TOP_N,
        mmr_lambda=settings.MMR_LAMBDA,
        strategy="balanced",
    )

    if query_type == "summarize":
        policy.top_k = max(policy.top_k, 30)
        policy.top_n = max(policy.top_n, 12)
        policy.mmr_lambda = 0.55
        policy.strategy = "coverage"
    elif query_type == "clarify":
        policy.top_n = min(policy.top_n, 4)
        policy.mmr_lambda = 0.8
        policy.strategy = "precision"

    if domain in {"tutorial", "educational"}:
        policy.mmr_lambda = min(policy.mmr_lambda, 0.65)
        policy.top_k = max(policy.top_k, 24)
    elif domain in {"news", "podcast"}:
        policy.top_k = max(policy.top_k, 28)
        policy.top_n = max(policy.top_n, 6)
    elif domain == "entertainment":
        policy.top_k = max(policy.top_k, 18)
        policy.mmr_lambda = max(policy.mmr_lambda, 0.72)

    return policy


def _doc_key(doc: Document) -> tuple[str, int]:
    return (doc.metadata.get("video_id", ""), int(doc.metadata.get("chunk_index", 0)))


def _rrf_fuse(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[Document]:
    """
    Fuse ranked dense/sparse results using reciprocal rank fusion.
    RRF score = sum(1 / (k + rank_i)).
    """
    scores: dict[tuple[str, int], float] = {}
    canonical: dict[tuple[str, int], Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in canonical:
                canonical[key] = doc
            else:
                if doc.metadata.get("sparse_score", 0) > canonical[key].metadata.get("sparse_score", 0):
                    canonical[key].metadata["sparse_score"] = doc.metadata.get("sparse_score", 0)
                if doc.metadata.get("score", 0) > canonical[key].metadata.get("score", 0):
                    canonical[key].metadata["score"] = doc.metadata.get("score", 0)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    output: list[Document] = []
    for key, fused_score in fused[:top_k]:
        doc = canonical[key]
        doc.metadata["fused_score"] = fused_score
        output.append(doc)
    return output


async def retrieve_with_mmr(
    queries: list[str],
    video_id: str,
    top_k: int = 20,
    lambda_mult: float = 0.7,
    time_window: tuple[float, float] | None = None,
) -> list[Document]:
    """
    Retrieve documents using multiple queries and apply MMR
    for diversity.

    1. Run each query against the dense vector store and local keyword index
    2. Merge and deduplicate results
    3. Apply MMR to select diverse, relevant documents
    """
    settings = get_settings()
    top_k = top_k or settings.RETRIEVAL_K
    lambda_mult = lambda_mult or settings.MMR_LAMBDA

    # ── Step 1: Retrieve dense + sparse and fuse ──
    all_docs: list[Document] = []
    seen_chunks: set[tuple[str, int]] = set()

    for query in queries:
        try:
            dense_docs = await query_vectors(
                query=query,
                video_id=video_id,
                top_k=top_k,
                time_window=time_window,
            )
        except Exception as e:
            logger.warning(f"Dense retrieval failed (falling back): {e}")
            dense_docs = []

        try:
            sparse_docs = await query_keyword_documents(
                query=query,
                video_id=video_id,
                top_k=top_k,
                time_window=time_window,
            )
        except Exception as e:
            logger.warning(f"Sparse retrieval failed (falling back): {e}")
            sparse_docs = []

        docs = _rrf_fuse([dense_docs, sparse_docs], top_k=top_k)

        for doc in docs:
            key = _doc_key(doc)
            if key not in seen_chunks:
                seen_chunks.add(key)
                all_docs.append(doc)

    logger.info(f"Retrieved {len(all_docs)} fused documents from {len(queries)} queries")

    if not all_docs:
        return []

    # ── Step 2: Apply MMR for diversity ──
    selected = _apply_mmr(all_docs, lambda_mult=lambda_mult, k=min(top_k, len(all_docs)))

    logger.info(f"MMR selected {len(selected)} diverse documents")
    return selected


def _apply_mmr(
    documents: list[Document],
    lambda_mult: float = 0.7,
    k: int = 10,
) -> list[Document]:
    """
    Apply Maximal Marginal Relevance to select diverse documents.

    Uses score-based MMR: balances relevance (original score)
    with diversity (penalizing similar documents).
    """
    if len(documents) <= k:
        return documents

    # Sort by score (highest first)
    scored_docs = sorted(
        documents,
        key=lambda d: d.metadata.get("score", 0),
        reverse=True,
    )

    # Simple MMR: start with highest-scored, then greedily add
    # documents that are different from already selected ones
    selected = [scored_docs[0]]
    remaining = scored_docs[1:]

    while len(selected) < k and remaining:
        best_idx = 0
        best_mmr_score = -float("inf")

        for i, candidate in enumerate(remaining):
            # Relevance component
            relevance = candidate.metadata.get("score", 0)

            # Diversity component: check overlap with selected docs
            max_similarity = 0
            for sel in selected:
                # Use simple text overlap as a proxy for similarity
                overlap = _text_overlap(candidate.page_content, sel.page_content)
                max_similarity = max(max_similarity, overlap)

            # MMR score: balance relevance and diversity
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _get_ranker():
    global _ranker, _ranker_unavailable
    if _ranker is not None:
        return _ranker
    if _ranker_unavailable:
        return None

    try:
        from flashrank import Ranker

        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        return _ranker
    except Exception as e:
        logger.warning(f"FlashRank unavailable, falling back to score-based ranking: {e}")
        _ranker_unavailable = True
        return None


def _text_overlap(text1: str, text2: str) -> float:
    """Compute simple word overlap ratio between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


async def rerank_documents(
    documents: list[Document],
    query: str,
    top_n: int = 5,
) -> list[Document]:
    """
    Rerank documents using FlashRank or fallback to score-based ranking.
    """

    if not documents:
        return []

    top_n = min(top_n, len(documents))

    # Try local FlashRank reranking
    try:
        ranker = _get_ranker()
        if ranker is not None:
            return await _flashrank_rerank(documents, query, top_n, ranker)
    except Exception as e:
        logger.warning(f"FlashRank reranking failed, falling back to score-based: {e}")

    # Fallback: sort by existing score and take top N
    sorted_docs = sorted(
        documents,
        key=lambda d: max(
            d.metadata.get("score", 0),
            d.metadata.get("fused_score", 0),
            d.metadata.get("sparse_score", 0),
        ),
        reverse=True,
    )
    return sorted_docs[:top_n]


async def _flashrank_rerank(
    documents: list[Document],
    query: str,
    top_n: int,
    ranker,
) -> list[Document]:
    """Rerank using FlashRank's local cross-encoder model."""
    from flashrank import RerankRequest

    passages = [
        {"id": str(index), "text": doc.page_content}
        for index, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    reranked = []
    for result in results[:top_n]:
        result_index = int(_get_result_value(result, "id") or 0)
        relevance_score = float(_get_result_value(result, "score") or 0.0)
        doc = documents[result_index]
        doc.metadata["rerank_score"] = relevance_score
        reranked.append(doc)

    logger.info(f"Reranked {len(documents)} → {len(reranked)} documents with FlashRank")
    return reranked


async def full_retrieval_pipeline(
    queries: list[str],
    video_id: str,
    query_type: str = "search",
    domain: str = "other",
    time_window: tuple[float, float] | None = None,
) -> list[Document]:
    """
    Execute the full retrieval pipeline:
    1. Multi-query retrieval
    2. MMR for diversity
    3. Reranking for relevance

    Returns the final top-N most relevant, diverse documents.
    """
    policy = get_retrieval_policy(query_type=query_type, domain=domain)
    # time_window may be provided by caller to filter retrieval by time overlap

    # Step 1 + 2: Retrieve with MMR
    mmr_docs = await retrieve_with_mmr(
        queries=queries,
        video_id=video_id,
        top_k=policy.top_k,
        lambda_mult=policy.mmr_lambda,
        time_window=time_window,
    )

    if not mmr_docs:
        return []

    # Step 3: Rerank
    reranked = await rerank_documents(
        documents=mmr_docs,
        query=queries[0],  # Use the primary query for reranking
        top_n=policy.top_n,
    )

    return reranked
