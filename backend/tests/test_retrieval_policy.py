from types import SimpleNamespace
from unittest.mock import patch

from app.rag.retriever import get_retrieval_policy


def _settings():
    return SimpleNamespace(
        RETRIEVAL_K=20,
        RETRIEVAL_TOP_N=5,
        MMR_LAMBDA=0.7,
    )


@patch("app.rag.retriever.get_settings", return_value=_settings())
def test_summarize_tutorial_policy_overrides(_mock_settings):
    policy = get_retrieval_policy(query_type="summarize", domain="tutorial")
    assert policy.top_k == 30
    assert policy.top_n == 8
    assert policy.mmr_lambda == 0.55
    assert policy.strategy == "coverage"


@patch("app.rag.retriever.get_settings", return_value=_settings())
def test_clarify_entertainment_policy_overrides(_mock_settings):
    policy = get_retrieval_policy(query_type="clarify", domain="entertainment")
    assert policy.top_k == 20
    assert policy.top_n == 4
    assert policy.mmr_lambda == 0.8
    assert policy.strategy == "precision"


@patch("app.rag.retriever.get_settings", return_value=_settings())
def test_news_domain_policy_expands_retrieval(_mock_settings):
    policy = get_retrieval_policy(query_type="search", domain="news")
    assert policy.top_k == 28
    assert policy.top_n == 6
    assert policy.mmr_lambda == 0.7
    assert policy.strategy == "balanced"
