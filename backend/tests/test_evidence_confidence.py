from langchain_core.documents import Document

from app.rag.nodes import _evidence_confidence


def test_evidence_confidence_all_supported():
    docs = [
        Document(page_content="chunk-a", metadata={"start_time": 10, "end_time": 20}),
        Document(page_content="chunk-b", metadata={"start_time": 50, "end_time": 60}),
    ]
    answer = "Key moments are at [00:10] and [00:50]."

    confidence = _evidence_confidence(answer, docs)

    assert confidence == 1.0


def test_evidence_confidence_partial_support():
    docs = [
        Document(page_content="chunk-a", metadata={"start_time": 120, "end_time": 140}),
    ]
    answer = "Look at [02:01] and [05:30]."

    confidence = _evidence_confidence(answer, docs)

    assert confidence == 0.5


def test_evidence_confidence_no_timestamps_returns_zero():
    docs = [
        Document(page_content="chunk-a", metadata={"start_time": 10, "end_time": 20}),
    ]

    confidence = _evidence_confidence("No explicit time mentioned.", docs)

    assert confidence == 0.0
