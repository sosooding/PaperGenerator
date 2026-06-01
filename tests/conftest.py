"""
Shared pytest fixtures for hermetic node testing (TEST-1).

Provides fake papers, gaps, and mock I/O objects so individual node
tests never reach ArXiv, Gemini, or ChromaDB.
"""

from typing import List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Domain helpers (importable by any test module)
# ---------------------------------------------------------------------------

def make_paper(
    title: str = "Test Paper",
    doi: str = "10.1/test",
    year: int = 2023,
    abstract: str = "An abstract.",
    score: float = 0.9,
):
    return {
        "title": title,
        "abstract": abstract,
        "authors": ["First Author", "Second Author"],
        "doi": doi,
        "pdf_url": None,
        "source": "arxiv",
        "year": year,
        "relevance_score": score,
    }


def make_gap(title: str = "Test Gap", novelty: float = 0.9):
    return {
        "gap_title": title,
        "description": "A gap description for testing.",
        "supporting_evidence": [],
        "novelty_score": novelty,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_papers() -> List[dict]:
    return [make_paper("Paper A", "10.1/a"), make_paper("Paper B", "10.1/b")]


@pytest.fixture
def fake_gaps() -> List[dict]:
    return [make_gap("Gap 1", 0.9), make_gap("Gap 2", 0.7)]


@pytest.fixture
def mock_llm_subqueries():
    """LLM that returns a valid JSON sub-query list."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='["sub-query 1", "sub-query 2"]')
    return llm


@pytest.fixture
def mock_vector_store(fake_papers):
    """VectorStore mock with canned similarity results."""
    vs = MagicMock()
    vs.collection.count.return_value = len(fake_papers)
    vs.get_relevant_papers.return_value = fake_papers
    vs.similarity_search.return_value = fake_papers
    return vs
