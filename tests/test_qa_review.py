"""
QA review — tests that demonstrate defects found during code review.

Run with:  python -m pytest tests/test_qa_review.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.graph import draft_reviewer_node, gap_selector_node, get_initial_state
from src.writing.writer import generate_section
from src.retrieval.vector_store import _paper_from_chroma_row, VectorStore


# ===========================================================================
# DEFECT 3 (was HIGH — FIX-1 applied) — citations recomputed after edit.
# These now PASS to confirm the fix holds.
# ===========================================================================

def _state_with_section(content, citations):
    state = get_initial_state("Q")
    state["draft_sections"] = [
        {"section_name": "abstract", "content": content, "citations": citations}
    ]
    return state


def test_edit_recomputes_citations_when_user_adds_one():
    state = _state_with_section("Original text citing [S1].", ["S1"])
    edits = json.dumps({"abstract": "Rewritten text citing [S1] and now [S2]."})

    with patch("src.agents.graph.interrupt", return_value=edits):
        result = draft_reviewer_node(state)

    abstract = result["draft_sections"][0]
    assert abstract["citations"] == ["S1", "S2"]


def test_edit_recomputes_citations_when_user_removes_one():
    state = _state_with_section("Old text [S1][S2].", ["S1", "S2"])
    edits = json.dumps({"abstract": "Clean prose with no citations at all."})

    with patch("src.agents.graph.interrupt", return_value=edits):
        result = draft_reviewer_node(state)

    abstract = result["draft_sections"][0]
    assert abstract["citations"] == []


# ===========================================================================
# DEFECT 5 (was MEDIUM — FIX-3 applied) — HumanDecision.edited_sections type.
# ===========================================================================

def test_gap_selector_edited_sections_is_dict_or_none():
    state = get_initial_state("Q")
    state["gaps"] = [
        {"gap_title": "G1", "description": "d", "supporting_evidence": [], "novelty_score": 0.9},
    ]

    with patch("src.agents.graph.interrupt", return_value="1"):
        result = gap_selector_node(state)

    decision = result["human_decisions"][0]
    assert decision["edited_sections"] is None or isinstance(decision["edited_sections"], dict)


def test_human_decision_edited_sections_type_consistent_across_nodes():
    s1 = get_initial_state("Q")
    s1["gaps"] = [{"gap_title": "G", "description": "d", "supporting_evidence": [], "novelty_score": 0.5}]
    with patch("src.agents.graph.interrupt", return_value="1"):
        gap_decision = gap_selector_node(s1)["human_decisions"][0]

    s2 = _state_with_section("text", [])
    with patch("src.agents.graph.interrupt", return_value=""):
        draft_decision = draft_reviewer_node(s2)["human_decisions"][0]

    # Both must be None or dict — never a list.
    for decision in (gap_decision, draft_decision):
        assert decision["edited_sections"] is None or isinstance(decision["edited_sections"], dict)


# ===========================================================================
# DEFECT 4 (was HIGH — FIX-4 applied) — hallucinated citations are dropped.
# ===========================================================================

def test_generated_citations_must_exist_in_source_map():
    source_map = {"S1": {"title": "Only Paper", "authors": ["A"], "year": 2020, "doi": "10/x",
                         "abstract": "abs", "pdf_url": None, "source": "arxiv",
                         "relevance_score": 0.9}}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="A claim [S1]. Another claim [S9].")
    gap = {"gap_title": "G", "description": "d", "supporting_evidence": [], "novelty_score": 0.5}

    section = generate_section("introduction", "Q", gap, source_map, "outline", [], llm)

    invalid = [c for c in section["citations"] if c not in source_map]
    assert not invalid, f"section contains citations not in source_map: {invalid}"


# ===========================================================================
# DEFECT 6 (was MEDIUM — FIX-5 applied) — relevance_score is non-negative.
# ===========================================================================

def test_relevance_score_is_non_negative():
    meta = {"title": "T", "authors": "A", "source": "arxiv",
            "year": "2020", "doi": "", "pdf_url": ""}
    paper = _paper_from_chroma_row(meta, document="abstract", distance=1.8)
    assert paper["relevance_score"] >= 0.0


# ===========================================================================
# DEFECT 7 (was LOW — FIX-6 applied) — k resolves from Config at call time.
# TD-4: rewritten to use REF-4 injection and assert resolved n_results, not
#       the signature default.
# ===========================================================================

def test_similarity_search_k_default_follows_config(monkeypatch):
    """Changing Config.TOP_K_PAPERS is reflected in the next similarity_search call."""
    from src.utils import config as config_module
    monkeypatch.setattr(config_module.Config, "TOP_K_PAPERS", 7)

    fake_collection = MagicMock()
    fake_collection.count.return_value = 100
    fake_collection.query.return_value = {
        "metadatas": [[]], "documents": [[]], "distances": [[]]
    }
    fake_chroma = MagicMock()
    fake_chroma.get_or_create_collection.return_value = fake_collection
    fake_genai = MagicMock()
    fake_genai.models.embed_content.return_value = MagicMock(
        embeddings=[MagicMock(values=[0.1])]
    )

    vs = VectorStore(chroma_client=fake_chroma, genai_client=fake_genai)
    vs.similarity_search("test query")  # no explicit k → should use Config value

    _, call_kwargs = fake_collection.query.call_args
    assert call_kwargs["n_results"] == 7, (
        f"expected n_results=7 (from config), got {call_kwargs['n_results']}"
    )
