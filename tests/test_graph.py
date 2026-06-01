"""
Unit tests for LangGraph workflow.
"""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.agents.graph import (
    create_graph,
    get_initial_state,
    planner_node,
    retriever_node,
    paper_approver_node,
    gap_finder_node,
    gap_selector_node,
    writer_node,
    critic_node,
    formatter_node,
    should_revise,
)
from src.utils.state import AgentState
from src.utils.config import Config


def test_get_initial_state():
    """Test initial state creation."""
    question = "What is graph coloring?"
    state = get_initial_state(question)

    assert isinstance(state, dict)
    assert state["research_question"] == question
    assert state["revision_count"] == 0
    assert state["current_phase"] == "initialized"


def test_planner_node():
    """Test planner node updates state correctly."""
    initial_state = get_initial_state("Test question")
    result = planner_node(initial_state)

    assert result["current_phase"] == "planning"
    assert result["research_question"] == "Test question"


def test_retriever_node():
    """Test retriever node updates state correctly."""
    initial_state = get_initial_state("Test question")
    initial_state["sub_queries"] = ["query1", "query2"]
    result = retriever_node(initial_state)

    assert result["current_phase"] == "retrieval"


def make_paper(title="Paper 1", doi="10.1/p1", year=2024, score=0.9):
    return {
        "title": title, "abstract": "Abstract.", "authors": ["Author"],
        "doi": doi, "pdf_url": None, "source": "arxiv",
        "year": year, "relevance_score": score,
    }


def test_paper_approver_node_no_papers():
    """Skip interrupt and return unchanged when no papers are in state."""
    state = get_initial_state("Test question")
    result = paper_approver_node(state)

    assert result["current_phase"] == "paper_approval"
    assert result["retrieved_papers"] == []
    assert result["human_decisions"] == []


def test_paper_approver_node_keep_all():
    """Empty input keeps all papers; no ChromaDB deletion."""
    state = get_initial_state("Test question")
    state["retrieved_papers"] = [make_paper("P1", "10.1/p1"), make_paper("P2", "10.1/p2")]

    with patch("src.agents.graph.interrupt", return_value=""):
        with patch("src.agents.graph.VectorStore") as mock_vs_cls:
            result = paper_approver_node(state)

    assert result["current_phase"] == "paper_approval"
    assert len(result["retrieved_papers"]) == 2
    assert len(result["human_decisions"]) == 1
    decision = result["human_decisions"][0]
    assert decision["checkpoint_name"] == "paper_approval"
    assert decision["removed_papers"] == []
    mock_vs_cls.assert_not_called()


def test_paper_approver_node_remove_papers():
    """Comma-separated indices remove the correct papers from state and ChromaDB."""
    state = get_initial_state("Test question")
    state["retrieved_papers"] = [
        make_paper("P1", "10.1/p1"),
        make_paper("P2", "10.1/p2"),
        make_paper("P3", "10.1/p3"),
    ]

    mock_vs = MagicMock()
    with patch("src.agents.graph.interrupt", return_value="1, 3"):
        with patch("src.agents.graph.VectorStore", return_value=mock_vs):
            result = paper_approver_node(state)

    assert len(result["retrieved_papers"]) == 1
    assert result["retrieved_papers"][0]["title"] == "P2"
    decision = result["human_decisions"][0]
    assert len(decision["removed_papers"]) == 2
    mock_vs.delete_papers.assert_called_once()


def test_gap_finder_node():
    """Test gap finder node updates state correctly."""
    initial_state = get_initial_state("Test question")
    initial_state["retrieved_papers"] = [
        {
            "title": "Paper 1",
            "abstract": "Abstract",
            "authors": ["Author"],
            "doi": "10.1234/test",
            "pdf_url": None,
            "source": "arxiv",
            "year": 2024,
            "relevance_score": 0.9,
        }
    ]
    result = gap_finder_node(initial_state)

    assert result["current_phase"] == "gap_finding"


def test_writer_node():
    """Test writer node generates outline and all 6 draft sections."""
    initial_state = get_initial_state("Test question")
    initial_state["selected_gap"] = {
        "gap_title": "Test gap",
        "description": "Description",
        "supporting_evidence": [],
        "novelty_score": 0.8,
    }
    initial_state["retrieved_papers"] = [make_paper("P1")]

    def _mock_section(section_name, **kwargs):
        return {"section_name": section_name, "content": "Content [S1].", "citations": ["S1"]}

    with patch("src.agents.graph.generate_outline", return_value="## Abstract\n- Point 1"), \
         patch("src.agents.graph.generate_section", side_effect=_mock_section), \
         patch("src.agents.graph.VectorStore") as mock_vs_cls:
        mock_vs_cls.return_value.similarity_search.return_value = []
        result = writer_node(initial_state)

    assert result["current_phase"] == "writing"
    assert result["outline"] == "## Abstract\n- Point 1"
    assert len(result["draft_sections"]) == 6
    assert result["citation_report"]["S1"]["title"] == "P1"


def test_writer_node_no_gap():
    """Writer node without a selected gap records an error and skips writing."""
    initial_state = get_initial_state("Test question")
    result = writer_node(initial_state)

    assert result["current_phase"] == "writing"
    assert any("no selected_gap" in e for e in result["errors"])


def test_critic_node():
    """Test critic node updates state correctly."""
    initial_state = get_initial_state("Test question")
    initial_state["draft_sections"] = [
        {"section_name": "abstract", "content": "Test content", "citations": []}
    ]
    result = critic_node(initial_state)

    assert result["current_phase"] == "critiquing"
    assert result["revision_count"] == 1


def test_formatter_node():
    """Test formatter node updates state correctly."""
    initial_state = get_initial_state("Test question")
    result = formatter_node(initial_state)

    assert result["current_phase"] == "formatting"


def test_should_revise_no_feedback():
    """Test routing when no critic feedback exists."""
    state = get_initial_state("Test question")
    result = should_revise(state)

    assert result == "formatter"


def test_should_revise_good_score():
    """Test routing when score is acceptable."""
    state = get_initial_state("Test question")
    state["critic_feedback"] = [
        {
            "score": 8.0,
            "flagged_sentences": [],
            "missing_sections": [],
            "coherence_issues": [],
            "grounding_score": 8.0,
        }
    ]
    result = should_revise(state)

    assert result == "formatter"


def test_should_revise_low_score():
    """Test routing when score is too low."""
    state = get_initial_state("Test question")
    state["critic_feedback"] = [
        {
            "score": 5.0,
            "flagged_sentences": [{"sentence": "test", "issue": "issue", "source_id": "S1"}],
            "missing_sections": [],
            "coherence_issues": [],
            "grounding_score": 5.0,
        }
    ]
    state["revision_count"] = 0
    result = should_revise(state)

    assert result == "writer"


def test_should_revise_max_revisions():
    """Test routing when max revisions reached."""
    state = get_initial_state("Test question")
    state["critic_feedback"] = [
        {
            "score": 5.0,
            "flagged_sentences": [{"sentence": "test", "issue": "issue", "source_id": "S1"}],
            "missing_sections": [],
            "coherence_issues": [],
            "grounding_score": 5.0,
        }
    ]
    state["revision_count"] = Config.MAX_REVISION_CYCLES
    result = should_revise(state)

    assert result == "formatter"


def test_create_graph_without_checkpointer():
    """Test graph creation without checkpointer."""
    graph = create_graph()

    assert graph is not None
    # Verify graph has the expected nodes
    # Note: LangGraph internals may not expose nodes directly, so we just check it compiles


def test_create_graph_with_checkpointer():
    """Test graph creation with SqliteSaver checkpointer."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with SqliteSaver.from_conn_string(tmp_path) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            assert graph is not None
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_graph_invoke_end_to_end():
    """Test full graph execution end-to-end."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    def _mock_section(section_name, **kwargs):
        return {"section_name": section_name, "content": "Content.", "citations": []}

    try:
        with SqliteSaver.from_conn_string(tmp_path) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            initial_state = get_initial_state("Test research question")
            config = {"configurable": {"thread_id": "test-thread"}}

            with patch("src.agents.graph.generate_outline", return_value="## Outline"), \
                 patch("src.agents.graph.generate_section", side_effect=_mock_section):

                # Execute graph — may pause at paper_approver or gap_selector
                result = graph.invoke(initial_state, config)

                # Resume paper_approver interrupt (keep all papers)
                if result.get("current_phase") != "formatting":
                    result = graph.invoke(Command(resume=""), config)

                # Resume gap_selector interrupt (select first gap)
                if result.get("current_phase") != "formatting":
                    result = graph.invoke(Command(resume="1"), config)

            assert result is not None
            assert result["research_question"] == "Test research question"
            assert result["current_phase"] == "formatting"

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
