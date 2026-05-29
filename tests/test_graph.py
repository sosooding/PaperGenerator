"""
Unit tests for LangGraph workflow.
"""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
import tempfile
import os

from src.agents.graph import (
    create_graph,
    get_initial_state,
    planner_node,
    retriever_node,
    gap_finder_node,
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
    """Test writer node updates state correctly."""
    initial_state = get_initial_state("Test question")
    initial_state["selected_gap"] = {
        "gap_title": "Test gap",
        "description": "Description",
        "supporting_evidence": [],
        "novelty_score": 0.8,
    }
    result = writer_node(initial_state)

    assert result["current_phase"] == "writing"


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
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with SqliteSaver.from_conn_string(tmp_path) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            initial_state = get_initial_state("Test research question")

            config = {"configurable": {"thread_id": "test-thread"}}

            # Execute graph
            result = graph.invoke(initial_state, config)

            # Verify result
            assert result is not None
            assert result["research_question"] == "Test research question"
            assert result["current_phase"] == "formatting"  # Should reach the end

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
