"""
Unit tests for LangGraph workflow.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agents.graph import (
    create_graph,
    get_initial_state,
    planner_node,
    retriever_node,
    paper_approver_node,
    gap_finder_node,
    gap_selector_node,
    writer_node,
    draft_reviewer_node,
    critic_node,
    formatter_node,
    should_revise,
)
from src.utils.state import AgentState
from src.utils.config import Config


def make_paper(title="Paper 1", doi="10.1/p1", year=2024, score=0.9):
    return {
        "title": title, "abstract": "Abstract.", "authors": ["Author"],
        "doi": doi, "pdf_url": None, "source": "arxiv",
        "year": year, "relevance_score": score,
    }


def make_gap(title="Test Gap", novelty=0.9):
    return {
        "gap_title": title, "description": "A gap.",
        "supporting_evidence": [], "novelty_score": novelty,
    }


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_get_initial_state():
    question = "What is graph coloring?"
    state = get_initial_state(question)

    assert isinstance(state, dict)
    assert state["research_question"] == question
    assert state["revision_count"] == 0
    assert state["current_phase"] == "initialized"


# ---------------------------------------------------------------------------
# planner_node (TEST-1: hermetic via get_llm mock)
# ---------------------------------------------------------------------------

def test_planner_node():
    initial_state = get_initial_state("Test question")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["sub-query 1", "sub-query 2"]')

    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = planner_node(initial_state)

    assert result["current_phase"] == "planning"
    assert result["research_question"] == "Test question"
    assert result["sub_queries"] == ["sub-query 1", "sub-query 2"]


def test_planner_node_fallback_on_bad_json():
    initial_state = get_initial_state("Fallback question")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json")

    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = planner_node(initial_state)

    assert result["sub_queries"] == ["Fallback question"]
    assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# retriever_node (TEST-1: hermetic via fetch + VectorStore mocks)
# ---------------------------------------------------------------------------

def test_retriever_node():
    initial_state = get_initial_state("Test question")
    initial_state["sub_queries"] = ["query1", "query2"]

    papers = [make_paper("P1", "10.1/p1"), make_paper("P2", "10.1/p2")]
    mock_vs = MagicMock()
    mock_vs.collection.count.return_value = len(papers)
    mock_vs.get_relevant_papers.return_value = papers

    with patch("src.agents.graph.fetch_papers_for_queries", return_value=papers), \
         patch("src.agents.graph.VectorStore", return_value=mock_vs):
        result = retriever_node(initial_state)

    assert result["current_phase"] == "retrieval"
    assert result["embeddings_ready"] is True
    assert len(result["retrieved_papers"]) == len(papers)


def test_retriever_node_no_sub_queries():
    state = get_initial_state("Test question")
    result = retriever_node(state)  # no live calls; sub_queries is empty → early return
    assert result["current_phase"] == "retrieval"
    assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# paper_approver_node
# ---------------------------------------------------------------------------

def test_paper_approver_node_no_papers():
    state = get_initial_state("Test question")
    result = paper_approver_node(state)

    assert result["current_phase"] == "paper_approval"
    assert result["retrieved_papers"] == []
    assert result["human_decisions"] == []


def test_paper_approver_node_keep_all():
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
    assert decision["removed_papers"] == []  # real empty list (TD-2: not None)
    mock_vs_cls.assert_not_called()


def test_paper_approver_node_remove_papers():
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


# ---------------------------------------------------------------------------
# gap_finder_node (TEST-1: hermetic via find_research_gaps mock)
# ---------------------------------------------------------------------------

def test_gap_finder_node():
    initial_state = get_initial_state("Test question")
    initial_state["retrieved_papers"] = [make_paper("Paper 1", "10.1234/test")]

    fake_gaps = [make_gap("Test Gap")]

    with patch("src.agents.graph.find_research_gaps", return_value=fake_gaps):
        result = gap_finder_node(initial_state)

    assert result["current_phase"] == "gap_finding"
    assert len(result["gaps"]) == 1


def test_gap_finder_node_no_papers():
    state = get_initial_state("Test question")
    result = gap_finder_node(state)  # no papers → early return without live call
    assert result["current_phase"] == "gap_finding"
    assert result["gaps"] == []
    assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# gap_selector_node — FIX-3: edited_sections must be None, not []
# ---------------------------------------------------------------------------

def test_gap_selector_node_decision_fields():
    state = get_initial_state("Test question")
    state["gaps"] = [make_gap("G1"), make_gap("G2", 0.7)]

    with patch("src.agents.graph.interrupt", return_value="1"):
        result = gap_selector_node(state)

    decision = result["human_decisions"][0]
    assert decision["checkpoint_name"] == "gap_selection"
    assert decision["approved_papers"] is None
    assert decision["removed_papers"] is None
    assert decision["edited_sections"] is None
    assert decision["selected_gap"]["gap_title"] == "G1"


# ---------------------------------------------------------------------------
# writer_node
# ---------------------------------------------------------------------------

def test_writer_node():
    initial_state = get_initial_state("Test question")
    initial_state["selected_gap"] = make_gap()
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
    initial_state = get_initial_state("Test question")
    result = writer_node(initial_state)

    assert result["current_phase"] == "writing"
    assert any("no selected_gap" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# critic_node / formatter_node
# ---------------------------------------------------------------------------

def test_critic_node():
    initial_state = get_initial_state("Test question")
    initial_state["draft_sections"] = [
        {"section_name": "abstract", "content": "Test content", "citations": []}
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"score": 7.0, "missing_sections": [], "coherence_issues": []}'
    )
    with patch("src.agents.critic.get_llm", return_value=mock_llm):
        result = critic_node(initial_state)

    assert result["current_phase"] == "critiquing"
    assert result["revision_count"] == 1


def test_formatter_node():
    initial_state = get_initial_state("Test question")
    result = formatter_node(initial_state)

    assert result["current_phase"] == "formatting"


# ---------------------------------------------------------------------------
# should_revise
# ---------------------------------------------------------------------------

def test_should_revise_no_feedback():
    state = get_initial_state("Test question")
    assert should_revise(state) == "formatter"


def test_should_revise_good_score():
    state = get_initial_state("Test question")
    state["critic_feedback"] = [{
        "score": 8.0, "flagged_sentences": [],
        "missing_sections": [], "coherence_issues": [], "grounding_score": 8.0,
    }]
    assert should_revise(state) == "formatter"


def test_should_revise_low_score():
    state = get_initial_state("Test question")
    state["critic_feedback"] = [{
        "score": 5.0,
        "flagged_sentences": [{"sentence": "test", "issue": "issue", "source_id": "S1"}],
        "missing_sections": [], "coherence_issues": [], "grounding_score": 5.0,
    }]
    state["revision_count"] = 0
    assert should_revise(state) == "writer"


def test_should_revise_max_revisions():
    state = get_initial_state("Test question")
    state["critic_feedback"] = [{
        "score": 5.0,
        "flagged_sentences": [{"sentence": "test", "issue": "issue", "source_id": "S1"}],
        "missing_sections": [], "coherence_issues": [], "grounding_score": 5.0,
    }]
    state["revision_count"] = Config.MAX_REVISION_CYCLES
    assert should_revise(state) == "formatter"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_create_graph_without_checkpointer():
    graph = create_graph()
    assert graph is not None


def test_create_graph_with_checkpointer():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with SqliteSaver.from_conn_string(tmp_path) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            assert graph is not None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# End-to-end (FIX-2 + TEST-1: fully mocked, 3 interrupt resumes)
# ---------------------------------------------------------------------------

def test_graph_invoke_end_to_end():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    fake_paper = make_paper("Test Paper", "10.1/test")
    fake_gap = make_gap("Test Gap")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["sub-query 1"]')

    mock_vs = MagicMock()
    mock_vs.collection.count.return_value = 1
    mock_vs.get_relevant_papers.return_value = [fake_paper]
    mock_vs.similarity_search.return_value = [fake_paper]

    def _mock_section(section_name, **kwargs):
        return {"section_name": section_name, "content": "Content.", "citations": []}

    try:
        with SqliteSaver.from_conn_string(tmp_path) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            initial_state = get_initial_state("Test research question")
            config = {"configurable": {"thread_id": "test-thread"}}

            with patch("src.agents.graph.get_llm", return_value=mock_llm), \
                 patch("src.agents.graph.fetch_papers_for_queries", return_value=[fake_paper]), \
                 patch("src.agents.graph.VectorStore", return_value=mock_vs), \
                 patch("src.agents.graph.find_research_gaps", return_value=[fake_gap]), \
                 patch("src.agents.graph.generate_outline", return_value="## Outline"), \
                 patch("src.agents.graph.generate_section", side_effect=_mock_section):

                result = graph.invoke(initial_state, config)

                # FIX-2: resume all 3 interrupts — paper_approver, gap_selector, draft_reviewer
                resume_values = ("", "1", "")
                for resume_value in resume_values:
                    if not result.get("__interrupt__"):
                        break
                    result = graph.invoke(Command(resume=resume_value), config)

            assert result is not None
            assert result["research_question"] == "Test research question"
            assert result["current_phase"] == "formatting"

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
