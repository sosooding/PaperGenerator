"""
Unit tests for state schema.
"""

import pytest
from src.utils.state import (
    AgentState,
    Paper,
    ResearchGap,
    DraftSection,
    CriticFeedback,
    EvalScores,
    HumanDecision,
)
from src.agents.graph import get_initial_state


def test_initial_state():
    """Test that initial state has correct structure."""
    research_question = "Test question about graph theory?"
    state = get_initial_state(research_question)

    assert state["research_question"] == research_question
    assert state["sub_queries"] == []
    assert state["retrieved_papers"] == []
    assert state["embeddings_ready"] is False
    assert state["gaps"] == []
    assert state["selected_gap"] is None
    assert state["outline"] is None
    assert state["draft_sections"] == []
    assert state["critic_feedback"] == []
    assert state["revision_count"] == 0
    assert state["eval_scores"] is None
    assert state["citation_report"] == {}
    assert state["human_decisions"] == []
    assert state["current_phase"] == "initialized"
    assert state["errors"] == []


def test_paper_structure():
    """Test Paper TypedDict structure."""
    paper: Paper = {
        "title": "Graph Coloring Algorithms",
        "abstract": "This paper discusses...",
        "authors": ["Author A", "Author B"],
        "doi": "10.1234/example",
        "pdf_url": "https://arxiv.org/pdf/1234.5678",
        "source": "arxiv",
        "year": 2024,
        "relevance_score": 0.95,
    }

    assert paper["title"] == "Graph Coloring Algorithms"
    assert len(paper["authors"]) == 2
    assert paper["source"] in ["arxiv", "semantic_scholar"]


def test_research_gap_structure():
    """Test ResearchGap TypedDict structure."""
    gap: ResearchGap = {
        "gap_title": "Unexplored chromatic properties",
        "description": "Limited research on...",
        "supporting_evidence": ["10.1234/paper1", "10.1234/paper2"],
        "novelty_score": 0.85,
    }

    assert gap["gap_title"] == "Unexplored chromatic properties"
    assert len(gap["supporting_evidence"]) == 2
    assert 0 <= gap["novelty_score"] <= 1


def test_draft_section_structure():
    """Test DraftSection TypedDict structure."""
    section: DraftSection = {
        "section_name": "introduction",
        "content": "Graph theory is...",
        "citations": ["SOURCE_1", "SOURCE_2"],
    }

    assert section["section_name"] == "introduction"
    assert len(section["citations"]) == 2


def test_critic_feedback_structure():
    """Test CriticFeedback TypedDict structure."""
    feedback: CriticFeedback = {
        "score": 7.5,
        "flagged_sentences": [
            {"sentence": "Test sentence", "issue": "Weak grounding", "source_id": "SOURCE_1"}
        ],
        "missing_sections": [],
        "coherence_issues": ["Flow between intro and related work"],
        "grounding_score": 8.0,
    }

    assert feedback["score"] == 7.5
    assert len(feedback["flagged_sentences"]) == 1
    assert feedback["grounding_score"] == 8.0


def test_eval_scores_structure():
    """Test EvalScores TypedDict structure."""
    scores: EvalScores = {
        "nli_distribution": {"entailment": 80, "neutral": 15, "contradiction": 5},
        "bert_scores": {"abstract": 0.85, "introduction": 0.82},
        "hallucinated_citations": ["SOURCE_X"],
        "llm_judge_scores": {
            "grounding": 2.5,
            "novelty": 3.0,
            "completeness": 2.0,
            "coherence": 2.5,
            "tone": 3.0,
        },
        "structural_completeness": {"abstract": True, "introduction": True},
    }

    assert sum(scores["nli_distribution"].values()) == 100
    assert len(scores["llm_judge_scores"]) == 5
    assert all(0 <= v <= 3 for v in scores["llm_judge_scores"].values())


def test_human_decision_structure():
    """Test HumanDecision TypedDict structure."""
    decision: HumanDecision = {
        "checkpoint_name": "checkpoint_1",
        "timestamp": "2024-01-15T10:30:00",
        "approved_papers": ["10.1234/paper1"],
        "removed_papers": ["10.1234/paper2"],
        "selected_gap": None,
        "edited_sections": None,
        "notes": "Approved papers look relevant",
    }

    assert decision["checkpoint_name"] == "checkpoint_1"
    assert len(decision["approved_papers"]) == 1
    assert decision["notes"] is not None
