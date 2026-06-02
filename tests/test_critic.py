"""
Tests for Phase 4d critic node helpers.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.critic import (
    _check_grounding,
    _extract_citation_sentences,
    _score_coherence,
)
from src.agents.graph import critic_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(content: str):
    m = MagicMock()
    m.invoke.return_value = MagicMock(content=content)
    return m


def _base_state(**overrides):
    state = {
        "research_question": "Test question",
        "sub_queries": [],
        "retrieved_papers": [],
        "embeddings_ready": False,
        "gaps": [],
        "selected_gap": None,
        "outline": "1. Abstract\n2. Introduction\n3. Methodology",
        "draft_sections": [
            {
                "section_name": "abstract",
                "content": "Graph theory is broad [S1]. It covers many topics [S2].",
                "citations": ["S1", "S2"],
            }
        ],
        "critic_feedback": [],
        "revision_count": 0,
        "eval_scores": None,
        "citation_report": {
            "S1": {"abstract": "Graph theory is a branch of mathematics.", "title": "Graph Theory"},
            "S2": {"abstract": "Networks and graphs have many applications.", "title": "Networks"},
        },
        "human_decisions": [],
        "current_phase": "draft_review",
        "errors": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _extract_citation_sentences
# ---------------------------------------------------------------------------

def test_extract_single_citation():
    sections = [{"section_name": "abstract", "content": "Graphs are useful [S1].", "citations": []}]
    result = _extract_citation_sentences(sections)
    assert len(result) == 1
    sentence, source_id, section_name = result[0]
    assert source_id == "S1"
    assert "Graphs are useful" in sentence
    assert section_name == "abstract"


def test_extract_two_citations_same_sentence():
    sections = [{"section_name": "intro", "content": "Foo [S1] and bar [S2].", "citations": []}]
    result = _extract_citation_sentences(sections)
    assert len(result) == 2
    assert result[0][0] == result[1][0]
    assert {r[1] for r in result} == {"S1", "S2"}


def test_extract_no_citations():
    sections = [{"section_name": "abstract", "content": "No citations here.", "citations": []}]
    assert _extract_citation_sentences(sections) == []


def test_extract_multiple_sections_carry_name():
    sections = [
        {"section_name": "abstract", "content": "Claim [S1].", "citations": []},
        {"section_name": "intro", "content": "Another [S2].", "citations": []},
    ]
    result = _extract_citation_sentences(sections)
    names = {r[2] for r in result}
    assert names == {"abstract", "intro"}


def test_extract_empty_content_no_crash():
    sections = [{"section_name": "abstract", "content": "", "citations": []}]
    assert _extract_citation_sentences(sections) == []


# ---------------------------------------------------------------------------
# _check_grounding
# ---------------------------------------------------------------------------

def test_grounding_supported_claim():
    pairs = [("Graph theory is broad.", "S1", "abstract")]
    citation_report = {"S1": {"abstract": "Graph theory is a branch of mathematics."}}
    llm = _make_llm("YES\nThe abstract confirms the claim.")
    flagged, score = _check_grounding(pairs, citation_report, llm)
    assert flagged == []
    assert score == 10.0


def test_grounding_unsupported_claim_flagged():
    pairs = [("Graphs solve NP-hard problems in polynomial time.", "S1", "abstract")]
    citation_report = {"S1": {"abstract": "Graph theory is a branch of mathematics."}}
    llm = _make_llm("NO\nThe abstract does not mention polynomial time.")
    flagged, score = _check_grounding(pairs, citation_report, llm)
    assert len(flagged) == 1
    assert flagged[0]["source_id"] == "S1"
    assert "sentence" in flagged[0]
    assert "issue" in flagged[0]
    assert score == 0.0


def test_grounding_missing_source_id_skipped():
    pairs = [("Claim.", "S99", "abstract")]
    citation_report = {}
    llm = _make_llm("YES\nReason.")
    flagged, score = _check_grounding(pairs, citation_report, llm)
    assert flagged == []
    assert score == 10.0
    llm.invoke.assert_not_called()


def test_grounding_empty_abstract_skipped():
    pairs = [("Claim.", "S1", "abstract")]
    citation_report = {"S1": {"abstract": ""}}
    llm = _make_llm("YES\nReason.")
    flagged, score = _check_grounding(pairs, citation_report, llm)
    assert flagged == []
    assert score == 10.0
    llm.invoke.assert_not_called()


def test_grounding_partial_score():
    pairs = [
        ("Claim A.", "S1", "abstract"),
        ("Claim B.", "S2", "abstract"),
    ]
    citation_report = {
        "S1": {"abstract": "Abstract A."},
        "S2": {"abstract": "Abstract B."},
    }
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content="YES\nSupported."),
        MagicMock(content="NO\nNot supported."),
    ]
    flagged, score = _check_grounding(pairs, citation_report, llm)
    assert len(flagged) == 1
    assert score == 5.0


def test_grounding_case_insensitive_yes():
    pairs = [("Claim.", "S1", "abstract")]
    citation_report = {"S1": {"abstract": "Some abstract."}}
    llm = _make_llm("yes\nSupports the claim.")
    flagged, score = _check_grounding(pairs, citation_report, llm)
    assert flagged == []
    assert score == 10.0


# ---------------------------------------------------------------------------
# _score_coherence
# ---------------------------------------------------------------------------

def test_coherence_valid_json():
    llm = _make_llm('{"score": 8.5, "missing_sections": ["conclusion"], "coherence_issues": ["flow gap"]}')
    score, missing, issues = _score_coherence("outline", [], llm)
    assert score == 8.5
    assert missing == ["conclusion"]
    assert issues == ["flow gap"]


def test_coherence_clamps_above_10():
    llm = _make_llm('{"score": 12.0, "missing_sections": [], "coherence_issues": []}')
    score, _, _ = _score_coherence("outline", [], llm)
    assert score == 10.0


def test_coherence_clamps_below_0():
    llm = _make_llm('{"score": -3.0, "missing_sections": [], "coherence_issues": []}')
    score, _, _ = _score_coherence("outline", [], llm)
    assert score == 0.0


def test_coherence_invalid_json_fallback():
    llm = _make_llm("The paper scores score: 6.5 overall but has some issues.")
    score, missing, issues = _score_coherence("outline", [], llm)
    assert score == 6.5
    assert missing == []
    assert issues == []


def test_coherence_total_parse_failure_defaults():
    llm = _make_llm("Completely unparseable response with no numbers at all!!")
    score, missing, issues = _score_coherence("outline", [], llm)
    assert score == 5.0
    assert missing == []
    assert issues == []


def test_coherence_outline_in_prompt():
    llm = _make_llm('{"score": 7.0, "missing_sections": [], "coherence_issues": []}')
    _score_coherence("MY SPECIAL OUTLINE", [], llm)
    call_content = llm.invoke.call_args[0][0][0].content
    assert "MY SPECIAL OUTLINE" in call_content


# ---------------------------------------------------------------------------
# critic_node (integration)
# ---------------------------------------------------------------------------

def _patch_helpers(coherence_response, grounding_response="YES\nSupported."):
    """Patch get_llm to return a mock that gives different responses per call count."""
    call_count = [0]

    def side_effect(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock(content=grounding_response)
        return MagicMock(content=coherence_response)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = side_effect
    return mock_llm


def test_critic_node_appends_feedback():
    prior = {"score": 6.0, "flagged_sentences": [], "missing_sections": [], "coherence_issues": [], "grounding_score": 6.0}
    state = _base_state(critic_feedback=[prior])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 8.0, "missing_sections": [], "coherence_issues": []}')
    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = critic_node(state)
    assert len(result["critic_feedback"]) == 2
    assert result["critic_feedback"][0] == prior


def test_critic_node_increments_revision_count():
    state = _base_state(revision_count=2, draft_sections=[])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 7.0, "missing_sections": [], "coherence_issues": []}')
    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = critic_node(state)
    assert result["revision_count"] == 3


def test_critic_node_score_formula():
    # coherence=8, grounding=10 (no citations to check) => 0.6*8 + 0.4*10 = 8.8
    state = _base_state(draft_sections=[{"section_name": "abstract", "content": "No citations.", "citations": []}])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 8.0, "missing_sections": [], "coherence_issues": []}')
    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = critic_node(state)
    assert result["critic_feedback"][-1]["score"] == pytest.approx(8.8)


def test_critic_node_phase():
    state = _base_state(draft_sections=[])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 7.0, "missing_sections": [], "coherence_issues": []}')
    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = critic_node(state)
    assert result["current_phase"] == "critiquing"


def test_critic_node_llm_exception_neutral_scores():
    state = _base_state()
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("API failure")
    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = critic_node(state)
    feedback = result["critic_feedback"][-1]
    assert feedback["score"] == pytest.approx(5.0)
    assert len(result["errors"]) >= 1
    assert "Critic LLM error" in result["errors"][-1]


def test_critic_node_no_sections_no_crash():
    state = _base_state(draft_sections=[])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 5.0, "missing_sections": [], "coherence_issues": []}')
    with patch("src.agents.graph.get_llm", return_value=mock_llm):
        result = critic_node(state)
    assert result["current_phase"] == "critiquing"
    assert len(result["critic_feedback"]) == 1
