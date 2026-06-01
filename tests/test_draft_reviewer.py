"""
Tests for draft_reviewer_node (Phase 4c).
"""

import json
from unittest.mock import patch

import pytest

from src.agents.graph import draft_reviewer_node


def _base_state(**overrides):
    state = {
        "research_question": "Test question",
        "sub_queries": [],
        "retrieved_papers": [],
        "embeddings_ready": False,
        "gaps": [],
        "selected_gap": None,
        "outline": "1. Abstract\n2. Introduction\n3. Related Work",
        "draft_sections": [
            {"section_name": "abstract", "content": "Original abstract text.", "citations": ["S1"]},
            {"section_name": "introduction", "content": "Original introduction text.", "citations": ["S2"]},
        ],
        "critic_feedback": [],
        "revision_count": 0,
        "eval_scores": None,
        "citation_report": {},
        "human_decisions": [],
        "current_phase": "writing",
        "errors": [],
    }
    state.update(overrides)
    return state


def _invoke_reviewer(state, resume_value):
    """Patch interrupt() to return resume_value, then call the node."""
    with patch("src.agents.graph.interrupt", return_value=resume_value):
        return draft_reviewer_node(state)


# ---------------------------------------------------------------------------
# Empty sections — node should pass through without calling interrupt
# ---------------------------------------------------------------------------

def test_empty_sections_passes_through():
    state = _base_state(draft_sections=[])
    result = draft_reviewer_node(state)
    assert result["current_phase"] == "draft_review"
    assert result["draft_sections"] == []
    assert result["human_decisions"] == []


# ---------------------------------------------------------------------------
# Empty user input — accept all, no edits applied
# ---------------------------------------------------------------------------

def test_empty_input_no_edits():
    state = _base_state()
    result = _invoke_reviewer(state, "")
    assert result["draft_sections"][0]["content"] == "Original abstract text."
    assert result["draft_sections"][1]["content"] == "Original introduction text."


def test_empty_input_records_decision():
    state = _base_state()
    result = _invoke_reviewer(state, "")
    assert len(result["human_decisions"]) == 1
    decision = result["human_decisions"][0]
    assert decision["checkpoint_name"] == "draft_review"
    assert decision["edited_sections"] == {}
    assert decision["notes"] == ""


# ---------------------------------------------------------------------------
# Valid JSON edit — only named section updated
# ---------------------------------------------------------------------------

def test_valid_json_applies_edit():
    state = _base_state()
    edits = json.dumps({"abstract": "Revised abstract content."})
    result = _invoke_reviewer(state, edits)
    sections = {s["section_name"]: s for s in result["draft_sections"]}
    assert sections["abstract"]["content"] == "Revised abstract content."
    assert sections["introduction"]["content"] == "Original introduction text."


def test_valid_json_edit_records_decision():
    state = _base_state()
    edits = json.dumps({"abstract": "New text."})
    result = _invoke_reviewer(state, edits)
    decision = result["human_decisions"][0]
    assert decision["edited_sections"] == {"abstract": "New text."}


def test_edit_multiple_sections():
    state = _base_state()
    edits = json.dumps({
        "abstract": "New abstract.",
        "introduction": "New intro.",
    })
    result = _invoke_reviewer(state, edits)
    sections = {s["section_name"]: s for s in result["draft_sections"]}
    assert sections["abstract"]["content"] == "New abstract."
    assert sections["introduction"]["content"] == "New intro."


def test_edit_unknown_section_ignored():
    state = _base_state()
    edits = json.dumps({"nonexistent_section": "Some text."})
    result = _invoke_reviewer(state, edits)
    # Existing sections unchanged
    assert result["draft_sections"][0]["content"] == "Original abstract text."
    assert result["draft_sections"][1]["content"] == "Original introduction text."


# ---------------------------------------------------------------------------
# Malformed JSON — graceful degradation, no edits applied
# ---------------------------------------------------------------------------

def test_malformed_json_no_crash():
    state = _base_state()
    result = _invoke_reviewer(state, "not valid json {{{")
    assert result["draft_sections"][0]["content"] == "Original abstract text."
    assert result["human_decisions"][0]["edited_sections"] == {}


def test_non_dict_json_no_edits():
    state = _base_state()
    result = _invoke_reviewer(state, json.dumps(["abstract", "introduction"]))
    assert result["draft_sections"][0]["content"] == "Original abstract text."
    assert result["human_decisions"][0]["edited_sections"] == {}


# ---------------------------------------------------------------------------
# Citations preserved after edit
# ---------------------------------------------------------------------------

def test_citations_preserved_after_edit():
    state = _base_state()
    edits = json.dumps({"abstract": "Completely new abstract."})
    result = _invoke_reviewer(state, edits)
    abstract = next(s for s in result["draft_sections"] if s["section_name"] == "abstract")
    assert abstract["citations"] == ["S1"]


# ---------------------------------------------------------------------------
# Phase and existing human_decisions
# ---------------------------------------------------------------------------

def test_current_phase_set():
    state = _base_state()
    result = _invoke_reviewer(state, "")
    assert result["current_phase"] == "draft_review"


def test_appends_to_existing_decisions():
    prior = {
        "checkpoint_name": "gap_selection",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "approved_papers": None,
        "removed_papers": None,
        "selected_gap": None,
        "edited_sections": None,
        "notes": "1",
    }
    state = _base_state(human_decisions=[prior])
    result = _invoke_reviewer(state, "")
    assert len(result["human_decisions"]) == 2
    assert result["human_decisions"][0]["checkpoint_name"] == "gap_selection"
    assert result["human_decisions"][1]["checkpoint_name"] == "draft_review"
