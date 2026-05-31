"""
Unit tests for Phase 3: gap finding and gap selection (all external calls mocked).
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from typing import List

from src.utils.state import Paper, ResearchGap
from src.gap_finding.gap_analyzer import (
    find_research_gaps,
    _build_abstract_corpus,
    _evidence_id,
    _strip_markdown_fences,
    _clamp_novelty,
    _parse_gaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_paper(
    title: str = "Graph Coloring",
    abstract: str = "We study chromatic polynomials.",
    authors: List[str] = None,
    doi: str = "10.1234/test",
    source: str = "arxiv",
    year: int = 2024,
) -> Paper:
    return {
        "title": title,
        "abstract": abstract,
        "authors": authors or ["Alice", "Bob"],
        "doi": doi,
        "pdf_url": None,
        "source": source,
        "year": year,
        "relevance_score": 0.8,
    }


def make_gap(
    gap_title: str = "Open Chromatic Conjecture",
    description: str = "This gap is unexplored in bipartite graphs.",
    supporting_evidence: List[str] = None,
    novelty_score: float = 0.7,
) -> ResearchGap:
    return {
        "gap_title": gap_title,
        "description": description,
        "supporting_evidence": supporting_evidence or ["10.1234/test"],
        "novelty_score": novelty_score,
    }


# ---------------------------------------------------------------------------
# _evidence_id
# ---------------------------------------------------------------------------

class TestEvidenceId:
    def test_doi_present_returns_doi(self):
        paper = make_paper(doi="10.1234/abc")
        assert _evidence_id(paper) == "10.1234/abc"

    def test_empty_doi_returns_title(self):
        paper = make_paper(doi="", title="My Paper Title")
        assert _evidence_id(paper) == "My Paper Title"

    def test_none_doi_returns_title(self):
        paper = make_paper(title="Another Paper")
        paper["doi"] = None
        assert _evidence_id(paper) == "Another Paper"

    def test_missing_doi_key_returns_title(self):
        paper = make_paper(title="No DOI Paper")
        del paper["doi"]
        assert _evidence_id(paper) == "No DOI Paper"


# ---------------------------------------------------------------------------
# _build_abstract_corpus
# ---------------------------------------------------------------------------

class TestBuildAbstractCorpus:
    def test_includes_all_papers(self):
        papers = [make_paper(title="Paper A"), make_paper(title="Paper B", doi="10/b")]
        corpus = _build_abstract_corpus(papers)
        assert "Paper A" in corpus
        assert "Paper B" in corpus

    def test_formats_doi_correctly(self):
        paper = make_paper(doi="10.9999/xyz")
        corpus = _build_abstract_corpus([paper])
        assert "DOI: 10.9999/xyz" in corpus

    def test_missing_doi_shows_no_doi(self):
        paper = make_paper(doi="")
        corpus = _build_abstract_corpus([paper])
        assert "DOI: no DOI" in corpus

    def test_includes_abstract(self):
        paper = make_paper(abstract="This is the abstract text.")
        corpus = _build_abstract_corpus([paper])
        assert "This is the abstract text." in corpus

    def test_empty_authors_handled(self):
        paper = make_paper()
        paper["authors"] = []  # bypass the `or` default in make_paper
        corpus = _build_abstract_corpus([paper])
        assert "unknown" in corpus

    def test_none_abstract_handled(self):
        paper = make_paper(abstract="")
        paper["abstract"] = None
        corpus = _build_abstract_corpus([paper])
        assert "Abstract:" in corpus  # Should not raise

    def test_corpus_has_header_and_footer(self):
        corpus = _build_abstract_corpus([make_paper()])
        assert "=== PAPER CORPUS ===" in corpus
        assert "=== END CORPUS ===" in corpus

    def test_papers_numbered_sequentially(self):
        papers = [make_paper(title=f"P{i}") for i in range(3)]
        corpus = _build_abstract_corpus(papers)
        assert "[1]" in corpus
        assert "[2]" in corpus
        assert "[3]" in corpus


# ---------------------------------------------------------------------------
# _strip_markdown_fences
# ---------------------------------------------------------------------------

class TestStripMarkdownFences:
    def test_strips_json_fence(self):
        content = '```json\n[{"a": 1}]\n```'
        assert _strip_markdown_fences(content) == '[{"a": 1}]'

    def test_strips_plain_fence(self):
        content = '```\n[1, 2, 3]\n```'
        assert _strip_markdown_fences(content) == '[1, 2, 3]'

    def test_no_fences_unchanged(self):
        content = '[{"a": 1}]'
        assert _strip_markdown_fences(content) == '[{"a": 1}]'

    def test_multiline_json_preserved(self):
        content = '```json\n[\n  {"a": 1},\n  {"b": 2}\n]\n```'
        result = _strip_markdown_fences(content)
        assert '{"a": 1}' in result
        assert '{"b": 2}' in result

    def test_empty_string_unchanged(self):
        assert _strip_markdown_fences("") == ""


# ---------------------------------------------------------------------------
# _clamp_novelty
# ---------------------------------------------------------------------------

class TestClampNovelty:
    def test_valid_score_unchanged(self):
        assert _clamp_novelty(0.75) == pytest.approx(0.75)

    def test_clamps_above_one(self):
        assert _clamp_novelty(1.5) == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert _clamp_novelty(-0.3) == pytest.approx(0.0)

    def test_zero_is_valid(self):
        assert _clamp_novelty(0.0) == pytest.approx(0.0)

    def test_one_is_valid(self):
        assert _clamp_novelty(1.0) == pytest.approx(1.0)

    def test_non_numeric_string_defaults_to_zero(self):
        assert _clamp_novelty("high") == pytest.approx(0.0)

    def test_none_defaults_to_zero(self):
        assert _clamp_novelty(None) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _parse_gaps
# ---------------------------------------------------------------------------

class TestParseGaps:
    def _valid_json(self, overrides=None):
        gap = {
            "gap_title": "Open Conjecture",
            "description": "This is unexplored.",
            "supporting_evidence": ["10.1/a", "10.1/b"],
            "novelty_score": 0.8,
        }
        if overrides:
            gap.update(overrides)
        return json.dumps([gap])

    def test_parses_valid_json_array(self):
        gaps = _parse_gaps(self._valid_json())
        assert len(gaps) == 1
        assert gaps[0]["gap_title"] == "Open Conjecture"
        assert gaps[0]["novelty_score"] == pytest.approx(0.8)

    def test_clamps_novelty_above_one(self):
        gaps = _parse_gaps(self._valid_json({"novelty_score": 2.5}))
        assert gaps[0]["novelty_score"] == pytest.approx(1.0)

    def test_filters_empty_supporting_evidence_strings(self):
        gaps = _parse_gaps(self._valid_json({"supporting_evidence": ["10.1/a", "", None]}))
        # empty string and None are falsy — filtered out
        assert "" not in gaps[0]["supporting_evidence"]

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_gaps("not valid json")

    def test_raises_on_non_array_root(self):
        with pytest.raises(ValueError):
            _parse_gaps('{"gap_title": "x"}')

    def test_skips_non_dict_items(self):
        raw = json.dumps(["not a dict", {"gap_title": "G", "description": "D",
                                          "supporting_evidence": [], "novelty_score": 0.5}])
        gaps = _parse_gaps(raw)
        assert len(gaps) == 1
        assert gaps[0]["gap_title"] == "G"

    def test_empty_array_returns_empty_list(self):
        assert _parse_gaps("[]") == []

    def test_coerces_gap_title_to_string(self):
        gaps = _parse_gaps(self._valid_json({"gap_title": 42}))
        assert isinstance(gaps[0]["gap_title"], str)
        assert gaps[0]["gap_title"] == "42"


# ---------------------------------------------------------------------------
# find_research_gaps (mocked LLM)
# ---------------------------------------------------------------------------

class TestFindResearchGaps:
    _VALID_RESPONSE = json.dumps([
        {"gap_title": "Gap A", "description": "Desc A",
         "supporting_evidence": ["10.1/a"], "novelty_score": 0.9},
        {"gap_title": "Gap B", "description": "Desc B",
         "supporting_evidence": ["10.1/b"], "novelty_score": 0.6},
    ])

    def _mock_llm(self, mock_get_llm, content):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content=content)
        return mock_llm

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_happy_path_returns_gaps(self, mock_get_llm):
        mock_llm = self._mock_llm(mock_get_llm, self._VALID_RESPONSE)
        papers = [make_paper(doi="10.1/a"), make_paper(doi="10.1/b")]
        gaps = find_research_gaps(papers, "What are open problems?")
        assert len(gaps) == 2
        assert gaps[0]["gap_title"] == "Gap A"
        mock_llm.invoke.assert_called_once()

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_empty_papers_returns_empty_no_llm_call(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        result = find_research_gaps([], "Any question")
        assert result == []
        mock_llm.invoke.assert_not_called()

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_strips_markdown_fences_from_response(self, mock_get_llm):
        fenced = f"```json\n{self._VALID_RESPONSE}\n```"
        self._mock_llm(mock_get_llm, fenced)
        gaps = find_research_gaps([make_paper()], "question")
        assert len(gaps) == 2

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_handles_list_of_parts_response(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        # Simulate list-of-parts content
        mock_llm.invoke.return_value = MagicMock(
            content=[{"text": self._VALID_RESPONSE}]
        )
        gaps = find_research_gaps([make_paper()], "question")
        assert len(gaps) == 2

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_llm_exception_propagates(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.side_effect = RuntimeError("API failure")
        with pytest.raises(RuntimeError):
            find_research_gaps([make_paper()], "question")

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_bad_json_raises(self, mock_get_llm):
        self._mock_llm(mock_get_llm, "this is not json")
        with pytest.raises(json.JSONDecodeError):
            find_research_gaps([make_paper()], "question")

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_novelty_clamped_in_returned_gaps(self, mock_get_llm):
        raw = json.dumps([{"gap_title": "G", "description": "D",
                           "supporting_evidence": [], "novelty_score": 99.0}])
        self._mock_llm(mock_get_llm, raw)
        gaps = find_research_gaps([make_paper()], "question")
        assert gaps[0]["novelty_score"] == pytest.approx(1.0)

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_title_used_when_no_doi(self, mock_get_llm):
        """Corpus should include the paper title as the evidence ID when DOI is absent."""
        self._mock_llm(mock_get_llm, self._VALID_RESPONSE)
        paper = make_paper(title="Planar Graph Study", doi="")
        find_research_gaps([paper], "question")
        # Verify title appears in the prompt sent to the LLM
        prompt_arg = mock_get_llm.return_value.invoke.call_args[0][0]
        assert "Planar Graph Study" in prompt_arg

    @patch("src.gap_finding.gap_analyzer.get_llm")
    def test_uses_config_max_gaps(self, mock_get_llm):
        """Prompt should reference Config.MAX_GAPS."""
        from src.utils.config import Config
        self._mock_llm(mock_get_llm, self._VALID_RESPONSE)
        find_research_gaps([make_paper()], "question")
        prompt_arg = mock_get_llm.return_value.invoke.call_args[0][0]
        assert str(Config.MAX_GAPS) in prompt_arg


# ---------------------------------------------------------------------------
# gap_finder_node integration (mocked find_research_gaps)
# ---------------------------------------------------------------------------

class TestGapFinderNode:
    @patch("src.agents.graph.find_research_gaps")
    def test_happy_path_sets_gaps(self, mock_find):
        from src.agents.graph import gap_finder_node, get_initial_state
        mock_find.return_value = [make_gap("G1", novelty_score=0.5),
                                  make_gap("G2", novelty_score=0.9)]
        state = get_initial_state("Test question")
        state["retrieved_papers"] = [make_paper()]
        result = gap_finder_node(state)
        assert len(result["gaps"]) == 2
        assert result["current_phase"] == "gap_finding"

    @patch("src.agents.graph.find_research_gaps")
    def test_selected_gap_always_none_after_finder(self, mock_find):
        from src.agents.graph import gap_finder_node, get_initial_state
        mock_find.return_value = [make_gap(novelty_score=0.9)]
        state = get_initial_state("Test")
        state["retrieved_papers"] = [make_paper()]
        result = gap_finder_node(state)
        # Selection happens in gap_selector_node, not here
        assert result["selected_gap"] is None

    @patch("src.agents.graph.find_research_gaps")
    def test_no_papers_skips_llm_adds_error(self, mock_find):
        from src.agents.graph import gap_finder_node, get_initial_state
        state = get_initial_state("Test")
        state["retrieved_papers"] = []
        result = gap_finder_node(state)
        assert result["gaps"] == []
        assert result["selected_gap"] is None
        assert len(result["errors"]) > 0
        mock_find.assert_not_called()

    @patch("src.agents.graph.find_research_gaps")
    def test_llm_failure_adds_error_returns_empty_gaps(self, mock_find):
        from src.agents.graph import gap_finder_node, get_initial_state
        mock_find.side_effect = RuntimeError("LLM timeout")
        state = get_initial_state("Test")
        state["retrieved_papers"] = [make_paper()]
        result = gap_finder_node(state)
        assert result["gaps"] == []
        assert any("LLM error" in e for e in result["errors"])

    @patch("src.agents.graph.find_research_gaps")
    def test_phase_set_to_gap_finding(self, mock_find):
        from src.agents.graph import gap_finder_node, get_initial_state
        mock_find.return_value = []
        state = get_initial_state("Test")
        state["retrieved_papers"] = [make_paper()]
        result = gap_finder_node(state)
        assert result["current_phase"] == "gap_finding"

    @patch("src.agents.graph.find_research_gaps")
    def test_preserves_existing_errors(self, mock_find):
        from src.agents.graph import gap_finder_node, get_initial_state
        mock_find.return_value = []
        state = get_initial_state("Test")
        state["retrieved_papers"] = [make_paper()]
        state["errors"] = ["pre-existing error"]
        result = gap_finder_node(state)
        assert "pre-existing error" in result["errors"]


# ---------------------------------------------------------------------------
# gap_selector_node integration (mocked interrupt)
# ---------------------------------------------------------------------------

class TestGapSelectorNode:
    def _state_with_gaps(self, gaps):
        from src.agents.graph import get_initial_state
        state = get_initial_state("Test")
        state["gaps"] = gaps
        return state

    @patch("src.agents.graph.interrupt")
    def test_selects_by_one_based_index(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "2"
        gaps = [make_gap("Gap A", novelty_score=0.9),
                make_gap("Gap B", novelty_score=0.5),
                make_gap("Gap C", novelty_score=0.3)]
        state = self._state_with_gaps(gaps)
        result = gap_selector_node(state)
        # sorted order: Gap A (0.9), Gap B (0.5), Gap C (0.3) → index 2 = Gap B
        assert result["selected_gap"]["gap_title"] == "Gap B"

    @patch("src.agents.graph.interrupt")
    def test_selects_by_gap_title_string(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "Gap C"
        gaps = [make_gap("Gap A", novelty_score=0.9),
                make_gap("Gap B", novelty_score=0.5),
                make_gap("Gap C", novelty_score=0.3)]
        state = self._state_with_gaps(gaps)
        result = gap_selector_node(state)
        assert result["selected_gap"]["gap_title"] == "Gap C"

    @patch("src.agents.graph.interrupt")
    def test_title_match_is_case_insensitive(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "gap a"
        gaps = [make_gap("Gap A", novelty_score=0.9)]
        result = gap_selector_node(self._state_with_gaps(gaps))
        assert result["selected_gap"]["gap_title"] == "Gap A"

    @patch("src.agents.graph.interrupt")
    def test_invalid_input_falls_back_to_highest_novelty(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "99"  # out of range
        gaps = [make_gap("Gap A", novelty_score=0.4),
                make_gap("Gap B", novelty_score=0.9)]
        result = gap_selector_node(self._state_with_gaps(gaps))
        # Fallback: highest novelty after sorting = Gap B
        assert result["selected_gap"]["gap_title"] == "Gap B"

    @patch("src.agents.graph.interrupt")
    def test_empty_gaps_returns_none_selected_gap(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        state = self._state_with_gaps([])
        result = gap_selector_node(state)
        assert result["selected_gap"] is None
        assert result["current_phase"] == "gap_selection"
        mock_interrupt.assert_not_called()

    @patch("src.agents.graph.interrupt")
    def test_human_decision_recorded(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "1"
        gaps = [make_gap("Gap A", novelty_score=0.9)]
        result = gap_selector_node(self._state_with_gaps(gaps))
        assert len(result["human_decisions"]) == 1
        decision = result["human_decisions"][0]
        assert decision["checkpoint_name"] == "gap_selection"
        assert decision["selected_gap"]["gap_title"] == "Gap A"
        assert "timestamp" in decision

    @patch("src.agents.graph.interrupt")
    def test_phase_set_to_gap_selection(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "1"
        result = gap_selector_node(self._state_with_gaps([make_gap()]))
        assert result["current_phase"] == "gap_selection"

    @patch("src.agents.graph.interrupt")
    def test_single_gap_auto_selected_on_index_one(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "1"
        result = gap_selector_node(self._state_with_gaps([make_gap("Only Gap")]))
        assert result["selected_gap"]["gap_title"] == "Only Gap"

    @patch("src.agents.graph.interrupt")
    def test_interrupt_payload_contains_gap_list(self, mock_interrupt):
        from src.agents.graph import gap_selector_node
        mock_interrupt.return_value = "1"
        gaps = [make_gap("Gap A", novelty_score=0.9), make_gap("Gap B", novelty_score=0.5)]
        gap_selector_node(self._state_with_gaps(gaps))
        payload = mock_interrupt.call_args[0][0]
        assert "gaps" in payload
        assert len(payload["gaps"]) == 2
        # Gaps should be sorted by novelty descending
        assert payload["gaps"][0]["novelty_score"] >= payload["gaps"][1]["novelty_score"]
