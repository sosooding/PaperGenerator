"""
Unit tests for the writer module (Phase 4b).
"""

import pytest
from unittest.mock import MagicMock

from src.writing.writer import (
    SECTIONS,
    build_source_map,
    generate_outline,
    generate_section,
    _extract_citations,
    _format_rag_context,
    _format_source_map_prompt,
    _strip_markdown_fences,
)


def _make_paper(title="Paper", doi="10.1/p", year=2023, abstract="An abstract."):
    return {
        "title": title,
        "abstract": abstract,
        "authors": ["First Author", "Second Author"],
        "doi": doi,
        "pdf_url": None,
        "source": "arxiv",
        "year": year,
        "relevance_score": 0.9,
    }


def _make_gap():
    return {
        "gap_title": "Test Gap",
        "description": "Gap description here.",
        "supporting_evidence": [],
        "novelty_score": 0.7,
    }


def _make_llm(content="Section content."):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


class TestSections:
    def test_six_sections(self):
        assert len(SECTIONS) == 6

    def test_expected_names(self):
        assert SECTIONS == [
            "abstract", "introduction", "related_work",
            "gap_analysis", "proposed_approach", "conclusion",
        ]


class TestBuildSourceMap:
    def test_assigns_sequential_ids(self):
        papers = [_make_paper("P1", "10.1/a"), _make_paper("P2", "10.1/b")]
        sm = build_source_map(papers)
        assert list(sm.keys()) == ["S1", "S2"]
        assert sm["S1"]["title"] == "P1"
        assert sm["S2"]["title"] == "P2"

    def test_empty_papers(self):
        assert build_source_map([]) == {}

    def test_single_paper(self):
        sm = build_source_map([_make_paper("Only")])
        assert list(sm.keys()) == ["S1"]

    def test_preserves_paper_data(self):
        paper = _make_paper("Graph Coloring", "10.1/gc", 2021)
        sm = build_source_map([paper])
        assert sm["S1"] is paper


class TestFormatSourceMapPrompt:
    def test_includes_sid_and_title(self):
        sm = {"S1": _make_paper("Graph Coloring", "10.1/gc", 2021)}
        result = _format_source_map_prompt(sm)
        assert "[S1]" in result
        assert "Graph Coloring" in result
        assert "2021" in result

    def test_includes_doi_when_present(self):
        sm = {"S1": _make_paper("Paper", "10.1/doi")}
        result = _format_source_map_prompt(sm)
        assert "10.1/doi" in result

    def test_uses_last_name_of_first_author(self):
        sm = {"S1": _make_paper("T")}
        result = _format_source_map_prompt(sm)
        assert "Author et al." in result

    def test_no_doi_omits_doi_field(self):
        paper = _make_paper("T", doi=None)
        sm = {"S1": paper}
        result = _format_source_map_prompt(sm)
        assert "DOI" not in result

    def test_empty_map_returns_empty_string(self):
        assert _format_source_map_prompt({}) == ""

    def test_multiple_sources_one_per_line(self):
        sm = {"S1": _make_paper("P1"), "S2": _make_paper("P2")}
        lines = _format_source_map_prompt(sm).splitlines()
        assert len(lines) == 2


class TestFormatRagContext:
    def test_empty_returns_sentinel(self):
        result = _format_rag_context([])
        assert "No additional context" in result

    def test_includes_title(self):
        result = _format_rag_context([_make_paper("Title X")])
        assert "Title X" in result

    def test_includes_abstract(self):
        result = _format_rag_context([_make_paper(abstract="Specific abstract text.")])
        assert "Specific abstract text." in result

    def test_truncates_long_abstract(self):
        paper = _make_paper(abstract="word " * 300)
        result = _format_rag_context([paper])
        # Abstract is capped at 500 chars; full result should be manageable
        assert len(result) < 3000

    def test_multiple_papers(self):
        papers = [_make_paper("P1"), _make_paper("P2")]
        result = _format_rag_context(papers)
        assert "P1" in result
        assert "P2" in result


class TestExtractCitations:
    def test_basic(self):
        assert _extract_citations("See [S1] and [S2].") == ["S1", "S2"]

    def test_deduplication(self):
        assert _extract_citations("[S1] foo [S1] bar [S2]") == ["S1", "S2"]

    def test_no_citations(self):
        assert _extract_citations("No citations here.") == []

    def test_preserves_appearance_order(self):
        assert _extract_citations("[S3] then [S1]") == ["S3", "S1"]

    def test_multidigit_ids(self):
        assert _extract_citations("[S10] and [S20]") == ["S10", "S20"]

    def test_mixed_non_citation_brackets(self):
        assert _extract_citations("[foo] [S1] [bar]") == ["S1"]

    def test_empty_string(self):
        assert _extract_citations("") == []


class TestStripMarkdownFences:
    def test_strips_backtick_fences(self):
        content = "```markdown\n## Outline\n```"
        assert "```" not in _strip_markdown_fences(content)

    def test_passes_through_plain_content(self):
        content = "## Abstract\n- Point 1"
        assert _strip_markdown_fences(content) == content

    def test_strips_json_fences(self):
        content = "```json\n{}\n```"
        assert "```" not in _strip_markdown_fences(content)


class TestGenerateOutline:
    def test_returns_string(self):
        llm = _make_llm("## Abstract\n- Point 1\n- Point 2")
        result = generate_outline("Q", _make_gap(), {}, llm)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_strips_markdown_fences(self):
        llm = _make_llm("```markdown\n## Abstract\n- Point\n```")
        result = generate_outline("Q", _make_gap(), {}, llm)
        assert "```" not in result

    def test_calls_llm_exactly_once(self):
        llm = _make_llm("Outline content")
        generate_outline("Q", _make_gap(), {}, llm)
        llm.invoke.assert_called_once()

    def test_prompt_includes_gap_title(self):
        llm = _make_llm("outline")
        gap = _make_gap()
        generate_outline("Research Q", gap, {}, llm)
        prompt = llm.invoke.call_args[0][0]
        assert gap["gap_title"] in prompt

    def test_prompt_includes_research_question(self):
        llm = _make_llm("outline")
        generate_outline("My Research Question", _make_gap(), {}, llm)
        prompt = llm.invoke.call_args[0][0]
        assert "My Research Question" in prompt

    def test_handles_list_content_response(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=[{"text": "## Abstract\n- Point"}])
        result = generate_outline("Q", _make_gap(), {}, llm)
        assert "Abstract" in result


class TestGenerateSection:
    def test_returns_draftsection_structure(self):
        llm = _make_llm("Section content with [S1].")
        section = generate_section("abstract", "Q", _make_gap(), {}, "outline", [], llm)
        assert "section_name" in section
        assert "content" in section
        assert "citations" in section

    def test_section_name_matches_input(self):
        llm = _make_llm("Content.")
        section = generate_section("introduction", "Q", _make_gap(), {}, "outline", [], llm)
        assert section["section_name"] == "introduction"

    def test_extracts_inline_citations(self):
        llm = _make_llm("See [S1] and [S2].")
        # FIX-4: source_map must contain S1 and S2 or they are dropped as hallucinations.
        sm = {"S1": _make_paper("P1"), "S2": _make_paper("P2")}
        section = generate_section("introduction", "Q", _make_gap(), sm, "outline", [], llm)
        assert section["citations"] == ["S1", "S2"]

    def test_no_citations_returns_empty_list(self):
        llm = _make_llm("Content without any citations.")
        section = generate_section("abstract", "Q", _make_gap(), {}, "outline", [], llm)
        assert section["citations"] == []

    def test_all_section_names_accepted(self):
        gap = _make_gap()
        for sec in SECTIONS:
            llm = _make_llm("Content.")
            s = generate_section(sec, "Q", gap, {}, "outline", [], llm)
            assert s["section_name"] == sec

    def test_calls_llm_exactly_once(self):
        llm = _make_llm("Content.")
        generate_section("abstract", "Q", _make_gap(), {}, "outline", [], llm)
        llm.invoke.assert_called_once()

    def test_prompt_includes_section_display_name(self):
        llm = _make_llm("Content.")
        generate_section("related_work", "Q", _make_gap(), {}, "outline", [], llm)
        prompt = llm.invoke.call_args[0][0]
        assert "Related Work" in prompt

    def test_prompt_includes_outline(self):
        llm = _make_llm("Content.")
        generate_section("abstract", "Q", _make_gap(), {}, "My Outline Text", [], llm)
        prompt = llm.invoke.call_args[0][0]
        assert "My Outline Text" in prompt

    def test_prompt_includes_source_map(self):
        llm = _make_llm("Content.")
        sm = {"S1": _make_paper("Graph Paper")}
        generate_section("abstract", "Q", _make_gap(), sm, "outline", [], llm)
        prompt = llm.invoke.call_args[0][0]
        assert "[S1]" in prompt
        assert "Graph Paper" in prompt

    def test_strips_markdown_fences_from_content(self):
        llm = _make_llm("```\nSection text.\n```")
        section = generate_section("conclusion", "Q", _make_gap(), {}, "outline", [], llm)
        assert "```" not in section["content"]
