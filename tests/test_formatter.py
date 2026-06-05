"""
Tests for Phase 5: formatter node (src/formatting/formatter.py).

All tests are hermetic — no filesystem side-effects escape each test,
no real pdflatex/bibtex processes are spawned.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.formatting.formatter import (
    apa_intext,
    bibtex_entry,
    build_bib,
    build_tex,
    collect_cited_papers,
    compile_pdf,
    format_paper,
    latex_escape,
    replace_citations,
    write_output_files,
)
from tests.conftest import make_paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_a():
    return make_paper(
        title="Graph Theory Advances",
        doi="10.1/graph",
        year=2022,
        abstract="An abstract about graphs.",
    )


@pytest.fixture
def paper_b():
    return make_paper(
        title="Spectral Methods",
        doi="10.1/spectral",
        year=2021,
        abstract="An abstract about spectral methods.",
    )


@pytest.fixture
def paper_no_author():
    p = make_paper(title="Orphan Paper", doi="10.1/orphan", year=2020)
    p["authors"] = []
    return p


@pytest.fixture
def paper_no_year():
    p = make_paper(title="Undated Paper", doi="10.1/undated")
    p["year"] = None
    return p


@pytest.fixture
def paper_no_doi():
    p = make_paper(title="No DOI Paper", year=2019)
    p["doi"] = None
    return p


@pytest.fixture
def citation_report(paper_a, paper_b):
    return {"S1": paper_a, "S2": paper_b}


@pytest.fixture
def draft_sections():
    return [
        {
            "section_name": "abstract",
            "content": "This paper studies graphs [S2].",
            "citations": ["S2"],
        },
        {
            "section_name": "introduction",
            "content": "We build on prior work [S1] and [S2].",
            "citations": ["S1", "S2"],
        },
        {
            "section_name": "related_work",
            "content": "Many methods exist [S1].",
            "citations": ["S1"],
        },
        {
            "section_name": "gap_analysis",
            "content": "The gap is clear.",
            "citations": [],
        },
        {
            "section_name": "proposed_approach",
            "content": "Our method [S1] is novel.",
            "citations": ["S1"],
        },
        {
            "section_name": "conclusion",
            "content": "We conclude [S2].",
            "citations": ["S2"],
        },
    ]


# ---------------------------------------------------------------------------
# collect_cited_papers
# ---------------------------------------------------------------------------

class TestCollectCitedPapers:
    def test_first_appearance_order(self, draft_sections, citation_report, paper_a, paper_b):
        result = collect_cited_papers(draft_sections, citation_report)
        sids = [sid for sid, _ in result]
        # S2 appears first (in abstract), S1 second (in introduction)
        assert sids == ["S2", "S1"]

    def test_deduplication(self, draft_sections, citation_report):
        result = collect_cited_papers(draft_sections, citation_report)
        sids = [sid for sid, _ in result]
        assert len(sids) == len(set(sids))

    def test_unknown_sid_excluded(self, citation_report):
        sections = [
            {"section_name": "introduction", "content": "[S99]", "citations": ["S99"]}
        ]
        result = collect_cited_papers(sections, citation_report)
        assert result == []

    def test_empty_sections(self, citation_report):
        assert collect_cited_papers([], citation_report) == []

    def test_section_with_no_citations(self, citation_report):
        sections = [{"section_name": "conclusion", "content": "Done.", "citations": []}]
        assert collect_cited_papers(sections, citation_report) == []


# ---------------------------------------------------------------------------
# apa_intext
# ---------------------------------------------------------------------------

class TestApaIntext:
    def test_multiple_authors(self, paper_a):
        paper_a["authors"] = ["Alice Smith", "Bob Jones", "Carol Brown"]
        result = apa_intext(paper_a)
        assert result == "(Smith et al., 2022)"

    def test_single_author(self, paper_a):
        paper_a["authors"] = ["Jane Smith"]
        assert apa_intext(paper_a) == "(Smith, 2022)"

    def test_two_authors(self, paper_a):
        paper_a["authors"] = ["Alice Brown", "Bob Green"]
        assert apa_intext(paper_a) == "(Brown & Green, 2022)"

    def test_no_authors(self, paper_no_author):
        result = apa_intext(paper_no_author)
        assert result == "(Unknown, 2020)"

    def test_no_year(self, paper_no_year):
        result = apa_intext(paper_no_year)
        assert "n.d." in result

    def test_no_authors_no_year(self, paper_no_author):
        paper_no_author["year"] = None
        result = apa_intext(paper_no_author)
        assert result == "(Unknown, n.d.)"


# ---------------------------------------------------------------------------
# latex_escape
# ---------------------------------------------------------------------------

class TestLatexEscape:
    @pytest.mark.parametrize("char,expected", [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("\\", r"\textbackslash{}"),
    ])
    def test_individual_chars(self, char, expected):
        assert latex_escape(char) == expected

    def test_plain_text_unchanged(self):
        assert latex_escape("hello world") == "hello world"

    def test_mixed(self):
        result = latex_escape("50% of $x_1$")
        assert r"\%" in result
        assert r"\$" in result
        assert r"\_" in result


# ---------------------------------------------------------------------------
# replace_citations
# ---------------------------------------------------------------------------

class TestReplaceCitations:
    def test_known_tag_replaced(self, citation_report):
        result = replace_citations("See [S1] for details.", citation_report)
        assert r"\citep{S1}" in result
        assert "[S1]" not in result

    def test_unknown_tag_preserved(self, citation_report):
        result = replace_citations("Unknown [S99].", citation_report)
        assert "[S99]" in result
        assert r"\citep{S99}" not in result

    def test_multiple_tags(self, citation_report):
        result = replace_citations("[S1] and [S2].", citation_report)
        assert r"\citep{S1}" in result
        assert r"\citep{S2}" in result

    def test_no_tags(self, citation_report):
        text = "Plain text only."
        assert replace_citations(text, citation_report) == text


# ---------------------------------------------------------------------------
# bibtex_entry
# ---------------------------------------------------------------------------

class TestBibtexEntry:
    def test_contains_required_fields(self, paper_a):
        entry = bibtex_entry("S1", paper_a)
        assert "@article{S1," in entry
        assert "author" in entry
        assert "title" in entry
        assert "year" in entry

    def test_doi_included_when_present(self, paper_a):
        entry = bibtex_entry("S1", paper_a)
        assert "doi" in entry
        assert "10.1/graph" in entry

    def test_doi_omitted_when_none(self, paper_no_doi):
        entry = bibtex_entry("S1", paper_no_doi)
        assert "doi" not in entry

    def test_url_included_when_present(self, paper_a):
        paper_a["pdf_url"] = "https://arxiv.org/pdf/1234"
        entry = bibtex_entry("S1", paper_a)
        assert "url" in entry

    def test_no_authors(self, paper_no_author):
        entry = bibtex_entry("S1", paper_no_author)
        assert "Unknown" in entry


# ---------------------------------------------------------------------------
# build_bib
# ---------------------------------------------------------------------------

class TestBuildBib:
    def test_contains_all_entries(self, citation_report, paper_a, paper_b):
        cited = [("S1", paper_a), ("S2", paper_b)]
        bib = build_bib(cited)
        assert "@article{S1," in bib
        assert "@article{S2," in bib

    def test_empty_cited_papers(self):
        assert build_bib([]) == ""


# ---------------------------------------------------------------------------
# build_tex
# ---------------------------------------------------------------------------

class TestBuildTex:
    def test_documentclass_present(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\documentclass{article}" in tex

    def test_natbib_and_apalike(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\usepackage{natbib}" in tex
        assert r"\bibliographystyle{apalike}" in tex

    def test_abstract_environment(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\begin{abstract}" in tex
        assert r"\end{abstract}" in tex

    def test_sections_present(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\section{Introduction}" in tex
        assert r"\section{Related Work}" in tex
        assert r"\section{Gap Analysis}" in tex
        assert r"\section{Proposed Approach}" in tex
        assert r"\section{Conclusion}" in tex

    def test_bibliography_command(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\bibliography{references}" in tex

    def test_citations_replaced(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\citep{S1}" in tex or r"\citep{S2}" in tex
        # Raw [Sx] tags should not appear for known citations
        assert "[S1]" not in tex
        assert "[S2]" not in tex

    def test_no_abstract_section_heading(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Test question", "Test Gap")
        assert r"\section{Abstract}" not in tex

    def test_title_uses_gap(self, draft_sections, citation_report):
        tex = build_tex(draft_sections, citation_report, "Research Q", "My Gap Title")
        assert "My Gap Title" in tex

    def test_special_chars_escaped_in_content(self, citation_report):
        sections = [
            {"section_name": "introduction", "content": "100% certain & sure.", "citations": []},
        ]
        tex = build_tex(sections, citation_report, "Q", "G")
        assert r"100\% certain \& sure." in tex


# ---------------------------------------------------------------------------
# write_output_files
# ---------------------------------------------------------------------------

class TestWriteOutputFiles:
    def test_files_created(self, tmp_path):
        tex_path, bib_path = write_output_files("tex content", "bib content", tmp_path)
        assert tex_path.exists()
        assert bib_path.exists()

    def test_file_contents(self, tmp_path):
        write_output_files("my tex", "my bib", tmp_path)
        assert (tmp_path / "paper.tex").read_text(encoding="utf-8") == "my tex"
        assert (tmp_path / "references.bib").read_text(encoding="utf-8") == "my bib"


# ---------------------------------------------------------------------------
# compile_pdf
# ---------------------------------------------------------------------------

class TestCompilePdf:
    def test_returns_none_when_pdflatex_missing(self, tmp_path):
        (tmp_path / "paper.tex").write_text("", encoding="utf-8")
        with patch("src.formatting.formatter.shutil.which", return_value=None):
            result = compile_pdf(tmp_path / "paper.tex", tmp_path)
        assert result is None

    def test_returns_pdf_path_on_success(self, tmp_path):
        tex_path = tmp_path / "paper.tex"
        tex_path.write_text("", encoding="utf-8")
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_text("fake pdf", encoding="utf-8")

        with patch("src.formatting.formatter.shutil.which", return_value="/usr/bin/pdflatex"):
            with patch("src.formatting.formatter.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                result = compile_pdf(tex_path, tmp_path)

        assert result == pdf_path

    def test_returns_none_when_pdf_not_produced(self, tmp_path):
        tex_path = tmp_path / "paper.tex"
        tex_path.write_text("", encoding="utf-8")
        # No pdf_path created — simulate compilation failure

        with patch("src.formatting.formatter.shutil.which", return_value="/usr/bin/pdflatex"):
            with patch("src.formatting.formatter.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="error", stderr="")
                result = compile_pdf(tex_path, tmp_path)

        assert result is None

    def test_runs_four_steps(self, tmp_path):
        tex_path = tmp_path / "paper.tex"
        tex_path.write_text("", encoding="utf-8")
        (tmp_path / "paper.pdf").write_text("pdf", encoding="utf-8")

        with patch("src.formatting.formatter.shutil.which", return_value="/usr/bin/pdflatex"):
            with patch("src.formatting.formatter.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                compile_pdf(tex_path, tmp_path)

        assert mock_run.call_count == 4  # pdflatex, bibtex, pdflatex, pdflatex


# ---------------------------------------------------------------------------
# format_paper (integration)
# ---------------------------------------------------------------------------

class TestFormatPaper:
    def test_returns_expected_keys(self, draft_sections, citation_report, tmp_path):
        with patch("src.formatting.formatter._make_output_dir", return_value=tmp_path):
            with patch("src.formatting.formatter.compile_pdf", return_value=None):
                result = format_paper(
                    draft_sections=draft_sections,
                    citation_report=citation_report,
                    research_question="Graph partitioning",
                    gap_title="Spectral gap",
                )

        for key in ("tex_path", "bib_path", "pdf_path", "reference_order", "sid_to_apa"):
            assert key in result

    def test_reference_order_correct(self, draft_sections, citation_report, tmp_path):
        with patch("src.formatting.formatter._make_output_dir", return_value=tmp_path):
            with patch("src.formatting.formatter.compile_pdf", return_value=None):
                result = format_paper(
                    draft_sections=draft_sections,
                    citation_report=citation_report,
                    research_question="Q",
                    gap_title="G",
                )
        # S2 appears first in abstract, then S1 in introduction
        assert result["reference_order"] == ["S2", "S1"]

    def test_sid_to_apa_populated(self, draft_sections, citation_report, tmp_path):
        with patch("src.formatting.formatter._make_output_dir", return_value=tmp_path):
            with patch("src.formatting.formatter.compile_pdf", return_value=None):
                result = format_paper(
                    draft_sections=draft_sections,
                    citation_report=citation_report,
                    research_question="Q",
                    gap_title="G",
                )
        assert "S1" in result["sid_to_apa"]
        assert "S2" in result["sid_to_apa"]
        assert result["sid_to_apa"]["S1"].startswith("(")

    def test_tex_file_written(self, draft_sections, citation_report, tmp_path):
        with patch("src.formatting.formatter._make_output_dir", return_value=tmp_path):
            with patch("src.formatting.formatter.compile_pdf", return_value=None):
                result = format_paper(
                    draft_sections=draft_sections,
                    citation_report=citation_report,
                    research_question="Q",
                    gap_title="G",
                )
        assert Path(result["tex_path"]).exists()
        assert Path(result["bib_path"]).exists()

    def test_original_report_entries_preserved(self, draft_sections, citation_report, tmp_path):
        with patch("src.formatting.formatter._make_output_dir", return_value=tmp_path):
            with patch("src.formatting.formatter.compile_pdf", return_value=None):
                result = format_paper(
                    draft_sections=draft_sections,
                    citation_report=citation_report,
                    research_question="Q",
                    gap_title="G",
                )
        # Original paper entries should still be accessible
        assert "S1" in result
        assert "S2" in result


# ---------------------------------------------------------------------------
# formatter_node (graph wiring)
# ---------------------------------------------------------------------------

class TestFormatterNode:
    def _make_state(self, draft_sections, citation_report):
        return {
            "research_question": "Graph partitioning algorithms",
            "sub_queries": [],
            "retrieved_papers": [],
            "embeddings_ready": True,
            "gaps": [],
            "selected_gap": {"gap_title": "Spectral gap", "description": "desc", "supporting_evidence": [], "novelty_score": 0.9},
            "outline": "## Abstract\n- bullet",
            "draft_sections": draft_sections,
            "critic_feedback": [],
            "revision_count": 1,
            "eval_scores": None,
            "citation_report": citation_report,
            "human_decisions": [],
            "current_phase": "critiquing",
            "errors": [],
        }

    def test_updates_citation_report(self, draft_sections, citation_report, tmp_path):
        from src.agents.graph import formatter_node

        state = self._make_state(draft_sections, citation_report)
        with patch("src.formatting.formatter._make_output_dir", return_value=tmp_path):
            with patch("src.formatting.formatter.compile_pdf", return_value=None):
                result = formatter_node(state)

        assert "tex_path" in result["citation_report"]
        assert result["current_phase"] == "formatting"

    def test_errors_on_exception(self, draft_sections, citation_report):
        from src.agents.graph import formatter_node

        state = self._make_state(draft_sections, citation_report)
        with patch("src.agents.graph.format_paper", side_effect=RuntimeError("boom")):
            result = formatter_node(state)

        assert result["current_phase"] == "formatting"
        assert any("boom" in e for e in result["errors"])
