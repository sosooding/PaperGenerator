"""
Phase 5: LaTeX/PDF formatter.

Pipeline:
  collect_cited_papers → build_tex + build_bib → write_output_files → compile_pdf

Entry point for the graph node: format_paper()
"""

import logging
import re
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.state import DraftSection, Paper
from src.utils.config import Config

logger = logging.getLogger(__name__)

# Section names → display titles for \section{} headings
_SECTION_TITLES: Dict[str, str] = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related_work": "Related Work",
    "gap_analysis": "Gap Analysis",
    "proposed_approach": "Proposed Approach",
    "conclusion": "Conclusion",
}

_LATEX_SPECIAL = str.maketrans(
    {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
)


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def collect_cited_papers(
    draft_sections: List[DraftSection],
    citation_report: Dict[str, Paper],
) -> List[Tuple[str, Paper]]:
    """
    Return (sid, paper) pairs in first-appearance order across all sections.
    Sections are processed in the order they appear in draft_sections.
    """
    seen: set = set()
    ordered: List[Tuple[str, Paper]] = []
    for section in draft_sections:
        for sid in section.get("citations", []):
            if sid not in seen and sid in citation_report:
                seen.add(sid)
                ordered.append((sid, citation_report[sid]))
    return ordered


def apa_intext(paper: Paper) -> str:
    """Format an APA-style in-text citation string: '(Last et al., Year)'."""
    authors = paper.get("authors") or []
    year = paper.get("year") or "n.d."

    if not authors:
        author_str = "Unknown"
    else:
        last_names = [a.split()[-1] for a in authors if a.strip()]
        if not last_names:
            author_str = "Unknown"
        elif len(last_names) == 1:
            author_str = last_names[0]
        elif len(last_names) == 2:
            author_str = f"{last_names[0]} & {last_names[1]}"
        else:
            author_str = f"{last_names[0]} et al."

    return f"({author_str}, {year})"


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return text.translate(_LATEX_SPECIAL)


def _bibtex_authors(authors: List[str]) -> str:
    """Format author list for BibTeX: 'Last, First and Last2, First2'."""
    if not authors:
        return "Unknown"
    parts = []
    for author in authors:
        tokens = author.strip().split()
        if len(tokens) >= 2:
            # Assume "First [Middle] Last" — put last name first
            parts.append(f"{tokens[-1]}, {' '.join(tokens[:-1])}")
        else:
            parts.append(author)
    return " and ".join(parts)


def bibtex_entry(sid: str, paper: Paper) -> str:
    """Generate a single BibTeX @article entry for a paper."""
    title = paper.get("title", "Untitled").replace("{", "").replace("}", "")
    authors = paper.get("authors") or []
    year = paper.get("year") or "n.d."
    doi = paper.get("doi") or ""
    url = paper.get("pdf_url") or ""

    lines = [
        f"@article{{{sid},",
        f"  author = {{{_bibtex_authors(authors)}}},",
        f"  title  = {{{title}}},",
        f"  year   = {{{year}}},",
    ]
    if doi:
        lines.append(f"  doi    = {{{doi}}},")
    if url:
        lines.append(f"  url    = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def replace_citations(content: str, citation_report: Dict[str, Paper]) -> str:
    r"""
    Replace [Sx] tags with \citep{Sx}.
    Tags not present in citation_report are left unchanged (defensive).
    The surrounding prose is NOT latex-escaped here — call latex_escape separately
    before this function so that \citep{} markers are not double-escaped.
    """
    def _replace(match: re.Match) -> str:
        sid = f"S{match.group(1)}"
        if sid in citation_report:
            return rf"\citep{{{sid}}}"
        return match.group(0)  # unknown tag — leave as-is

    return re.sub(r"\[S(\d+)\]", _replace, content)


def _process_section_content(content: str, citation_report: Dict[str, Paper]) -> str:
    """Escape LaTeX specials then swap [Sx] tags for \\citep{}."""
    # Split on [Sx] tags so we only escape the prose, not the tags themselves.
    parts = re.split(r"(\[S\d+\])", content)
    processed = []
    for part in parts:
        if re.fullmatch(r"\[S\d+\]", part):
            # Citation tag — convert, don't escape
            sid = part[1:-1]  # strip brackets
            if sid in citation_report:
                processed.append(rf"\citep{{{sid}}}")
            else:
                processed.append(latex_escape(part))
        else:
            processed.append(latex_escape(part))
    return "".join(processed)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def build_tex(
    draft_sections: List[DraftSection],
    citation_report: Dict[str, Paper],
    research_question: str,
    gap_title: str,
) -> str:
    """Assemble the complete LaTeX document string."""
    title = latex_escape(gap_title) if gap_title else latex_escape(research_question)

    preamble = rf"""\documentclass{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{natbib}}
\usepackage{{hyperref}}
\usepackage{{amsmath,amssymb}}
\usepackage{{microtype}}

\bibliographystyle{{apalike}}

\title{{{title}}}
\author{{AI-Generated Research Paper}}
\date{{\today}}

\begin{{document}}
\maketitle
"""

    body_parts = [preamble]

    section_map = {s["section_name"]: s for s in draft_sections}

    # Abstract gets a special environment, not \section
    abstract_section = section_map.get("abstract")
    if abstract_section:
        abstract_content = _process_section_content(
            abstract_section["content"], citation_report
        )
        body_parts.append(r"\begin{abstract}")
        body_parts.append(abstract_content.strip())
        body_parts.append(r"\end{abstract}")
        body_parts.append("")

    # Remaining sections as \section{}
    for section_name in ["introduction", "related_work", "gap_analysis", "proposed_approach", "conclusion"]:
        section = section_map.get(section_name)
        if not section:
            continue
        display = _SECTION_TITLES.get(section_name, section_name.replace("_", " ").title())
        content = _process_section_content(section["content"], citation_report)
        body_parts.append(rf"\section{{{display}}}")
        body_parts.append(content.strip())
        body_parts.append("")

    body_parts.append(r"\bibliography{references}")
    body_parts.append(r"\end{document}")

    return "\n".join(body_parts)


def build_bib(cited_papers: List[Tuple[str, Paper]]) -> str:
    """Assemble the complete BibTeX file string."""
    entries = [bibtex_entry(sid, paper) for sid, paper in cited_papers]
    return "\n\n".join(entries)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _make_output_dir(research_question: str) -> Path:
    """Create a timestamped subdirectory under OUTPUT_DIR for this run."""
    slug = re.sub(r"[^a-z0-9]+", "_", research_question.lower().strip())[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(Config.OUTPUT_DIR) / f"{slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_output_files(
    tex_str: str,
    bib_str: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write .tex and .bib files to output_dir; return their paths."""
    tex_path = output_dir / "paper.tex"
    bib_path = output_dir / "references.bib"
    tex_path.write_text(tex_str, encoding="utf-8")
    bib_path.write_text(bib_str, encoding="utf-8")
    logger.info("[FORMATTER] Wrote %s and %s", tex_path, bib_path)
    return tex_path, bib_path


# ---------------------------------------------------------------------------
# PDF compilation
# ---------------------------------------------------------------------------

def compile_pdf(tex_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Compile .tex → PDF using the standard 4-step pdflatex/bibtex sequence.
    Returns the PDF path on success, None if pdflatex is not on PATH or fails.
    """
    if not shutil.which("pdflatex"):
        logger.warning("[FORMATTER] pdflatex not found on PATH; skipping PDF compilation")
        return None
    if not shutil.which("bibtex"):
        logger.warning("[FORMATTER] bibtex not found on PATH; skipping PDF compilation")
        return None

    base_name = tex_path.stem  # "paper"
    out_dir = str(output_dir)

    def _run(cmd: List[str]) -> bool:
        try:
            result = subprocess.run(
                cmd,
                cwd=out_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning("[FORMATTER] %s exited %d:\n%s", cmd[0], result.returncode, result.stdout[-2000:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("[FORMATTER] %s timed out after 60s", cmd[0])
            return False
        except Exception as exc:
            logger.error("[FORMATTER] %s failed: %s", cmd[0], exc)
            return False

    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory", out_dir,
        str(tex_path),
    ]

    # Step 1: first pdflatex pass (generates .aux)
    _run(pdflatex_cmd)
    # Step 2: bibtex (processes .aux → .bbl)
    _run(["bibtex", base_name])
    # Steps 3 & 4: two more pdflatex passes to resolve references
    _run(pdflatex_cmd)
    _run(pdflatex_cmd)

    pdf_path = output_dir / f"{base_name}.pdf"
    if pdf_path.exists():
        logger.info("[FORMATTER] PDF written to %s", pdf_path)
        return pdf_path

    logger.error("[FORMATTER] PDF not produced after compilation; check %s", output_dir / f"{base_name}.log")
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def format_paper(
    draft_sections: List[DraftSection],
    citation_report: Dict[str, Paper],
    research_question: str,
    gap_title: str,
) -> Dict:
    """
    Orchestrate the full formatting pipeline.

    Returns an updated citation_report dict with keys:
        tex_path, bib_path, pdf_path, reference_order, sid_to_apa, papers
    """
    # 1. Determine which papers are actually cited (first-appearance order)
    cited_papers = collect_cited_papers(draft_sections, citation_report)
    reference_order = [sid for sid, _ in cited_papers]
    sid_to_apa = {sid: apa_intext(paper) for sid, paper in cited_papers}

    logger.info(
        "[FORMATTER] %d unique citations in reference order: %s",
        len(cited_papers),
        reference_order,
    )

    # 2. Build document strings
    tex_str = build_tex(draft_sections, citation_report, research_question, gap_title)
    bib_str = build_bib(cited_papers)

    # 3. Write files
    output_dir = _make_output_dir(research_question)
    tex_path, bib_path = write_output_files(tex_str, bib_str, output_dir)

    # 4. Compile PDF
    pdf_path = compile_pdf(tex_path, output_dir)

    return {
        **citation_report,
        "tex_path": str(tex_path),
        "bib_path": str(bib_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "reference_order": reference_order,
        "sid_to_apa": sid_to_apa,
    }
