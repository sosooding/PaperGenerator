"""
Phase 4b: outline generation and section-by-section writing with grounded [Sx] citations.
"""

import re
import logging
from typing import Dict, List

from src.utils.config import Config
from src.utils.llm import get_llm
from src.utils.state import DraftSection, Paper, ResearchGap

logger = logging.getLogger(__name__)

SECTIONS: List[str] = [
    "abstract",
    "introduction",
    "related_work",
    "gap_analysis",
    "proposed_approach",
    "conclusion",
]

SECTION_RAG_SUFFIXES: Dict[str, str] = {
    "abstract": "overview motivation contributions results summary",
    "introduction": "problem motivation background context contributions",
    "related_work": "existing approaches methods techniques limitations comparison",
    "gap_analysis": "research gaps open problems limitations unexplored",
    "proposed_approach": "novel methodology algorithm framework solution design",
    "conclusion": "findings contributions limitations future work",
}

_SECTION_INSTRUCTIONS: Dict[str, str] = {
    "abstract": (
        "Write a single paragraph summarising the paper: motivation, selected research gap, "
        "proposed approach, and main contributions. Do NOT include inline citations."
    ),
    "introduction": (
        "Motivate the problem with theoretical or practical context. State the research gap "
        "clearly and summarise the paper's contributions. End with a brief roadmap of the "
        "remaining sections."
    ),
    "related_work": (
        "Survey existing work grouped by theme or approach. For each group, describe "
        "strengths and limitations. Show explicitly why existing work leaves the selected "
        "gap open."
    ),
    "gap_analysis": (
        "Articulate the selected research gap precisely: what is unknown, why it matters, "
        "and what challenges prevent straightforward solutions. Ground every claim with a "
        "citation."
    ),
    "proposed_approach": (
        "Describe the novel methodology or framework in detail. Include key definitions, "
        "algorithms, or theoretical constructs. Justify all design choices with citations "
        "or reasoning."
    ),
    "conclusion": (
        "Summarise the paper's contributions, discuss limitations, and propose concrete "
        "directions for future work."
    ),
}


def build_source_map(papers: List[Paper]) -> Dict[str, Paper]:
    """Assign stable S1, S2, ... IDs to papers in retrieval order."""
    return {f"S{i + 1}": paper for i, paper in enumerate(papers)}


def _format_source_map_prompt(source_map: Dict[str, Paper]) -> str:
    lines = []
    for sid, paper in source_map.items():
        authors = paper.get("authors") or []
        last_name = authors[0].split()[-1] if authors else "Unknown"
        year = paper.get("year") or "n.d."
        title = paper.get("title", "Untitled")
        doi = paper.get("doi") or ""
        doi_str = f" DOI: {doi}" if doi else ""
        lines.append(f"[{sid}] {last_name} et al. ({year}). {title}.{doi_str}")
    return "\n".join(lines)


def _format_rag_context(papers: List[Paper]) -> str:
    if not papers:
        return "(No additional context retrieved.)"
    lines = []
    for paper in papers:
        title = paper.get("title", "Untitled")
        abstract = (paper.get("abstract") or "").strip()[:500]
        lines.append(f"Title: {title}\nAbstract excerpt: {abstract}\n")
    return "\n".join(lines)


def _extract_citations(content: str) -> List[str]:
    """Return deduplicated [Sx] IDs in order of first appearance."""
    seen: set = set()
    result: List[str] = []
    for m in re.finditer(r"\[S(\d+)\]", content):
        sid = f"S{m.group(1)}"
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def _normalise_llm_content(raw) -> str:
    if isinstance(raw, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        ).strip()
    return str(raw).strip()


def _strip_markdown_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return content


def generate_outline(
    research_question: str,
    gap: ResearchGap,
    source_map: Dict[str, Paper],
    llm,
) -> str:
    """1 LLM call: section headers with 2-3 bullets each."""
    source_list = _format_source_map_prompt(source_map)
    prompt = (
        "You are a research paper writing assistant specialising in graph theory.\n\n"
        f"Research question: {research_question}\n"
        f"Selected research gap: {gap['gap_title']}\n"
        f"Gap description: {gap['description']}\n\n"
        "Available sources:\n"
        f"{source_list}\n\n"
        "Generate an outline for a 6-section academic paper with these sections:\n"
        "1. Abstract  2. Introduction  3. Related Work  "
        "4. Gap Analysis  5. Proposed Approach  6. Conclusion\n\n"
        "For each section provide 2-3 bullet points (one sentence each). Format exactly as:\n\n"
        "## Abstract\n- ...\n- ...\n\n## Introduction\n- ...\n\n"
        "Return only the outline — no preamble or explanation."
    )
    logger.info("Generating outline for gap %r", gap["gap_title"])
    response = llm.invoke(prompt)
    outline = _strip_markdown_fences(_normalise_llm_content(response.content))
    logger.info("Outline generated (%d chars)", len(outline))
    return outline


def generate_section(
    section_name: str,
    research_question: str,
    gap: ResearchGap,
    source_map: Dict[str, Paper],
    outline: str,
    rag_papers: List[Paper],
    llm,
) -> DraftSection:
    """1 LLM call: generate one section with inline [Sx] citations."""
    source_list = _format_source_map_prompt(source_map)
    rag_context = _format_rag_context(rag_papers)
    min_words = Config.MIN_WORD_COUNTS.get(section_name, 200)
    instruction = _SECTION_INSTRUCTIONS.get(section_name, "Write this section.")
    display_name = section_name.replace("_", " ").title()

    prompt = (
        f"You are writing the **{display_name}** section of a research paper.\n\n"
        f"Research question: {research_question}\n"
        f"Research gap: {gap['gap_title']} — {gap['description']}\n\n"
        "Paper outline (for coherence):\n"
        f"{outline}\n\n"
        "Available sources — cite inline as [S1], [S2], etc. "
        "Only cite sources listed here:\n"
        f"{source_list}\n\n"
        "Relevant literature context:\n"
        f"{rag_context}\n\n"
        f"Writing instruction: {instruction}\n"
        f"Minimum length: {min_words} words.\n\n"
        f"Write the {display_name} section now. "
        "Output only the section content — no section heading, no preamble."
    )
    logger.info("Generating section: %s", section_name)
    response = llm.invoke(prompt)
    content = _strip_markdown_fences(_normalise_llm_content(response.content))
    citations = _extract_citations(content)
    return {
        "section_name": section_name,
        "content": content,
        "citations": citations,
    }
