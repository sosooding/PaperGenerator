"""
Gap analysis module for Phase 3.

Sends retrieved paper abstracts to Gemini and identifies research gaps
(open conjectures, unexplored graph families, missing proofs, complexity gaps).
"""

import json
import logging
from typing import List

from src.utils.config import Config
from src.utils.llm import get_llm
from src.utils.state import Paper, ResearchGap

logger = logging.getLogger(__name__)


def _evidence_id(paper: Paper) -> str:
    """Return DOI if present, otherwise fall back to the paper title."""
    doi = paper.get("doi", "")
    return doi if doi else paper.get("title", "unknown")


def _build_abstract_corpus(papers: List[Paper]) -> str:
    """Format papers into a numbered digest for the Gemini prompt."""
    lines = ["=== PAPER CORPUS ==="]
    for idx, paper in enumerate(papers, start=1):
        doi_str = paper.get("doi") or "no DOI"
        year_str = str(paper.get("year") or "unknown")
        authors_str = ", ".join(paper.get("authors") or []) or "unknown"
        lines.append(
            f"\n[{idx}] Title: {paper.get('title', 'Untitled')} | "
            f"Year: {year_str} | Authors: {authors_str} | DOI: {doi_str}"
        )
        lines.append(f"Abstract: {(paper.get('abstract') or '').strip()}")
    lines.append("\n=== END CORPUS ===")
    return "\n".join(lines)


def _strip_markdown_fences(content: str) -> str:
    """Remove ```json or ``` fences from LLM output (mirrors planner_node logic)."""
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return content


def _clamp_novelty(score) -> float:
    """Clamp novelty score to [0.0, 1.0]; return 0.0 for non-numeric values."""
    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.0


def _parse_gaps(raw_json: str) -> List[ResearchGap]:
    """Parse and validate a JSON array of gap objects into List[ResearchGap]."""
    parsed = json.loads(raw_json)  # raises json.JSONDecodeError on bad JSON
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    gaps: List[ResearchGap] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        evidence = [
            str(e) for e in item.get("supporting_evidence", []) if e
        ]
        gap: ResearchGap = {
            "gap_title": str(item.get("gap_title", "")),
            "description": str(item.get("description", "")),
            "supporting_evidence": evidence,
            "novelty_score": _clamp_novelty(item.get("novelty_score", 0.0)),
        }
        gaps.append(gap)
    return gaps


def find_research_gaps(papers: List[Paper], research_question: str) -> List[ResearchGap]:
    """
    Identify research gaps by sending all paper abstracts to Gemini.

    Returns an empty list when no papers are provided. Raises on LLM or
    JSON parse failure so the caller (gap_finder_node) can handle the error.
    """
    if not papers:
        logger.warning("find_research_gaps called with no papers; returning []")
        return []

    corpus = _build_abstract_corpus(papers)
    max_gaps = Config.MAX_GAPS

    # Build evidence-id instructions so Gemini knows what to put in supporting_evidence
    evidence_note = (
        "For each gap, list the evidence IDs of papers that support it. "
        "Use the paper's DOI when available (shown as 'DOI: <value>' in the corpus). "
        "When no DOI is available (shown as 'DOI: no DOI'), use the paper's title exactly "
        "as shown in the corpus. Do not leave supporting_evidence empty."
    )

    prompt = (
        "You are a graph theory research expert. Analyse the paper abstracts below "
        "and identify significant research gaps relative to the research question.\n\n"
        f"{corpus}\n\n"
        "Identify 3 to " + str(max_gaps) + " distinct research gaps. "
        "Consider the following gap categories:\n"
        "1. Open conjectures: well-known unsolved problems approached but not resolved by these papers\n"
        "2. Unexplored graph families: results proven for specific graph classes "
        "(planar, bipartite, chordal, etc.) but not extended to others\n"
        "3. Missing proofs: results stated without proof, or proofs relying on unverified lemmas\n"
        "4. Complexity gaps: problems whose computational complexity is unknown "
        "or whose bounds are not tight\n"
        "5. Algorithmic gaps: theoretical results with no known efficient algorithm\n\n"
        + evidence_note + "\n\n"
        "Return ONLY a valid JSON array — no explanation, no markdown fences. "
        "Each element must have exactly these fields:\n"
        '  "gap_title": string (concise name, max 80 chars)\n'
        '  "description": string (2-4 sentences explaining the gap and why it matters)\n'
        '  "supporting_evidence": array of DOI strings or paper titles from the corpus\n'
        '  "novelty_score": float 0.0-1.0 '
        "(1.0 = completely unexplored, 0.0 = extensively studied)\n\n"
        f"Research question: {research_question}"
    )

    llm = get_llm(temperature=0.3)

    logger.info("Calling LLM (%s/%s) for gap analysis on %d papers",
                Config.LLM_PROVIDER, Config.LLM_MODEL, len(papers))
    response = llm.invoke(prompt)

    # Normalise content — langchain-google-genai may return list-of-parts
    raw = response.content
    if isinstance(raw, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    else:
        content = str(raw)

    content = _strip_markdown_fences(content.strip())
    gaps = _parse_gaps(content)
    logger.info("Identified %d research gaps", len(gaps))
    return gaps
