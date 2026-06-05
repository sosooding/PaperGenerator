"""
Phase 4d: critic node helpers — grounding check and coherence check.

Exactly 2 LLM calls per critique cycle to stay within Gemini's RPM limit.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from src.utils.llm_parse import normalise_content, strip_fences
from src.utils.state import DraftSection

logger = logging.getLogger(__name__)

_MAX_CITED_SENTENCES = 50   # cap sent to LLM so prompt stays reasonable
_MAX_ABSTRACT_CHARS = 300   # abstract excerpt per source
_MAX_SECTION_CHARS = 800    # section excerpt per section for coherence check
_MAX_SENTENCE_CHARS = 300   # per-sentence cap in grounding prompt

_GROUNDING_PROMPT = """\
You are a critical reviewer of an academic research paper on graph theory.

Task: identify sentences that misrepresent or hallucinate claims not supported by \
their cited source.

SOURCES (title + abstract excerpt):
{source_excerpts}

CITED SENTENCES (format: index. [section] "sentence"):
{cited_sentences}

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "grounding_score": <float 0-10, where 10 = every citation is accurate>,
  "flagged": [
    {{"sentence": "<exact sentence>", "issue": "<one sentence explaining mismatch>", \
"source_id": "<Sx>"}},
    ...
  ]
}}
Cap flagged to at most 10 items. If all citations are accurate, return an empty flagged list.\
"""

_COHERENCE_PROMPT = """\
You are a critical reviewer of an academic research paper on graph theory.

Task: evaluate the logical flow, internal consistency, and completeness of the paper.

PAPER OUTLINE:
{outline}

PAPER SECTIONS (excerpts in order):
{sections_text}

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "coherence_score": <float 0-10, where 10 = excellent logical flow>,
  "coherence_issues": ["<issue 1>", ...],
  "missing_sections": ["<section name>", ...]
}}
Cap coherence_issues to at most 5 items. \
List a section in missing_sections only if it is truly absent or completely empty.\
"""


def _extract_cited_sentences(draft_sections: List[DraftSection]) -> List[Dict[str, str]]:
    """Pull sentences containing at least one [Sx] citation from all sections."""
    result: List[Dict[str, str]] = []
    for section in draft_sections:
        content = section.get("content", "")
        for sent in re.split(r"(?<=[.!?])\s+", content):
            if re.search(r"\[S\d+\]", sent):
                result.append({
                    "section": section["section_name"],
                    "sentence": sent.strip()[:_MAX_SENTENCE_CHARS],
                })
    return result


def _format_source_excerpts(citation_report: Dict[str, Any]) -> str:
    lines: List[str] = []
    for sid, paper in citation_report.items():
        if not isinstance(paper, dict):
            lines.append(f"[{sid}] {paper}")
            continue
        title = paper.get("title", "Untitled")
        abstract = (paper.get("abstract") or "")[:_MAX_ABSTRACT_CHARS]
        doi = paper.get("doi", "")
        doi_str = f"  DOI: {doi}" if doi else ""
        lines.append(f"[{sid}] {title}{doi_str}\nAbstract: {abstract}")
    return "\n\n".join(lines) if lines else "(No sources available)"


def _clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return 5.0


def run_grounding_check(
    draft_sections: List[DraftSection],
    citation_report: Dict[str, Any],
    llm,
) -> Tuple[List[Dict[str, str]], float]:
    """
    1 LLM call: verify cited sentences are supported by their sources.
    Returns (flagged_sentences, grounding_score).
    Fallback: ([], 5.0) on any error.
    """
    cited = _extract_cited_sentences(draft_sections)
    if not cited:
        logger.info("[CRITIC] No cited sentences found; grounding score defaults to 10.0")
        return [], 10.0

    source_excerpts = _format_source_excerpts(citation_report)
    numbered = "\n".join(
        f'{i + 1}. [{s["section"]}] "{s["sentence"]}"'
        for i, s in enumerate(cited[:_MAX_CITED_SENTENCES])
    )

    prompt = _GROUNDING_PROMPT.format(
        source_excerpts=source_excerpts,
        cited_sentences=numbered,
    )

    try:
        response = llm.invoke(prompt)
        data = json.loads(strip_fences(normalise_content(response.content)))
        flagged = [
            {
                "sentence": str(f.get("sentence", "")),
                "issue": str(f.get("issue", "")),
                "source_id": str(f.get("source_id", "")),
            }
            for f in data.get("flagged", [])
            if isinstance(f, dict)
        ]
        score = _clamp_score(data.get("grounding_score", 5.0))
        logger.info("[CRITIC] Grounding score=%.1f, flagged=%d", score, len(flagged))
        return flagged, score
    except Exception as exc:
        logger.error("[CRITIC] Grounding check failed (%s); defaulting to score=5.0", exc)
        return [], 5.0


def run_coherence_check(
    draft_sections: List[DraftSection],
    outline: str,
    llm,
) -> Tuple[List[str], List[str], float]:
    """
    1 LLM call: evaluate logical flow and completeness across all sections.
    Returns (coherence_issues, missing_sections, coherence_score).
    Fallback: ([], [], 5.0) on any error.
    """
    if not draft_sections:
        logger.warning("[CRITIC] No draft sections; coherence score defaults to 5.0")
        return [], [], 5.0

    sections_text = "\n\n".join(
        f"## {s['section_name'].replace('_', ' ').title()}\n"
        f"{s.get('content', '')[:_MAX_SECTION_CHARS]}"
        for s in draft_sections
    )

    prompt = _COHERENCE_PROMPT.format(
        outline=outline or "(No outline available)",
        sections_text=sections_text,
    )

    try:
        response = llm.invoke(prompt)
        data = json.loads(strip_fences(normalise_content(response.content)))
        coherence_issues = [str(x) for x in data.get("coherence_issues", []) if x]
        missing_sections = [str(x) for x in data.get("missing_sections", []) if x]
        score = _clamp_score(data.get("coherence_score", 5.0))
        logger.info(
            "[CRITIC] Coherence score=%.1f, issues=%d, missing=%s",
            score, len(coherence_issues), missing_sections,
        )
        return coherence_issues, missing_sections, score
    except Exception as exc:
        logger.error("[CRITIC] Coherence check failed (%s); defaulting to score=5.0", exc)
        return [], [], 5.0
