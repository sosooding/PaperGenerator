"""
Critic node helpers for Phase 4d.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage

from src.utils.llm import get_llm
from src.utils.llm_parse import normalise_content, strip_fences

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r'\[S(\d+)\]')
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

_GROUNDING_PROMPT = (
    "You are a fact-checking assistant.\n\n"
    "Abstract:\n{abstract}\n\n"
    "Claim:\n{sentence}\n\n"
    "Does this abstract directly support this claim?\n"
    "Answer YES or NO on the first line, then a one-sentence reason."
)

_COHERENCE_PROMPT = (
    "You are an academic paper reviewer.\n\n"
    "OUTLINE:\n{outline}\n\n"
    "SECTIONS:\n{sections_text}\n\n"
    "Evaluate the logical flow and completeness of this research paper draft.\n"
    "Return ONLY valid JSON (no markdown fences):\n"
    '{{"score": <float 0-10>, "missing_sections": ["..."], "coherence_issues": ["..."]}}'
)


def _extract_citation_sentences(
    sections: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """
    Returns (sentence, source_id, section_name) for every [Sx] tag found.
    A sentence with two tags produces two tuples.
    """
    results = []
    for section in sections:
        content = section.get("content", "")
        section_name = section.get("section_name", "")
        sentences = _SENTENCE_SPLIT_RE.split(content)
        for sentence in sentences:
            for match in _CITATION_RE.finditer(sentence):
                source_id = "S" + match.group(1)
                results.append((sentence.strip(), source_id, section_name))
    return results


def _check_grounding(
    pairs: List[Tuple[str, str, str]],
    citation_report: Dict[str, Any],
    llm,
) -> Tuple[List[Dict[str, str]], float]:
    """
    For each (sentence, source_id) pair, ask the LLM whether the abstract
    supports the claim. Returns (flagged_sentences, grounding_score 0-10).
    """
    flagged: List[Dict[str, str]] = []
    supported = 0
    total = 0

    for sentence, source_id, _section in pairs:
        paper = citation_report.get(source_id)
        if not paper:
            continue
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue

        total += 1
        # TODO(Option C): batch all claims into a single LLM call to avoid rate-limit delays
        if total > 1:
            time.sleep(4)  # stay under 15 req/min free-tier limit
        prompt = _GROUNDING_PROMPT.format(abstract=abstract, sentence=sentence)
        response = llm.invoke([HumanMessage(content=prompt)])
        text = normalise_content(response.content)
        first_line = text.splitlines()[0].strip().upper() if text.strip() else ""

        if first_line.startswith("NO"):
            lines = text.splitlines()
            reason = lines[1].strip() if len(lines) > 1 else "Unsupported claim."
            flagged.append({"sentence": sentence, "issue": reason, "source_id": source_id})
        else:
            supported += 1

    grounding_score = (supported / total * 10.0) if total > 0 else 10.0
    return flagged, grounding_score


def _score_coherence(
    outline: str,
    sections: List[Dict[str, Any]],
    llm,
) -> Tuple[float, List[str], List[str]]:
    """
    Ask the LLM to score the draft's coherence and completeness (0-10).
    Returns (coherence_score, missing_sections, coherence_issues).
    """
    sections_text = "\n\n".join(
        f"### {s.get('section_name', '').upper()}\n{s.get('content', '')}"
        for s in sections
    )
    prompt = _COHERENCE_PROMPT.format(
        outline=outline or "(no outline)",
        sections_text=sections_text or "(no sections)",
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = strip_fences(normalise_content(response.content))

    try:
        parsed = json.loads(text)
        score = float(parsed.get("score", 5.0))
        missing = list(parsed.get("missing_sections", []))
        issues = list(parsed.get("coherence_issues", []))
        return max(0.0, min(10.0, score)), missing, issues
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        m = re.search(r'"?score"?\s*[:\s]\s*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
        score = float(m.group(1)) if m else 5.0
        return max(0.0, min(10.0, score)), [], []
