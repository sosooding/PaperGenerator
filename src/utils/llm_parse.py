"""
Shared utilities for normalising and cleaning LLM output.

Centralises the two patterns duplicated across planner_node, writer.py,
and gap_analyzer.py (REF-1).
"""


def normalise_content(raw) -> str:
    """Coerce an LLM response's .content to a plain string.

    langchain-google-genai may return content as a list of part-dicts;
    all other providers return a plain string.
    """
    if isinstance(raw, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        ).strip()
    return str(raw).strip()


def strip_fences(content: str) -> str:
    """Remove ``` markdown code fences that LLMs sometimes add around output."""
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return content
