"""
Encode/decode helpers for LangGraph interrupt payloads (REF-3).

Centralises the JSON round-trip for the draft-review interrupt so the
node (graph.py) and the CLI (main.py) share exactly one implementation.
"""

import json
import logging

logger = logging.getLogger(__name__)


def encode_draft_edits(edits: dict) -> str:
    """Serialize a {section_name: new_content} dict as the resume value."""
    return json.dumps(edits)


def decode_draft_edits(raw: str) -> dict:
    """
    Deserialize the draft-review resume value.

    Returns {} for empty input (accept all) or any malformed input.
    The caller should treat {} as "no edits requested".
    """
    text = str(raw).strip() if raw else ""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
        logger.warning(
            "decode_draft_edits: expected a JSON object, got %s; no edits applied",
            type(parsed).__name__,
        )
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("decode_draft_edits: parse error (%s); no edits applied", exc)
        return {}
