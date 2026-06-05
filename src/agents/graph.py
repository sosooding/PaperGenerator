"""
Main LangGraph workflow for research paper generation.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

from src.utils.state import AgentState, CriticFeedback, ResearchGap
from src.utils.config import Config
from src.utils.llm import get_llm
from src.utils.llm_parse import normalise_content, strip_fences
from src.retrieval.paper_fetcher import fetch_papers_for_queries
from src.retrieval.vector_store import VectorStore
from src.gap_finding.gap_analyzer import find_research_gaps
from src.agents.critic import run_grounding_check, run_coherence_check
from src.agents.interrupts import decode_draft_edits
from src.writing.writer import (
    build_source_map,
    generate_outline,
    generate_section,
    _extract_citations,
    SECTIONS,
    SECTION_RAG_SUFFIXES,
)
from src.formatting.formatter import format_paper

logger = logging.getLogger(__name__)


def _question_collection_name(question: str) -> str:
    """Derive a stable, isolated ChromaDB collection name from the research question."""
    digest = hashlib.md5(question.strip().lower().encode()).hexdigest()[:16]
    return f"papers_{digest}"


# ============================================================================
# Node Implementations
# ============================================================================

def planner_node(state: AgentState) -> AgentState:
    """
    Decompose the research question into 6-8 sub-queries using Gemini.
    Falls back to the raw question on parse failure.
    """
    research_question = state.get("research_question", "")
    logger.info("[PLANNER] Decomposing: %r", research_question)

    llm = get_llm(temperature=0)

    prompt = (
        "You are a research assistant specialising in graph theory.\n\n"
        "Break the following research question into 6-8 specific sub-queries suitable "
        "for searching academic literature databases (ArXiv, Semantic Scholar).\n"
        "Each sub-query should cover a distinct aspect of the topic.\n"
        "Return ONLY a valid JSON array of strings — no explanation, no markdown fences.\n\n"
        f"Research question: {research_question}"
    )

    errors = list(state.get("errors", []))
    sub_queries = [research_question]  # safe fallback

    try:
        response = llm.invoke(prompt)
        content = strip_fences(normalise_content(response.content))
        parsed = json.loads(content)
        if isinstance(parsed, list) and parsed:
            sub_queries = [str(q) for q in parsed]
        else:
            raise ValueError("Parsed value is not a non-empty list")
    except Exception as exc:
        msg = f"Planner LLM parse error ({exc}); using research question as single query"
        logger.warning(msg)
        errors.append(msg)

    logger.info("[PLANNER] Generated %d sub-queries", len(sub_queries))
    return {**state, "sub_queries": sub_queries, "current_phase": "planning", "errors": errors}


def retriever_node(state: AgentState) -> AgentState:
    """
    Fetch papers from ArXiv and Semantic Scholar, embed with text-embedding-004,
    store in ChromaDB, then return the top relevant papers via similarity search.
    """
    sub_queries = state.get("sub_queries", [])
    errors = list(state.get("errors", []))

    if not sub_queries:
        msg = "No sub-queries available; skipping retrieval"
        logger.warning(msg)
        errors.append(msg)
        return {**state, "current_phase": "retrieval", "errors": errors}

    logger.info("[RETRIEVER] Fetching papers for %d sub-queries", len(sub_queries))

    papers = fetch_papers_for_queries(sub_queries)
    logger.info("[RETRIEVER] Fetched %d unique papers", len(papers))

    if not papers:
        msg = "No papers retrieved from ArXiv or Semantic Scholar"
        logger.warning(msg)
        errors.append(msg)
        return {
            **state,
            "retrieved_papers": [],
            "embeddings_ready": False,
            "current_phase": "retrieval",
            "errors": errors,
        }

    collection_name = _question_collection_name(state.get("research_question", ""))
    vector_store = VectorStore(collection_name=collection_name)
    vector_store.embed_and_store(papers)

    if vector_store.collection.count() == 0:
        msg = "Embedding failed for all batches — ChromaDB collection is empty"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "retrieved_papers": [],
            "embeddings_ready": False,
            "current_phase": "retrieval",
            "errors": errors,
        }

    relevant_papers = vector_store.get_relevant_papers(sub_queries)
    logger.info(
        "[RETRIEVER] Selected %d relevant papers: %s",
        len(relevant_papers),
        ", ".join(f"[{p.get('year', '?')}] {p['title'][:40]}" for p in relevant_papers),
    )

    return {
        **state,
        "retrieved_papers": relevant_papers,
        "embeddings_ready": True,
        "current_phase": "retrieval",
        "errors": errors,
    }


def paper_approver_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop: display retrieved papers and allow the user to remove irrelevant ones.

    The graph pauses here via LangGraph interrupt. Resume by calling:
        graph.invoke(Command(resume=<choice>), config=...)
    where <choice> is a comma-separated list of 1-based indices to remove
    (e.g. "2,4") or an empty string to keep all papers.

    Removed papers are deleted from both state.retrieved_papers and ChromaDB.
    """
    papers = state.get("retrieved_papers", [])
    errors = list(state.get("errors", []))

    if not papers:
        return {**state, "current_phase": "paper_approval", "errors": errors}

    paper_list = [
        {
            "index": i + 1,
            "title": p["title"],
            "year": p.get("year", "?"),
            "relevance_score": round(p.get("relevance_score") or 0.0, 3),
        }
        for i, p in enumerate(papers)
    ]

    user_choice = interrupt({
        "message": (
            f"Retrieved {len(papers)} papers. "
            "Enter comma-separated indices to remove, or press Enter to keep all:"
        ),
        "papers": paper_list,
    })

    removed_indices: set = set()
    choice_str = str(user_choice).strip() if user_choice else ""
    if choice_str:
        for part in choice_str.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(papers):
                    removed_indices.add(idx)
            except ValueError:
                pass

    removed_papers = [papers[i] for i in sorted(removed_indices)]
    kept_papers = [p for i, p in enumerate(papers) if i not in removed_indices]

    if removed_papers:
        collection_name = _question_collection_name(state.get("research_question", ""))
        vector_store = VectorStore(collection_name=collection_name)
        vector_store.delete_papers(removed_papers)
        logger.info("[PAPER_APPROVER] Removed %d paper(s) from state and ChromaDB", len(removed_papers))

    logger.info("[PAPER_APPROVER] Keeping %d paper(s)", len(kept_papers))

    human_decisions = list(state.get("human_decisions", []))
    human_decisions.append({
        "checkpoint_name": "paper_approval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "approved_papers": [p.get("doi") or p["title"] for p in kept_papers],
        "removed_papers": [p.get("doi") or p["title"] for p in removed_papers],
        "selected_gap": None,
        "edited_sections": None,
        "notes": choice_str,
    })

    return {
        **state,
        "retrieved_papers": kept_papers,
        "human_decisions": human_decisions,
        "current_phase": "paper_approval",
        "errors": errors,
    }


def gap_finder_node(state: AgentState) -> AgentState:
    """
    Send all retrieved paper abstracts to Gemini and identify research gaps.
    Populates state.gaps; gap selection happens in gap_selector_node.
    """
    papers = state.get("retrieved_papers", [])
    research_question = state.get("research_question", "")
    errors = list(state.get("errors", []))

    logger.info("[GAP_FINDER] Analyzing %d papers for research gaps", len(papers))

    if not papers:
        msg = "gap_finder: no retrieved papers; skipping gap analysis"
        logger.warning(msg)
        errors.append(msg)
        return {**state, "gaps": [], "selected_gap": None,
                "current_phase": "gap_finding", "errors": errors}

    try:
        gaps = find_research_gaps(papers, research_question)
    except Exception as exc:
        msg = f"gap_finder: LLM error ({exc}); no gaps identified"
        logger.error(msg)
        errors.append(msg)
        return {**state, "gaps": [], "selected_gap": None,
                "current_phase": "gap_finding", "errors": errors}

    logger.info("[GAP_FINDER] Found %d gaps", len(gaps))
    return {**state, "gaps": gaps, "selected_gap": None,
            "current_phase": "gap_finding", "errors": errors}


def gap_selector_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop: display ranked gaps and pause for user selection.

    The graph pauses here via LangGraph interrupt. Resume by calling:
        graph.invoke(Command(resume=<choice>), config=...)
    where <choice> is a 1-based index (e.g. "2") or an exact gap_title string.
    """
    gaps = state.get("gaps", [])
    sorted_gaps = sorted(gaps, key=lambda g: g["novelty_score"], reverse=True)

    if not sorted_gaps:
        logger.info("[GAP_SELECTOR] No gaps available; proceeding with no selected gap")
        return {**state, "selected_gap": None, "current_phase": "gap_selection"}

    gap_list = [
        {
            "index": i + 1,
            "gap_title": g["gap_title"],
            "novelty_score": g["novelty_score"],
            "description": g["description"],
            "supporting_evidence": g["supporting_evidence"],
        }
        for i, g in enumerate(sorted_gaps)
    ]

    user_choice = interrupt({
        "message": (
            f"Found {len(sorted_gaps)} research gaps (ranked by novelty score). "
            "Enter the number of the gap you want to investigate:"
        ),
        "gaps": gap_list,
    })

    selected_gap: Optional[ResearchGap] = None
    try:
        idx = int(user_choice) - 1
        if 0 <= idx < len(sorted_gaps):
            selected_gap = sorted_gaps[idx]
    except (ValueError, TypeError):
        title = str(user_choice).strip().lower()
        for g in sorted_gaps:
            if g["gap_title"].lower() == title:
                selected_gap = g
                break

    if selected_gap is None:
        logger.warning("gap_selector: unrecognised choice %r; defaulting to highest novelty", user_choice)
        selected_gap = sorted_gaps[0]

    logger.info(
        "[GAP_SELECTOR] Selected: %r (novelty=%.2f)",
        selected_gap["gap_title"],
        selected_gap["novelty_score"],
    )

    human_decisions = list(state.get("human_decisions", []))
    # FIX-3: use None for not-applicable fields (schema: Optional[...]).
    human_decisions.append({
        "checkpoint_name": "gap_selection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "approved_papers": None,
        "removed_papers": None,
        "selected_gap": selected_gap,
        "edited_sections": None,
        "notes": str(user_choice),
    })

    return {**state, "selected_gap": selected_gap,
            "human_decisions": human_decisions, "current_phase": "gap_selection"}


def writer_node(state: AgentState) -> AgentState:
    """
    Phase 4b: generate paper outline then all 6 sections with grounded [Sx] citations.
    Step 1 — outline (1 LLM call): section headers + 2-3 bullets each.
    Step 2 — sections (6 LLM calls): per-section hybrid RAG context + source_map.
    """
    selected_gap = state.get("selected_gap")
    research_question = state.get("research_question", "")
    papers = state.get("retrieved_papers", [])
    errors = list(state.get("errors", []))

    if not selected_gap:
        msg = "writer_node: no selected_gap in state; skipping writing"
        logger.warning(msg)
        errors.append(msg)
        return {**state, "current_phase": "writing", "errors": errors}

    logger.info("[WRITER] Gap: %r", selected_gap["gap_title"])
    logger.info("[WRITER] Writing with %d source papers", len(papers))

    source_map = build_source_map(papers)
    llm = get_llm(temperature=0.7)

    # Step 1: outline
    try:
        outline = generate_outline(research_question, selected_gap, source_map, llm)
        logger.info("[WRITER] Outline generated (%d chars)", len(outline))
    except Exception as exc:
        msg = f"writer_node: outline generation failed ({exc})"
        logger.error(msg)
        errors.append(msg)
        outline = ""

    # Step 2: one LLM call per section with hybrid RAG context
    collection_name = _question_collection_name(research_question)
    vector_store = VectorStore(collection_name=collection_name)
    draft_sections = []

    for section_name in SECTIONS:
        try:
            rag_query = f"{selected_gap['description']} — {SECTION_RAG_SUFFIXES[section_name]}"
            rag_papers = vector_store.similarity_search(rag_query, k=5)
            section = generate_section(
                section_name=section_name,
                research_question=research_question,
                gap=selected_gap,
                source_map=source_map,
                outline=outline,
                rag_papers=rag_papers,
                llm=llm,
            )
            draft_sections.append(section)
            logger.info(
                "[WRITER] %s: %d words, %d citations",
                section_name,
                len(section["content"].split()),
                len(section["citations"]),
            )
        except Exception as exc:
            msg = f"writer_node: section '{section_name}' failed ({exc})"
            logger.error(msg)
            errors.append(msg)
            draft_sections.append({
                "section_name": section_name,
                "content": f"[Generation failed: {exc}]",
                "citations": [],
            })

    return {
        **state,
        "outline": outline,
        "draft_sections": draft_sections,
        "citation_report": {sid: paper for sid, paper in source_map.items()},
        "current_phase": "writing",
        "errors": errors,
    }


def draft_reviewer_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop: display outline + all draft sections and allow the user to edit any of them.

    The graph pauses here via LangGraph interrupt. Resume by calling:
        graph.invoke(Command(resume=<choice>), config=...)
    where <choice> is a JSON-encoded dict mapping section_name -> new_content
    (e.g. '{"abstract": "revised text..."}') or an empty string to accept all sections as-is.
    """
    draft_sections = list(state.get("draft_sections", []))
    errors = list(state.get("errors", []))

    if not draft_sections:
        return {**state, "current_phase": "draft_review", "errors": errors}

    section_list = [
        {
            "section_name": s["section_name"],
            "content": s["content"],
            "citations": s["citations"],
        }
        for s in draft_sections
    ]

    user_choice = interrupt({
        "message": (
            "Review the draft sections. "
            "Enter edits as JSON {section_name: new_content}, or press Enter to accept all:"
        ),
        "outline": state.get("outline", ""),
        "sections": section_list,
    })

    choice_str = str(user_choice).strip() if user_choice else ""
    edited_sections = decode_draft_edits(choice_str)

    # FIX-1: re-extract citations from the new content so they stay in sync.
    updated_sections = []
    for section in draft_sections:
        name = section["section_name"]
        if name in edited_sections:
            new_content = edited_sections[name]
            updated_sections.append({
                **section,
                "content": new_content,
                "citations": _extract_citations(new_content),
            })
        else:
            updated_sections.append(section)

    if edited_sections:
        logger.info("[DRAFT_REVIEWER] Edited sections: %s", list(edited_sections.keys()))
    else:
        logger.info("[DRAFT_REVIEWER] No edits; accepting all sections")

    human_decisions = list(state.get("human_decisions", []))
    human_decisions.append({
        "checkpoint_name": "draft_review",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "approved_papers": None,
        "removed_papers": None,
        "selected_gap": None,
        "edited_sections": edited_sections,
        "notes": choice_str,
    })

    return {
        **state,
        "draft_sections": updated_sections,
        "human_decisions": human_decisions,
        "current_phase": "draft_review",
        "errors": errors,
    }


def critic_node(state: AgentState) -> AgentState:
    """
    Phase 4d: critique the draft with exactly 2 LLM calls.

    Call 1 — grounding check: batch all cited sentences + source excerpts into one
    prompt; LLM returns flagged sentences and a grounding score.

    Call 2 — coherence check: send truncated section excerpts + outline; LLM returns
    coherence issues, missing sections, and a coherence score.

    A BATCH_DELAY_SECONDS pause separates the two calls to respect Gemini's RPM limit.
    On any LLM / parse failure the node falls back gracefully (score=5.0, empty lists).
    """
    draft_sections = state.get("draft_sections", [])
    citation_report = state.get("citation_report", {})
    outline = state.get("outline", "")
    errors = list(state.get("errors", []))
    critic_feedback = list(state.get("critic_feedback", []))
    revision_count = state.get("revision_count", 0)

    if not draft_sections:
        msg = "critic_node: no draft sections; skipping critique"
        logger.warning(msg)
        errors.append(msg)
        return {
            **state,
            "current_phase": "critiquing",
            "revision_count": revision_count + 1,
            "errors": errors,
        }

    logger.info("[CRITIC] Critiquing %d sections (revision %d)", len(draft_sections), revision_count + 1)

    llm = get_llm(temperature=0)

    # Call 1: grounding
    flagged, grounding_score = run_grounding_check(draft_sections, citation_report, llm)

    # Rate-limit pause before the second call
    time.sleep(Config.BATCH_DELAY_SECONDS)

    # Call 2: coherence
    coherence_issues, missing_sections, coherence_score = run_coherence_check(
        draft_sections, outline, llm
    )

    overall_score = round((grounding_score + coherence_score) / 2.0, 2)

    feedback: CriticFeedback = {
        "score": overall_score,
        "flagged_sentences": flagged,
        "missing_sections": missing_sections,
        "coherence_issues": coherence_issues,
        "grounding_score": round(grounding_score, 2),
    }
    critic_feedback.append(feedback)

    logger.info(
        "[CRITIC] score=%.1f (grounding=%.1f, coherence=%.1f), flagged=%d, issues=%d",
        overall_score, grounding_score, coherence_score, len(flagged), len(coherence_issues),
    )

    return {
        **state,
        "critic_feedback": critic_feedback,
        "revision_count": revision_count + 1,
        "current_phase": "critiquing",
        "errors": errors,
    }


def formatter_node(state: AgentState) -> AgentState:
    """
    Phase 5: assemble LaTeX + BibTeX, write output files, attempt PDF compilation.
    Updates citation_report with tex_path, bib_path, pdf_path, reference_order, sid_to_apa.
    """
    draft_sections = state.get("draft_sections", [])
    citation_report = state.get("citation_report", {})
    research_question = state.get("research_question", "")
    selected_gap = state.get("selected_gap")
    gap_title = selected_gap["gap_title"] if selected_gap else research_question
    errors = list(state.get("errors", []))

    logger.info("[FORMATTER] Finalizing paper with %d sections...", len(draft_sections))

    try:
        updated_report = format_paper(
            draft_sections=draft_sections,
            citation_report=citation_report,
            research_question=research_question,
            gap_title=gap_title,
        )
    except Exception as exc:
        msg = f"formatter_node: formatting failed ({exc})"
        logger.error(msg)
        errors.append(msg)
        updated_report = citation_report

    return {**state, "citation_report": updated_report, "current_phase": "formatting", "errors": errors}


# ============================================================================
# Conditional Routing
# ============================================================================

def should_revise(state: AgentState) -> Literal["writer", "formatter"]:
    """
    Route back to writer if critic score is low or too many flagged sentences.
    Otherwise, proceed to formatter.
    """
    critic_feedback = state.get("critic_feedback", [])
    if not critic_feedback:
        return "formatter"

    latest_feedback = critic_feedback[-1]
    score = latest_feedback.get("score", 0)
    flagged_count = len(latest_feedback.get("flagged_sentences", []))
    revision_count = state.get("revision_count", 0)

    if revision_count >= Config.MAX_REVISION_CYCLES:
        logger.info("[ROUTER] Max revisions reached (%d), proceeding to formatter", revision_count)
        return "formatter"

    if score >= Config.MIN_CRITIC_SCORE and flagged_count < Config.MAX_FLAGGED_SENTENCES:
        logger.info("[ROUTER] Quality acceptable (score=%.1f), proceeding to formatter", score)
        return "formatter"

    logger.info("[ROUTER] Needs revision (score=%.1f, flagged=%d)", score, flagged_count)
    return "writer"


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph(checkpointer=None) -> StateGraph:
    """
    Build the LangGraph workflow.

    Nodes:
    1. planner         -> Query decomposition
    2. retriever       -> Paper retrieval & ChromaDB embedding
    3. paper_approver  -> Human-in-the-loop: user removes irrelevant papers (interrupt)
    4. gap_finder      -> Research gap identification
    5. gap_selector    -> Human-in-the-loop: user picks a gap (interrupt)
    6. writer          -> Outline + section-by-section writing
    7. draft_reviewer  -> Human-in-the-loop: user edits sections (interrupt)
    8. critic          -> Grounding & coherence check
    9. formatter       -> LaTeX + APA citation PDF generation
    10. evaluator      -> Post-hoc NLI/BERTScore metrics [TODO]

    Flow:
    planner -> retriever -> paper_approver* -> gap_finder -> gap_selector*
           -> writer -> draft_reviewer* -> critic -> [revise loop or formatter] -> END
    (* pauses for user input via LangGraph interrupt)
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("paper_approver", paper_approver_node)
    workflow.add_node("gap_finder", gap_finder_node)
    workflow.add_node("gap_selector", gap_selector_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("draft_reviewer", draft_reviewer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("formatter", formatter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "paper_approver")
    workflow.add_edge("paper_approver", "gap_finder")
    workflow.add_edge("gap_finder", "gap_selector")
    workflow.add_edge("gap_selector", "writer")
    workflow.add_edge("writer", "draft_reviewer")
    workflow.add_edge("draft_reviewer", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "writer": "writer",
            "formatter": "formatter",
        }
    )

    workflow.add_edge("formatter", END)

    return workflow.compile(checkpointer=checkpointer)


def get_initial_state(research_question: str) -> AgentState:
    """Create initial state for a new paper generation run."""
    return {
        "research_question": research_question,
        "sub_queries": [],
        "retrieved_papers": [],
        "embeddings_ready": False,
        "gaps": [],
        "selected_gap": None,
        "outline": None,
        "draft_sections": [],
        "critic_feedback": [],
        "revision_count": 0,
        "eval_scores": None,
        "citation_report": {},
        "human_decisions": [],
        "current_phase": "initialized",
        "errors": [],
    }
