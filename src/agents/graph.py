"""
Main LangGraph workflow for research paper generation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

from src.utils.state import AgentState, ResearchGap
from src.utils.config import Config
from src.utils.llm import get_llm
from src.retrieval.paper_fetcher import fetch_papers_for_queries
from src.retrieval.vector_store import VectorStore
from src.gap_finding.gap_analyzer import find_research_gaps

logger = logging.getLogger(__name__)


# ============================================================================
# Node Implementations
# ============================================================================

def planner_node(state: AgentState) -> AgentState:
    """
    Decompose the research question into 6-8 sub-queries using Gemini.
    Falls back to the raw question on parse failure.
    """
    research_question = state.get("research_question", "")
    print(f"[PLANNER] Decomposing: {research_question!r}")

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
        # langchain-google-genai may return content as a list of parts or a plain string
        raw = response.content
        if isinstance(raw, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            )
        else:
            content = str(raw)
        content = content.strip()

        # Strip accidental markdown code fences
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        parsed = json.loads(content)
        if isinstance(parsed, list) and parsed:
            sub_queries = [str(q) for q in parsed]
        else:
            raise ValueError("Parsed value is not a non-empty list")
    except Exception as exc:
        msg = f"Planner LLM parse error ({exc}); using research question as single query"
        logger.warning(msg)
        errors.append(msg)

    print(f"[PLANNER] Generated {len(sub_queries)} sub-queries")
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

    print(f"[RETRIEVER] Fetching papers for {len(sub_queries)} sub-queries")

    # 1. Fetch raw papers from both sources
    papers = fetch_papers_for_queries(sub_queries)
    print(f"[RETRIEVER] Fetched {len(papers)} unique papers")

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

    # 2. Embed and store in ChromaDB (collection is cleared fresh each run by default)
    vector_store = VectorStore()
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

    # 3. Similarity search to rank, deduplicate, and threshold-filter
    relevant_papers = vector_store.get_relevant_papers(sub_queries)
    print(f"[RETRIEVER] Selected {len(relevant_papers)} relevant papers:")
    for paper in relevant_papers:
        print(f"  [{paper.get('year', '?')}] {paper['title']}")

    return {
        **state,
        "retrieved_papers": relevant_papers,
        "embeddings_ready": True,
        "current_phase": "retrieval",
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

    print(f"[GAP_FINDER] Analyzing {len(papers)} papers for research gaps")

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

    print(f"[GAP_FINDER] Found {len(gaps)} gaps")
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
        print("[GAP_SELECTOR] No gaps available; proceeding with no selected gap")
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

    # Resolve selection: accept 1-based integer or exact gap_title string
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

    print(f"[GAP_SELECTOR] Selected: {selected_gap['gap_title']!r} "
          f"(novelty={selected_gap['novelty_score']:.2f})")

    human_decisions = list(state.get("human_decisions", []))
    human_decisions.append({
        "checkpoint_name": "gap_selection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "approved_papers": [],
        "removed_papers": [],
        "selected_gap": selected_gap,
        "edited_sections": [],
        "notes": str(user_choice),
    })

    return {**state, "selected_gap": selected_gap,
            "human_decisions": human_decisions, "current_phase": "gap_selection"}


def writer_node(state: AgentState) -> AgentState:
    """
    Generate paper sections with grounded citations.

    Phase 4 will implement:
    - For each section (abstract, intro, related_work, gap_analysis, proposed_approach, conclusion)
    - Retrieve relevant chunks from ChromaDB
    - Generate content with [SOURCE_ID] tags
    - Enforce grounding requirement
    - Update state.draft_sections
    """
    selected_gap = state.get('selected_gap')
    gap_title = selected_gap.get('gap_title', 'N/A') if selected_gap else 'N/A'
    print(f"[WRITER] Selected gap: {gap_title}")
    return {**state, "current_phase": "writing"}


def critic_node(state: AgentState) -> AgentState:
    """
    Critique draft for grounding and coherence.

    Phase 4 will implement:
    - Grounding check: verify each [SOURCE_ID] supports the claim
    - Coherence check: evaluate logical flow and completeness
    - Score 0-10
    - Update state.critic_feedback
    """
    print(f"[CRITIC] Sections: {len(state.get('draft_sections', []))}")
    revision_count = state.get("revision_count", 0)
    return {**state, "current_phase": "critiquing", "revision_count": revision_count + 1}


def formatter_node(state: AgentState) -> AgentState:
    """
    Format citations and generate PDF.

    Phase 6 will implement:
    - Resolve [SOURCE_ID] to full metadata
    - Format as APA citations
    - Assemble markdown
    - Convert to PDF via Pandoc
    - Update state.citation_report
    """
    print(f"[FORMATTER] Finalizing paper...")
    return {**state, "current_phase": "formatting"}


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
        # No feedback yet, proceed to formatter
        return "formatter"

    latest_feedback = critic_feedback[-1]
    score = latest_feedback.get("score", 0)
    flagged_count = len(latest_feedback.get("flagged_sentences", []))
    revision_count = state.get("revision_count", 0)

    # Check if we've exceeded max revisions
    if revision_count >= Config.MAX_REVISION_CYCLES:
        print(f"[ROUTER] Max revisions reached ({revision_count}), proceeding to formatter")
        return "formatter"

    # Check if quality is acceptable
    if score >= Config.MIN_CRITIC_SCORE and flagged_count < Config.MAX_FLAGGED_SENTENCES:
        print(f"[ROUTER] Quality acceptable (score={score:.1f}), proceeding to formatter")
        return "formatter"

    print(f"[ROUTER] Needs revision (score={score:.1f}, flagged={flagged_count})")
    return "writer"


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph(checkpointer=None) -> StateGraph:
    """
    Build the LangGraph workflow.

    Nodes:
    1. planner       -> Query decomposition
    2. retriever     -> Paper retrieval
    3. gap_finder    -> Research gap identification (calls Gemini)
    4. gap_selector  -> Human-in-the-loop: user picks a gap (interrupt)
    5. writer        -> Section-by-section writing
    6. critic        -> Grounding & coherence check
    7. formatter     -> Citation formatting & PDF generation

    Flow:
    planner -> retriever -> gap_finder -> gap_selector* -> writer
           -> critic -> [revise loop or formatter] -> END
    (* pauses for user input via LangGraph interrupt)
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("gap_finder", gap_finder_node)
    workflow.add_node("gap_selector", gap_selector_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("formatter", formatter_node)

    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "gap_finder")
    workflow.add_edge("gap_finder", "gap_selector")
    workflow.add_edge("gap_selector", "writer")
    workflow.add_edge("writer", "critic")

    # Conditional edge: critic -> writer (revise) or formatter (accept)
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "writer": "writer",  # Needs revision
            "formatter": "formatter",  # Quality acceptable
        }
    )

    workflow.add_edge("formatter", END)

    # Compile with checkpointer
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
