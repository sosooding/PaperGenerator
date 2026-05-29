"""
Main LangGraph workflow for research paper generation.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.utils.state import AgentState
from src.utils.config import Config


# ============================================================================
# Node Implementations (Stubs for Phase 1)
# ============================================================================

def planner_node(state: AgentState) -> AgentState:
    """
    Decompose research question into sub-queries.

    Phase 2 will implement:
    - Gemini call to decompose research_question into 6-8 sub-queries
    - Parse response into list
    - Update state.sub_queries
    """
    print(f"[PLANNER] Processing: {state.get('research_question', 'N/A')}")
    return {**state, "current_phase": "planning"}


def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant papers from ArXiv and Semantic Scholar.

    Phase 2 will implement:
    - Fetch papers from ArXiv (cs.DM, math.CO categories)
    - Fetch papers from Semantic Scholar
    - Embed abstracts using text-embedding-004
    - Store in ChromaDB
    - Run similarity search for each sub-query
    - Update state.retrieved_papers
    """
    print(f"[RETRIEVER] Sub-queries: {len(state.get('sub_queries', []))}")
    return {**state, "current_phase": "retrieval"}


def gap_finder_node(state: AgentState) -> AgentState:
    """
    Identify research gaps from retrieved papers.

    Phase 3 will implement:
    - Send all approved abstracts to Gemini (1M context window)
    - Identify open conjectures, unexplored graph families, missing proofs
    - Structure gaps as JSON with novelty scores
    - Update state.gaps
    """
    print(f"[GAP_FINDER] Papers: {len(state.get('retrieved_papers', []))}")
    return {**state, "current_phase": "gap_finding"}


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
    1. planner -> Query decomposition
    2. retriever -> Paper retrieval (with interrupt for human checkpoint 1)
    3. gap_finder -> Research gap identification (with interrupt for human checkpoint 2)
    4. writer -> Section-by-section writing
    5. critic -> Grounding & coherence check
    6. formatter -> Citation formatting & PDF generation (with interrupt for human checkpoint 3)

    Flow:
    planner -> retriever -> gap_finder -> writer -> critic -> [revise loop or formatter] -> END
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("gap_finder", gap_finder_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("formatter", formatter_node)

    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "gap_finder")
    workflow.add_edge("gap_finder", "writer")
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
