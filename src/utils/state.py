"""
State schema for the LangGraph research paper generation workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional


class Paper(TypedDict):
    """Represents a retrieved research paper."""
    title: str
    abstract: str
    authors: List[str]
    doi: Optional[str]
    pdf_url: Optional[str]
    source: str  # 'arxiv' or 'semantic_scholar'
    year: Optional[int]
    relevance_score: Optional[float]


class ResearchGap(TypedDict):
    """Represents an identified research gap."""
    gap_title: str
    description: str
    supporting_evidence: List[str]  # List of DOIs
    novelty_score: float  # 0-1 scale


class DraftSection(TypedDict):
    """Represents a section of the draft paper."""
    section_name: str
    content: str
    citations: List[str]  # List of SOURCE_IDs used


class CriticFeedback(TypedDict):
    """Feedback from the critic agent."""
    score: float  # 0-10 scale
    flagged_sentences: List[Dict[str, str]]  # {sentence, issue, source_id}
    missing_sections: List[str]
    coherence_issues: List[str]
    grounding_score: float  # 0-10 scale


class EvalScores(TypedDict):
    """Evaluation scores for the generated paper."""
    nli_distribution: Dict[str, int]  # {entailment, neutral, contradiction}
    bert_scores: Dict[str, float]  # section_name -> F1 score
    hallucinated_citations: List[str]
    llm_judge_scores: Dict[str, float]  # {grounding, novelty, completeness, coherence, tone}
    structural_completeness: Dict[str, bool]  # section_name -> pass/fail


class HumanDecision(TypedDict):
    """Captures human decisions at checkpoints."""
    checkpoint_name: str
    timestamp: str
    approved_papers: Optional[List[str]]  # List of DOIs
    removed_papers: Optional[List[str]]  # List of DOIs
    selected_gap: Optional[str]  # Gap title
    edited_sections: Optional[Dict[str, str]]  # section_name -> new_content
    notes: Optional[str]


class AgentState(TypedDict):
    """
    Global state shared across all LangGraph nodes.

    This state is passed through the entire workflow and updated by each agent.
    """
    # Input
    research_question: str

    # Phase 2: Retrieval
    sub_queries: List[str]
    retrieved_papers: List[Paper]
    embeddings_ready: bool

    # Phase 3: Gap Finding
    gaps: List[ResearchGap]
    selected_gap: Optional[ResearchGap]

    # Phase 4: Writing
    outline: Optional[str]
    draft_sections: List[DraftSection]
    critic_feedback: List[CriticFeedback]  # History of all feedback
    revision_count: int

    # Phase 5: Evaluation
    eval_scores: Optional[EvalScores]

    # Phase 6: Citations & Output
    citation_report: Dict[str, Any]  # Maps SOURCE_ID to full citation metadata

    # Human-in-the-loop
    human_decisions: List[HumanDecision]

    # Metadata
    current_phase: str
    errors: List[str]  # Track any errors encountered
