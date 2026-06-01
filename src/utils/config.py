"""
Configuration and environment setup.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # LangSmith Tracing
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "graph-theory-paper-gen")

    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")   # google | anthropic | openai
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    # Rate Limiting
    GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
    BATCH_DELAY_SECONDS = int(os.getenv("BATCH_DELAY_SECONDS", "4"))

    # Database Paths
    CHECKPOINT_DB_PATH = "checkpoints.db"
    EVAL_DB_PATH = "evaluations.db"
    CHROMA_DB_PATH = "./chroma_data"

    # Retrieval Settings
    TOP_K_PAPERS = int(os.getenv("TOP_K_PAPERS", "20"))
    MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
    ARXIV_CATEGORY_FILTERS = [
        c.strip() for c in os.getenv("ARXIV_CATEGORY_FILTERS", "cs.DM,math.CO").split(",")
    ]
    SEMANTIC_SCHOLAR_FIELD = "Mathematics"

    # Generation Settings
    MAX_REVISION_CYCLES = 3
    MIN_CRITIC_SCORE = 7.0
    MAX_FLAGGED_SENTENCES = 3
    MAX_GAPS = int(os.getenv("MAX_GAPS", "5"))

    # Section Word Counts
    MIN_WORD_COUNTS = {
        "abstract": 150,
        "introduction": 300,
        "related_work": 400,
        "gap_analysis": 300,
        "proposed_approach": 400,
        "conclusion": 200,
    }

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment")
        if not cls.LANGCHAIN_API_KEY:
            raise ValueError("LANGCHAIN_API_KEY is not set in environment")


# Validate configuration on import
Config.validate()
