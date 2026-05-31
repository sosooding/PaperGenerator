"""
Unit tests for configuration.
"""

import pytest
from src.utils.config import Config


def test_config_has_api_keys():
    """Test that API keys are loaded."""
    assert Config.GOOGLE_API_KEY is not None
    assert Config.LANGCHAIN_API_KEY is not None
    assert len(Config.GOOGLE_API_KEY) > 0
    assert len(Config.LANGCHAIN_API_KEY) > 0


def test_config_model_settings():
    """Test model configuration."""
    assert Config.LLM_PROVIDER is not None
    assert Config.LLM_MODEL is not None
    assert Config.EMBEDDING_MODEL is not None
    assert Config.LLM_PROVIDER in ("google", "anthropic", "openai")
    assert len(Config.LLM_MODEL) > 0
    assert "embedding" in Config.EMBEDDING_MODEL.lower()


def test_config_rate_limits():
    """Test rate limiting configuration."""
    assert Config.GEMINI_RPM_LIMIT > 0
    assert Config.BATCH_SIZE > 0
    assert Config.BATCH_DELAY_SECONDS >= 0


def test_config_database_paths():
    """Test database path configuration."""
    assert Config.CHECKPOINT_DB_PATH.endswith(".db")
    assert Config.EVAL_DB_PATH.endswith(".db")
    assert "chroma" in Config.CHROMA_DB_PATH.lower()


def test_config_retrieval_settings():
    """Test retrieval configuration."""
    assert Config.TOP_K_PAPERS > 0
    assert len(Config.ARXIV_CATEGORY_FILTERS) > 0
    assert "cs.DM" in Config.ARXIV_CATEGORY_FILTERS
    assert "math.CO" in Config.ARXIV_CATEGORY_FILTERS
    assert Config.SEMANTIC_SCHOLAR_FIELD == "Mathematics"


def test_config_generation_settings():
    """Test generation configuration."""
    assert Config.MAX_REVISION_CYCLES > 0
    assert Config.MIN_CRITIC_SCORE > 0
    assert Config.MAX_FLAGGED_SENTENCES >= 0


def test_config_min_word_counts():
    """Test section word count requirements."""
    expected_sections = [
        "abstract",
        "introduction",
        "related_work",
        "gap_analysis",
        "proposed_approach",
        "conclusion",
    ]

    for section in expected_sections:
        assert section in Config.MIN_WORD_COUNTS
        assert Config.MIN_WORD_COUNTS[section] > 0


def test_config_validation():
    """Test that validation doesn't raise errors."""
    # This should not raise if .env is properly configured
    try:
        Config.validate()
    except ValueError as e:
        pytest.fail(f"Config validation failed: {e}")
