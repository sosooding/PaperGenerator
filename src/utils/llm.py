"""
Provider-agnostic LLM factory.

Set LLM_PROVIDER in .env to switch between providers:
  google    (default) — requires GOOGLE_API_KEY
  anthropic           — requires ANTHROPIC_API_KEY
  openai              — requires OPENAI_API_KEY
"""

from src.utils.config import Config


def get_llm(temperature: float = 0.3):
    """Return a LangChain chat model based on LLM_PROVIDER config."""
    provider = Config.LLM_PROVIDER.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=Config.LLM_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=temperature,
        )
    else:  # "google" (default)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=temperature,
        )
