# Graph Theory Research Paper Generator

An AI-powered system that generates academic research papers in graph theory using LangGraph, Gemini, and RAG.

## Features

- Automated literature retrieval from ArXiv and Semantic Scholar
- Research gap identification using LLM analysis (open conjectures, unexplored graph families, missing proofs)
- Interactive gap selection with human-in-the-loop approval
- Self-correcting paper generation with grounding checks
- Human-in-the-loop checkpoints at key stages
- Comprehensive evaluation metrics (NLI, BERTScore, citation verification)
- PDF output with proper citations in APA format

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Configurable — Google Gemini, Anthropic Claude, or OpenAI GPT
- **Vector Store**: ChromaDB
- **Embeddings**: Google text-embedding-004
- **Paper Sources**: ArXiv, Semantic Scholar
- **UI**: Streamlit
- **Document Generation**: Pandoc + LaTeX

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=graph-theory-paper-gen

# LLM provider — choose one: google (default), anthropic, openai
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash

# Optional overrides
EMBEDDING_MODEL=models/gemini-embedding-001
MAX_GAPS=5
```

Switch providers by changing `LLM_PROVIDER` and `LLM_MODEL`:

| Provider | `LLM_PROVIDER` | Example `LLM_MODEL` | Key needed |
|---|---|---|---|
| Google Gemini | `google` | `gemini-1.5-flash`, `gemini-2.0-flash-lite` | `GOOGLE_API_KEY` |
| Anthropic Claude | `anthropic` | `claude-3-5-haiku-20241022`, `claude-opus-4-8-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o-mini`, `gpt-4o` | `OPENAI_API_KEY` |

## Usage

```bash
# Run the Streamlit UI
streamlit run src/ui/app.py

# Or run programmatically
python -m src.main
```

## Project Structure

```
src/
├── agents/       # LangGraph agent nodes and graph definition
├── retrieval/    # RAG pipeline & paper fetching (Phase 2)
├── gap_finding/  # Research gap analysis via Gemini (Phase 3)
├── eval/         # Evaluation metrics (Phase 5)
├── ui/           # Streamlit interface
└── utils/        # Shared utilities (state schema, config)

tests/            # Unit & integration tests
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## License

MIT
