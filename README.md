# Graph Theory Research Paper Generator

An AI-powered system that generates academic research papers in graph theory using LangGraph, Gemini, and RAG.

## Features

- Automated literature retrieval from ArXiv and Semantic Scholar
- Research gap identification using LLM analysis
- Self-correcting paper generation with grounding checks
- Human-in-the-loop checkpoints at key stages
- Comprehensive evaluation metrics (NLI, BERTScore, citation verification)
- PDF output with proper citations in APA format

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Google Gemini 3.1 Flash Lite
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
```

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
├── agents/       # LangGraph agent nodes
├── retrieval/    # RAG pipeline & paper fetching
├── eval/         # Evaluation metrics
├── ui/           # Streamlit interface
└── utils/        # Shared utilities

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
