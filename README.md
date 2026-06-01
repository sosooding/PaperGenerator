# Graph Theory Research Paper Generator

An AI-powered system that generates academic research papers in graph theory using LangGraph, Gemini, and RAG.

## Features

- Automated literature retrieval from ArXiv
- Research gap identification using LLM analysis (open conjectures, unexplored graph families, missing proofs)
- 3 human-in-the-loop checkpoints: paper approval, gap selection, draft review
- Outline-first section generation with hybrid RAG retrieval
- Self-correcting paper generation with LLM grounding checks and revision loop
- LaTeX PDF output with APA citations (`natbib`/`apalike`)
- Post-hoc evaluation metrics (NLI, BERTScore, citation verification)
- Configurable domain via ArXiv category filters (default: graph theory)

## Tech Stack

- **Orchestration**: LangGraph + SQLite checkpoints
- **LLM**: Configurable — Google Gemini, Anthropic Claude, or OpenAI GPT
- **Vector Store**: ChromaDB (persistent, keyed by research question)
- **Embeddings**: Google text-embedding-004
- **Paper Sources**: ArXiv (Semantic Scholar: coming soon)
- **Document Generation**: LaTeX + pdflatex, APA via natbib/apalike
- **UI**: Streamlit (planned — after CLI pipeline is complete)

## Pipeline

```
planner → retriever → paper_approver[INTERRUPT]
  → gap_finder → gap_selector[INTERRUPT]
  → writer → draft_reviewer[INTERRUPT]
  → critic → {revise or} → formatter → evaluator → END
```

| Step | Node | Description |
|------|------|-------------|
| 1 | `planner` | Decompose question into 6-8 sub-queries |
| 2 | `retriever` | Fetch ArXiv papers, embed, store in ChromaDB |
| 3 | `paper_approver` *(interrupt)* | User removes irrelevant papers |
| 4 | `gap_finder` | Identify 3-5 research gaps with novelty scores |
| 5 | `gap_selector` *(interrupt)* | User picks a gap by index |
| 6 | `writer` | Outline + 6 section calls with `[S1]` citation tags |
| 7 | `draft_reviewer` *(interrupt)* | User edits sections before critic runs |
| 8 | `critic` | LLM grounding check + coherence score; routes to revise loop or formatter |
| 9 | `formatter` | LaTeX + APA citations; compile PDF via pdflatex |
| 10 | `evaluator` | Post-hoc NLI/BERTScore/hallucination checks logged to `evaluations.db` |

## Implementation Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Scaffolding (state, config, graph) | ✅ Done |
| 2 | Retrieval (ArXiv, ChromaDB, embeddings) | ✅ Done |
| 3 | Gap finding + gap selector interrupt | ✅ Done |
| 4a | Paper approver interrupt | 🔧 TODO |
| 4b | Writer node (outline + sections) | 🔧 TODO |
| 4c | Draft reviewer interrupt | 🔧 TODO |
| 4d | Critic node (LLM grounding judge) | 🔧 TODO |
| 5 | Formatter (LaTeX + APA + PDF) | 🔧 TODO |
| 6 | Evaluator (NLI, BERTScore) | 🔧 TODO |
| 7 | Streamlit UI | 🔧 Deferred |

## Installation

```bash
python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

You will also need a LaTeX distribution (e.g., [MiKTeX](https://miktex.org/) on Windows) for PDF compilation.

## Configuration

Create a `.env` file:

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

# ArXiv domain filters (comma-separated). Default: graph theory / combinatorics
ARXIV_CATEGORY_FILTERS=cs.DM,math.CO
```

Switch providers by changing `LLM_PROVIDER` and `LLM_MODEL`:

| Provider | `LLM_PROVIDER` | Example `LLM_MODEL` | Key needed |
|---|---|---|---|
| Google Gemini | `google` | `gemini-1.5-flash`, `gemini-2.0-flash-lite` | `GOOGLE_API_KEY` |
| Anthropic Claude | `anthropic` | `claude-3-5-haiku-20241022`, `claude-opus-4-8-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o-mini`, `gpt-4o` | `OPENAI_API_KEY` |

## Usage

```bash
# Run programmatically (CLI)
python -m src.main
```

## Project Structure

```
src/
├── agents/       # LangGraph graph definition and all node functions
├── retrieval/    # RAG pipeline & ArXiv paper fetching
├── gap_finding/  # Research gap analysis via LLM
├── eval/         # Evaluation metrics (Phase 6)
├── ui/           # Streamlit interface (deferred)
└── utils/        # State schema, config, LLM factory

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