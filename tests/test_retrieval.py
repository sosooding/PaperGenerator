"""
Unit tests for Phase 2: paper fetching and vector store (all external calls mocked).
"""

import pytest
from unittest.mock import patch, MagicMock, call
from typing import List

from src.utils.state import Paper
from src.utils.config import Config
from src.retrieval.paper_fetcher import (
    fetch_arxiv_papers,
    fetch_semantic_scholar_papers,
    fetch_papers_for_queries,
    _paper_unique_key,
    _author_name,
)
import google.genai as genai

from src.retrieval.vector_store import (
    VectorStore,
    _chroma_id,
    _metadata_from_paper,
    _paper_from_chroma_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_paper(
    title: str = "Graph Coloring",
    abstract: str = "We study chromatic polynomials.",
    authors: List[str] = None,
    doi: str = "10.1234/test",
    source: str = "arxiv",
    year: int = 2024,
) -> Paper:
    return {
        "title": title,
        "abstract": abstract,
        "authors": authors or ["Alice", "Bob"],
        "doi": doi,
        "pdf_url": None,
        "source": source,
        "year": year,
        "relevance_score": None,
    }


# ---------------------------------------------------------------------------
# paper_fetcher helpers
# ---------------------------------------------------------------------------

class TestAuthorName:
    def test_dict_author(self):
        assert _author_name({"name": "Alice"}) == "Alice"

    def test_dict_author_missing_key(self):
        assert _author_name({}) is None

    def test_object_author(self):
        obj = MagicMock()
        obj.name = "Bob"
        assert _author_name(obj) == "Bob"

    def test_object_without_name(self):
        obj = MagicMock(spec=[])  # no attributes
        assert _author_name(obj) is None


class TestPaperUniqueKey:
    def test_doi_preferred(self):
        assert _paper_unique_key("10.1234/test", "Some title") == "10.1234/test"

    def test_doi_lowercased(self):
        assert _paper_unique_key("10.1234/TEST", "Title") == "10.1234/test"

    def test_fallback_to_title(self):
        key = _paper_unique_key(None, "Graph Theory Paper")
        assert key.startswith("title:")
        assert "graph theory paper" in key

    def test_title_truncated_at_80(self):
        long_title = "A" * 200
        key = _paper_unique_key(None, long_title)
        assert len(key) <= len("title:") + 80


# ---------------------------------------------------------------------------
# fetch_arxiv_papers (mocked)
# ---------------------------------------------------------------------------

class TestFetchArxivPapers:
    def _make_arxiv_result(self, title="Test Paper", doi="10.1/test"):
        result = MagicMock()
        result.title = title
        result.summary = "Abstract text."
        result.authors = [MagicMock(name="Author One")]
        result.authors[0].name = "Author One"
        result.doi = doi
        result.pdf_url = "http://example.com/paper.pdf"
        result.published = MagicMock()
        result.published.year = 2024
        return result

    @patch("src.retrieval.paper_fetcher._arxiv_client")
    def test_returns_paper_list(self, mock_client):
        mock_client.results.return_value = [self._make_arxiv_result()]

        papers = fetch_arxiv_papers("graph coloring", max_results=5)

        assert len(papers) == 1
        assert papers[0]["title"] == "Test Paper"
        assert papers[0]["source"] == "arxiv"
        assert papers[0]["doi"] == "10.1/test"
        assert papers[0]["year"] == 2024

    @patch("src.retrieval.paper_fetcher._arxiv_client")
    def test_category_filter_in_query(self, mock_client):
        mock_client.results.return_value = []

        fetch_arxiv_papers("spanning trees", max_results=3)

        # Verify Search was called with category filters
        search_arg = mock_client.results.call_args[0][0]
        assert "cat:" in search_arg.query
        assert "spanning trees" in search_arg.query

    @patch("src.retrieval.paper_fetcher._arxiv_client")
    def test_handles_no_doi(self, mock_client):
        result = self._make_arxiv_result(doi=None)
        result.doi = None
        mock_client.results.return_value = [result]

        papers = fetch_arxiv_papers("query")

        assert papers[0]["doi"] is None


# ---------------------------------------------------------------------------
# fetch_semantic_scholar_papers (mocked)
# ---------------------------------------------------------------------------

class TestFetchSemanticScholarPapers:
    def _make_ss_item(self, title="SS Paper", doi="10.2/sstest", has_abstract=True):
        item = MagicMock()
        item.title = title
        item.abstract = "Abstract." if has_abstract else None
        item.authors = [{"name": "Carol"}, {"name": "Dave"}]
        item.year = 2023
        item.externalIds = {"DOI": doi}
        item.openAccessPdf = {"url": "http://example.com/ss.pdf"}
        return item

    @patch("src.retrieval.paper_fetcher.SemanticScholar")
    def test_returns_paper_list(self, mock_ss_cls):
        mock_ss = MagicMock()
        mock_ss_cls.return_value = mock_ss
        mock_ss.search_paper.return_value = [self._make_ss_item()]

        papers = fetch_semantic_scholar_papers("planar graph", max_results=5)

        assert len(papers) == 1
        assert papers[0]["source"] == "semantic_scholar"
        assert papers[0]["doi"] == "10.2/sstest"
        assert papers[0]["authors"] == ["Carol", "Dave"]

    @patch("src.retrieval.paper_fetcher.SemanticScholar")
    def test_skips_papers_without_abstract(self, mock_ss_cls):
        mock_ss = MagicMock()
        mock_ss_cls.return_value = mock_ss
        mock_ss.search_paper.return_value = [self._make_ss_item(has_abstract=False)]

        papers = fetch_semantic_scholar_papers("query")

        assert len(papers) == 0

    @patch("src.retrieval.paper_fetcher.SemanticScholar")
    def test_handles_missing_pdf_and_doi(self, mock_ss_cls):
        mock_ss = MagicMock()
        mock_ss_cls.return_value = mock_ss
        item = self._make_ss_item()
        item.externalIds = {}
        item.openAccessPdf = None
        mock_ss.search_paper.return_value = [item]

        papers = fetch_semantic_scholar_papers("query")

        assert papers[0]["doi"] is None
        assert papers[0]["pdf_url"] is None


# ---------------------------------------------------------------------------
# fetch_papers_for_queries (mocked)
# ---------------------------------------------------------------------------

class TestFetchPapersForQueries:
    @patch("src.retrieval.paper_fetcher.fetch_arxiv_papers")
    @patch("src.retrieval.paper_fetcher.time.sleep")
    def test_deduplicates_by_doi(self, mock_sleep, mock_arxiv):
        # Same paper returned for two different sub-queries → deduped
        paper = make_paper(doi="10.1/same")
        mock_arxiv.return_value = [paper]

        result = fetch_papers_for_queries(["query1", "query2"])

        assert len(result) == 1

    @patch("src.retrieval.paper_fetcher.fetch_arxiv_papers")
    @patch("src.retrieval.paper_fetcher.time.sleep")
    def test_combines_unique_papers(self, mock_sleep, mock_arxiv):
        mock_arxiv.side_effect = [
            [make_paper(doi="10.1/a")],
            [make_paper(doi="10.1/b")],
        ]

        result = fetch_papers_for_queries(["q1", "q2"])

        assert len(result) == 2

    @patch("src.retrieval.paper_fetcher.fetch_arxiv_papers")
    @patch("src.retrieval.paper_fetcher.time.sleep")
    def test_handles_arxiv_failure_gracefully(self, mock_sleep, mock_arxiv):
        mock_arxiv.side_effect = Exception("network error")

        result = fetch_papers_for_queries(["query"])

        assert len(result) == 0  # ArXiv failed, SS disabled — no papers

    @patch("src.retrieval.paper_fetcher.fetch_arxiv_papers")
    @patch("src.retrieval.paper_fetcher.time.sleep")
    def test_sleeps_between_queries(self, mock_sleep, mock_arxiv):
        mock_arxiv.return_value = []

        fetch_papers_for_queries(["q1", "q2", "q3"])

        assert mock_sleep.call_count == 3


# ---------------------------------------------------------------------------
# vector_store helpers
# ---------------------------------------------------------------------------

class TestVectorStoreHelpers:
    def test_chroma_id_uses_doi(self):
        paper = make_paper(doi="10.1234/graph-theory")
        cid = _chroma_id(paper)
        assert "/" not in cid
        assert ":" not in cid

    def test_chroma_id_fallback_title(self):
        paper = make_paper(doi=None, title="Graph Theory Paper")
        cid = _chroma_id(paper)
        assert len(cid) > 0
        assert "/" not in cid

    def test_metadata_from_paper(self):
        paper = make_paper(doi="10.1/x", year=2024)
        meta = _metadata_from_paper(paper)
        assert meta["title"] == paper["title"]
        assert meta["year"] == "2024"
        assert meta["doi"] == "10.1/x"
        assert "Alice" in meta["authors"]

    def test_paper_from_chroma_row(self):
        meta = {
            "title": "Chromatic Polynomials",
            "authors": "Alice, Bob",
            "source": "arxiv",
            "year": "2023",
            "doi": "10.1/cp",
            "pdf_url": "",
        }
        paper = _paper_from_chroma_row(meta, "Abstract text.", 0.2)
        assert paper["title"] == "Chromatic Polynomials"
        assert paper["authors"] == ["Alice", "Bob"]
        assert paper["year"] == 2023
        assert paper["relevance_score"] == pytest.approx(0.8, abs=0.001)

    def test_paper_from_chroma_row_empty_year(self):
        meta = {"title": "T", "authors": "", "source": "s", "year": "", "doi": "", "pdf_url": ""}
        paper = _paper_from_chroma_row(meta, "abstract", 0.0)
        assert paper["year"] is None


# ---------------------------------------------------------------------------
# VectorStore (mocked chromadb + embedder)
# ---------------------------------------------------------------------------

def _make_embedding_response(vectors: List[List[float]]):
    """Build a fake google.genai EmbedContentResponse."""
    response = MagicMock()
    response.embeddings = [MagicMock(values=v) for v in vectors]
    return response


class TestVectorStore:
    def _make_store(self, clear: bool = False):
        with patch("src.retrieval.vector_store.chromadb.PersistentClient") as mock_chroma_cls, \
             patch("src.retrieval.vector_store.genai.Client") as mock_genai_cls:

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_chroma_cls.return_value.get_or_create_collection.return_value = mock_collection

            mock_genai = MagicMock()
            mock_genai_cls.return_value = mock_genai

            store = VectorStore(persist_dir="./test_chroma", clear=clear)
            return store, mock_collection, mock_genai, mock_chroma_cls.return_value

    def test_embed_and_store_calls_upsert(self):
        store, mock_col, mock_genai, _ = self._make_store()
        mock_genai.models.embed_content.return_value = _make_embedding_response([[0.1, 0.2]])

        papers = [make_paper()]
        store.embed_and_store(papers)

        mock_genai.models.embed_content.assert_called_once()
        mock_col.upsert.assert_called_once()

    def test_embed_and_store_skips_papers_without_abstract(self):
        store, mock_col, mock_genai, _ = self._make_store()

        paper = make_paper()
        paper["abstract"] = ""
        store.embed_and_store([paper])

        mock_col.upsert.assert_not_called()

    def test_clear_true_deletes_collection(self):
        store, mock_col, mock_genai, mock_chroma = self._make_store(clear=True)
        mock_chroma.delete_collection.assert_called_once_with("papers")

    def test_clear_false_does_not_delete_collection(self):
        store, mock_col, mock_genai, mock_chroma = self._make_store(clear=False)
        mock_chroma.delete_collection.assert_not_called()

    def test_similarity_search_returns_empty_on_empty_collection(self):
        store, mock_col, mock_genai, _ = self._make_store()
        mock_col.count.return_value = 0

        result = store.similarity_search("query")

        assert result == []

    def test_similarity_search_returns_papers(self):
        store, mock_col, mock_genai, _ = self._make_store()
        mock_col.count.return_value = 2
        mock_genai.models.embed_content.return_value = _make_embedding_response([[0.1, 0.2]])
        mock_col.query.return_value = {
            "ids": [["id1", "id2"]],
            "metadatas": [[
                {"title": "P1", "authors": "A", "source": "arxiv", "year": "2023", "doi": "10.1/p1", "pdf_url": ""},
                {"title": "P2", "authors": "B", "source": "semantic_scholar", "year": "2022", "doi": "", "pdf_url": ""},
            ]],
            "documents": [["Abstract 1", "Abstract 2"]],
            "distances": [[0.1, 0.3]],
        }

        result = store.similarity_search("graph coloring", k=2)

        assert len(result) == 2
        assert result[0]["title"] == "P1"
        assert result[0]["relevance_score"] == pytest.approx(0.9, abs=0.001)

    def test_get_relevant_papers_deduplicates(self):
        store, mock_col, mock_genai, _ = self._make_store()
        mock_col.count.return_value = 1
        mock_genai.models.embed_content.return_value = _make_embedding_response([[0.1]])
        shared_meta = {"title": "P1", "authors": "A", "source": "arxiv", "year": "2023", "doi": "10.1/p1", "pdf_url": ""}
        mock_col.query.return_value = {
            "ids": [["id1"]],
            "metadatas": [[shared_meta]],
            "documents": [["Abstract"]],
            "distances": [[0.1]],  # relevance = 0.9 — above threshold
        }

        result = store.get_relevant_papers(["q1", "q2"])  # same paper returned for both

        assert len(result) == 1

    def test_get_relevant_papers_filters_low_scores(self):
        store, mock_col, mock_genai, _ = self._make_store()
        mock_col.count.return_value = 1
        mock_genai.models.embed_content.return_value = _make_embedding_response([[0.1]])
        mock_col.query.return_value = {
            "ids": [["id1"]],
            "metadatas": [[{"title": "Low", "authors": "", "source": "arxiv", "year": "", "doi": "10/low", "pdf_url": ""}]],
            "documents": [["abs"]],
            "distances": [[0.8]],  # relevance = 0.2 — below MIN_RELEVANCE_SCORE=0.5
        }

        result = store.get_relevant_papers(["query"])

        assert len(result) == 0

    def test_get_relevant_papers_sorted_by_relevance(self):
        store, mock_col, mock_genai, _ = self._make_store()
        mock_col.count.return_value = 2
        mock_genai.models.embed_content.return_value = _make_embedding_response([[0.1]])

        def query_side_effect(**kwargs):
            return {
                "ids": [["a", "b"]],
                "metadatas": [[
                    {"title": "Low", "authors": "", "source": "arxiv", "year": "", "doi": "10/low", "pdf_url": ""},
                    {"title": "High", "authors": "", "source": "arxiv", "year": "", "doi": "10/high", "pdf_url": ""},
                ]],
                "documents": [["abs1", "abs2"]],
                "distances": [[0.4, 0.1]],  # relevance = 0.6 and 0.9 — both above threshold
            }

        mock_col.query.side_effect = query_side_effect

        result = store.get_relevant_papers(["query"])

        assert result[0]["relevance_score"] > result[1]["relevance_score"]


# ---------------------------------------------------------------------------
# planner_node integration (mocked LLM)
# ---------------------------------------------------------------------------

class TestPlannerNode:
    @patch("src.agents.graph.get_llm")
    def test_parses_json_sub_queries(self, mock_get_llm):
        from src.agents.graph import planner_node
        from src.agents.graph import get_initial_state

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(
            content='["sub-query 1", "sub-query 2", "sub-query 3"]'
        )

        state = get_initial_state("What are chromatic polynomials?")
        result = planner_node(state)

        assert result["current_phase"] == "planning"
        assert len(result["sub_queries"]) == 3
        assert result["sub_queries"][0] == "sub-query 1"

    @patch("src.agents.graph.get_llm")
    def test_falls_back_on_bad_json(self, mock_get_llm):
        from src.agents.graph import planner_node
        from src.agents.graph import get_initial_state

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="not valid json at all")

        state = get_initial_state("Research question fallback test")
        result = planner_node(state)

        assert result["sub_queries"] == ["Research question fallback test"]
        assert len(result["errors"]) == 1

    @patch("src.agents.graph.get_llm")
    def test_strips_markdown_fences(self, mock_get_llm):
        from src.agents.graph import planner_node
        from src.agents.graph import get_initial_state

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n["q1", "q2"]\n```'
        )

        state = get_initial_state("Test question")
        result = planner_node(state)

        assert len(result["sub_queries"]) == 2


# ---------------------------------------------------------------------------
# retriever_node integration (mocked fetcher + vector store)
# ---------------------------------------------------------------------------

class TestRetrieverNode:
    @patch("src.agents.graph.VectorStore")
    @patch("src.agents.graph.fetch_papers_for_queries")
    def test_happy_path(self, mock_fetch, mock_vs_cls):
        from src.agents.graph import retriever_node, get_initial_state

        papers = [make_paper(doi=f"10.1/{i}") for i in range(3)]
        mock_fetch.return_value = papers

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.collection.count.return_value = 3  # non-zero after embedding
        mock_vs.get_relevant_papers.return_value = papers

        state = get_initial_state("Test")
        state["sub_queries"] = ["q1", "q2"]
        result = retriever_node(state)

        assert result["current_phase"] == "retrieval"
        assert result["embeddings_ready"] is True
        assert len(result["retrieved_papers"]) == 3
        mock_vs.embed_and_store.assert_called_once_with(papers)

    @patch("src.agents.graph.VectorStore")
    @patch("src.agents.graph.fetch_papers_for_queries")
    def test_embedding_failure_sets_embeddings_ready_false(self, mock_fetch, mock_vs_cls):
        from src.agents.graph import retriever_node, get_initial_state

        mock_fetch.return_value = [make_paper()]
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.collection.count.return_value = 0  # all batches failed

        state = get_initial_state("Test")
        state["sub_queries"] = ["q1"]
        result = retriever_node(state)

        assert result["embeddings_ready"] is False
        assert len(result["errors"]) > 0

    @patch("src.agents.graph.VectorStore")
    @patch("src.agents.graph.fetch_papers_for_queries")
    def test_no_sub_queries_adds_error(self, mock_fetch, mock_vs_cls):
        from src.agents.graph import retriever_node, get_initial_state

        state = get_initial_state("Test")
        state["sub_queries"] = []
        result = retriever_node(state)

        assert len(result["errors"]) > 0
        mock_fetch.assert_not_called()

    @patch("src.agents.graph.VectorStore")
    @patch("src.agents.graph.fetch_papers_for_queries")
    def test_empty_fetch_result_adds_error(self, mock_fetch, mock_vs_cls):
        from src.agents.graph import retriever_node, get_initial_state

        mock_fetch.return_value = []
        state = get_initial_state("Test")
        state["sub_queries"] = ["q1"]
        result = retriever_node(state)

        assert result["embeddings_ready"] is False
        assert len(result["errors"]) > 0
