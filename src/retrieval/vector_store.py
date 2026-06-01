"""
ChromaDB vector store for paper abstract embeddings and similarity search.
Uses google.genai directly for embeddings (langchain-google-genai uses v1beta
which doesn't serve all embedding models; google.genai 2.x uses the correct endpoint).
"""

import time
import logging
from typing import List, Dict, Any, Optional

import chromadb
import google.genai as genai

from src.utils.state import Paper
from src.utils.config import Config

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "papers"


def _chroma_id(paper: Paper) -> str:
    """Build a valid ChromaDB document ID (no slashes) from paper metadata."""
    raw = paper.get("doi") or f"title:{paper['title'][:80]}"
    return raw.replace("/", "_").replace(":", "_").replace(" ", "_")


def _metadata_from_paper(paper: Paper) -> Dict[str, Any]:
    return {
        "title": paper["title"],
        "authors": ", ".join(paper["authors"]),
        "source": paper["source"],
        "year": str(paper.get("year") or ""),
        "doi": paper.get("doi") or "",
        "pdf_url": paper.get("pdf_url") or "",
    }


def _paper_from_chroma_row(
    meta: Dict[str, Any], document: str, distance: float
) -> Paper:
    authors_str = meta.get("authors", "")
    authors = [a.strip() for a in authors_str.split(",")] if authors_str else []
    year_str = meta.get("year", "")
    year = int(year_str) if year_str and year_str.isdigit() else None
    # FIX-5: cosine distance ∈ [0, 2] → clamp to [0, 1] so score is never negative.
    return {
        "title": meta.get("title", ""),
        "abstract": document,
        "authors": authors,
        "doi": meta.get("doi") or None,
        "pdf_url": meta.get("pdf_url") or None,
        "source": meta.get("source", "unknown"),
        "year": year,
        "relevance_score": round(max(0.0, 1.0 - distance), 4),
    }


class VectorStore:
    """ChromaDB-backed store for paper abstracts with Google GenAI embeddings."""

    def __init__(
        self,
        persist_dir: str = Config.CHROMA_DB_PATH,
        collection_name: str = _DEFAULT_COLLECTION,
        clear: bool = False,
        chroma_client=None,
        genai_client=None,
    ):
        # REF-4: accept injected clients so tests can construct without real I/O.
        self.chroma = (
            chroma_client
            if chroma_client is not None
            else chromadb.PersistentClient(path=persist_dir)
        )
        if clear:
            try:
                self.chroma.delete_collection(collection_name)
                logger.info("Cleared existing ChromaDB collection '%s'", collection_name)
            except Exception:
                pass
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._genai = (
            genai_client
            if genai_client is not None
            else genai.Client(api_key=Config.GOOGLE_API_KEY)
        )
        self.embedding_model = Config.EMBEDDING_MODEL

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts; returns a list of float vectors."""
        response = self._genai.models.embed_content(
            model=self.embedding_model,
            contents=texts,
        )
        return [list(e.values) for e in response.embeddings]

    def _embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        response = self._genai.models.embed_content(
            model=self.embedding_model,
            contents=query,
        )
        return list(response.embeddings[0].values)

    def embed_and_store(self, papers: List[Paper]) -> None:
        """Embed paper abstracts in batches and upsert into ChromaDB."""
        valid = [p for p in papers if p.get("abstract")]
        if not valid:
            logger.warning("No papers with abstracts to embed")
            return

        total_batches = (len(valid) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        for batch_idx in range(0, len(valid), Config.BATCH_SIZE):
            batch = valid[batch_idx : batch_idx + Config.BATCH_SIZE]
            texts = [p["abstract"] for p in batch]
            ids = [_chroma_id(p) for p in batch]
            metadatas = [_metadata_from_paper(p) for p in batch]

            try:
                vectors = self._embed_texts(texts)
                self.collection.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )
                current_batch = batch_idx // Config.BATCH_SIZE + 1
                logger.info(
                    "Embedded batch %d/%d (%d papers)",
                    current_batch,
                    total_batches,
                    len(batch),
                )
            except Exception as exc:
                logger.error(
                    "Embedding batch %d failed: %s",
                    batch_idx // Config.BATCH_SIZE + 1,
                    exc,
                )

            if batch_idx + Config.BATCH_SIZE < len(valid):
                time.sleep(Config.BATCH_DELAY_SECONDS)

    def delete_papers(self, papers: List[Paper]) -> None:
        """Delete papers from the collection by their computed IDs."""
        ids = [_chroma_id(p) for p in papers]
        try:
            self.collection.delete(ids=ids)
            logger.info("Deleted %d papers from ChromaDB", len(ids))
        except Exception as exc:
            logger.error("Failed to delete papers from ChromaDB: %s", exc)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Paper]:
        """Return the top-k most relevant papers for a query string.

        FIX-6: k defaults to None and resolves from Config at call time, so
        runtime changes to Config.TOP_K_PAPERS are respected.
        """
        effective_k = k if k is not None else Config.TOP_K_PAPERS
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(effective_k, count)
        query_vector = self._embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        papers: List[Paper] = []
        metas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for meta, doc, dist in zip(metas, docs, dists):
            papers.append(_paper_from_chroma_row(meta, doc, dist))

        return papers

    def get_relevant_papers(
        self, sub_queries: List[str], k_per_query: Optional[int] = None
    ) -> List[Paper]:
        """
        Retrieve and deduplicate top-k papers across all sub-queries.
        Returns papers sorted by relevance score descending.
        """
        seen_keys: set = set()
        all_papers: List[Paper] = []

        for query in sub_queries:
            for paper in self.similarity_search(query, k=k_per_query):
                key = paper.get("doi") or paper["title"].lower().strip()
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_papers.append(paper)

        all_papers.sort(key=lambda p: p.get("relevance_score") or 0.0, reverse=True)
        filtered = [
            p for p in all_papers
            if (p.get("relevance_score") or 0.0) >= Config.MIN_RELEVANCE_SCORE
        ]
        logger.info(
            "Relevance filter (>= %.2f): %d/%d papers kept",
            Config.MIN_RELEVANCE_SCORE,
            len(filtered),
            len(all_papers),
        )
        return filtered
