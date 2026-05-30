"""
Paper retrieval from ArXiv and Semantic Scholar.
"""

import time
import logging
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

import arxiv
from semanticscholar import SemanticScholar

from src.utils.state import Paper
from src.utils.config import Config

logger = logging.getLogger(__name__)

# Shared client so delay_seconds is enforced across all sub-query calls, not just within one.
_arxiv_client = arxiv.Client(page_size=10, delay_seconds=5.0, num_retries=3)


def _paper_unique_key(doi: Optional[str], title: str) -> str:
    """Generate a deduplication key: prefer DOI, fall back to normalized title."""
    if doi:
        return doi.lower()
    return f"title:{title.lower().strip()[:80]}"


def _author_name(author) -> Optional[str]:
    """Extract name from an Author object or dict (semanticscholar returns either)."""
    if isinstance(author, dict):
        return author.get("name")
    return getattr(author, "name", None)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_arxiv_papers(query: str, max_results: int = 20) -> List[Paper]:
    """Fetch papers from ArXiv filtered to graph theory categories."""
    category_filter = " OR ".join(f"cat:{c}" for c in Config.ARXIV_CATEGORY_FILTERS)
    full_query = f"{query} AND ({category_filter})"

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[Paper] = []
    for result in _arxiv_client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "doi": result.doi or None,
            "pdf_url": result.pdf_url or None,
            "source": "arxiv",
            "year": result.published.year if result.published else None,
            "relevance_score": None,
        })

    logger.info("ArXiv returned %d papers for query: %r", len(papers), query)
    return papers


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_semantic_scholar_papers(query: str, max_results: int = 20) -> List[Paper]:
    """Fetch papers from Semantic Scholar in the Mathematics field."""
    sch = SemanticScholar()

    results = sch.search_paper(
        query,
        fields=["title", "abstract", "authors", "year", "externalIds", "openAccessPdf"],
        limit=max_results,
    )

    papers: List[Paper] = []
    for item in results:
        if not item.abstract:
            continue

        doi = (item.externalIds or {}).get("DOI") or None
        pdf_url = None
        if item.openAccessPdf and isinstance(item.openAccessPdf, dict):
            pdf_url = item.openAccessPdf.get("url") or None

        authors = [n for a in (item.authors or []) if (n := _author_name(a))]

        papers.append({
            "title": item.title or "",
            "abstract": item.abstract,
            "authors": authors,
            "doi": doi,
            "pdf_url": pdf_url,
            "source": "semantic_scholar",
            "year": item.year,
            "relevance_score": None,
        })

    logger.info("Semantic Scholar returned %d papers for query: %r", len(papers), query)
    return papers


def fetch_papers_for_queries(
    sub_queries: List[str],
    papers_per_query: int = 5,
) -> List[Paper]:
    """
    Fetch and deduplicate papers from ArXiv and Semantic Scholar for all sub-queries.
    Deduplicates by DOI (preferred) or normalized title.
    """
    seen_keys: set = set()
    all_papers: List[Paper] = []

    for query in sub_queries:
        logger.info("Fetching papers for sub-query: %r", query)

        for fetcher, source_name in [
            (fetch_arxiv_papers, "ArXiv"),
            # (fetch_semantic_scholar_papers, "Semantic Scholar"),  # disabled: ConnectionRefusedError
        ]:
            try:
                batch = fetcher(query, max_results=papers_per_query)
                for paper in batch:
                    key = _paper_unique_key(paper.get("doi"), paper["title"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_papers.append(paper)
            except Exception as exc:
                logger.warning("%s fetch failed for query %r: %s", source_name, query, exc)

        # Avoid hammering the APIs between queries
        time.sleep(5)

    logger.info("Total unique papers fetched: %d", len(all_papers))
    return all_papers
