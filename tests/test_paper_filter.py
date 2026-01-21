"""Test script to compare paper filtering methods."""
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper_filter import PaperFilter
from phases.paper_search.paper_ranking import PaperRanker
from phases.context_analysis.paper_conception import PaperConception
from settings import Settings

# ============================================================
# CONFIGURATION
# ============================================================
RUN_PAPER_SEARCH = False  # Set to True to run a new paper search
PAPER_COUNT = 40          # Number of papers to show for each method


def search_papers():
    """Run a new paper search based on the paper concept."""
    print("=" * 60)
    print("PAPER SEARCH")
    print("=" * 60)
    
    paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
    print(f"Paper Concept: {paper_concept.description}")
    
    lit_search = LiteratureSearch(model_name=Settings.LITERATURE_SEARCH_MODEL)
    
    print("\n--- Generating Search Queries ---")
    queries = lit_search.build_search_queries(paper_concept)
    
    print("\n--- Executing Search ---")
    papers = lit_search.search_papers(queries, max_results_per_query=30)
    print(f"\nFound {len(papers)} unique papers")
    
    print("\n--- Ranking Papers ---")
    ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
    ranking_context = paper_concept.description
    papers = ranker.rank_papers(
        papers=papers,
        context=ranking_context,
        weights={'relevance': 0.75, 'citations': 0.15, 'recency': 0.1}
    )
    
    LiteratureSearch.save_papers(papers, filename="papers.json", output_dir="output")
    print(f"Saved {len(papers)} papers to output/papers.json")
    
    return papers


def load_papers():
    """Load existing papers from papers.json."""
    print("Loading papers.json...")
    papers = LiteratureSearch.load_papers("output/papers.json")
    print(f"Loaded {len(papers)} papers")
    return papers


def ensure_embeddings(papers):
    """Ensure papers have embeddings."""
    has_embeddings = all(getattr(p, 'title_abstract_embedding', None) is not None for p in papers)
    
    if not has_embeddings:
        print("Re-computing embeddings...")
        paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
        ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
        papers = ranker.rank_papers(
            papers=papers,
            context=paper_concept.description,
            weights={'relevance': 0.75, 'citations': 0.15, 'recency': 0.1}
        )
        LiteratureSearch.save_papers(papers, filename="papers.json", output_dir="output")
    
    return papers


def print_papers(papers, title):
    """Print a list of papers with open access stats."""
    open_count = sum(1 for p in papers if p.is_open_access)
    closed_count = len(papers) - open_count
    
    print(f"\n{title}")
    print(f"Open Access: {open_count} | Closed Access: {closed_count}")
    print("-" * 60)
    for i, p in enumerate(papers, 1):
        score = p.ranking.final_score if p.ranking else 0
        year = p.published.year if hasattr(p.published, 'year') else str(p.published)[:4] if p.published else 'N/A'
        access = "OA" if p.is_open_access else "CA"
        print(f"{i:2d}. [{score:.3f}] [{access}] ({year}) {p.title[:45]}...")


def main():
    # Load or search papers
    if RUN_PAPER_SEARCH:
        papers = search_papers()
    else:
        papers = load_papers()
    
    if not papers:
        print("No papers found.")
        return
    
    papers = ensure_embeddings(papers)
    
    # Get research context for LLM verification
    paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
    research_context = f"{paper_concept.description}\n\nOpen Research Questions:\n{paper_concept.open_questions}"
    
    # Filter
    print("\n" + "=" * 60)
    print("FILTER (Composite Score + LLM)")
    print("=" * 60)
    simple_papers = PaperFilter.filter_papers(
        papers=papers,
        research_context=research_context,
        model_name=Settings.LITERATURE_SEARCH_MODEL,
        target_count=PAPER_COUNT,
        min_relevance=0.5
    )
    print_papers(simple_papers, f"Filter Results ({len(simple_papers)} papers)")
    

if __name__ == "__main__":
    main()
