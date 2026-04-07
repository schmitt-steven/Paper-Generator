from typing import List, Optional, Callable
from dateutil import parser
import textwrap
import json
import time
import re
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from dataclasses import asdict
from difflib import SequenceMatcher

from phases.literature_search.paper import Paper, RankingScores
from phases.literature_search.semantic_scholar_api import SemanticScholarAPI
from phases.literature_search.paper_ranking import PaperRanker
from utils.llm_utils import remove_thinking_blocks
from phases.literature_search.paper_filter import PaperFilter
from phases.literature_search.citation_gap_finder import CitationGapFinder
from utils.pdf_downloader import PDFDownloader
from utils.open_access_finder import find_open_access_pdfs
from utils.lazy_model_loader import LazyModelMixin
from utils.file_utils import save_json, load_json
from phases.context_analysis.research_context_generator import ResearchContext
from settings import Settings


class SearchQuery(BaseModel):
    query: str
    year: Optional[str] = None  # Optional year filter (e.g. "2020-2024" or "2020")


class SearchQueriesResult(BaseModel):
    """Structured response format for multiple search queries"""
    queries: list[SearchQuery]


class LiteratureSearch(LazyModelMixin):
    
    def __init__(self, model_name: str):
        """Initialize LiteratureSearch with a language model."""
        self.model_name = model_name
        self._model = None
        self.s2_api = SemanticScholarAPI(api_key=Settings.SEMANTIC_SCHOLAR_API_KEY or None)


    def build_search_queries(self, research_context: ResearchContext) -> list[SearchQuery]:
        """Generate multiple search queries from research context for comprehensive literature search."""

        prompt = textwrap.dedent(f"""\
            Generate 15 Semantic Scholar search queries for a comprehensive academic literature review.

            STRICT QUERY RULES:
            - Use ONLY established academic terminology from the research domain
            - Use "quoted phrases" for multi-word algorithm names or concepts
            - Keep queries concise: 2-4 words
            - NO boolean operators (+, |, -)
            - AVOID generic terms that appear in many fields (e.g., "deterministic", "optimal", "efficient", "one-pass")
            - PREFER specific algorithm names, method names, and domain-specific vocabulary

            MANDATORY CATEGORY DISTRIBUTION:

            1. SURVEYS & REVIEWS (3 queries):
               - Focus on high-level overviews of the specific research topic
               - Include "survey" or "review" combined with the specific domain/method name
            
            2. FOUNDATIONAL (3 queries):
               - Target the core theories and seminal algorithms of this topic
               - Use exact names of the foundational math or models
            
            3. CORE METHODS (3 queries):
               - Specific algorithm names or technical approaches central to this topic
               - Focus on the most distinct terminology for the method
            
            4. RELATED WORK (3 queries):
               - Alternative approaches to the same problem
               - Competing algorithms or methods often cited in comparison
            
            5. BENCHMARKS (3 queries):
               - Standard evaluation environments, datasets, or tasks used in this field
               - Specific names of test suites or data repositories

            RESEARCH TOPIC:
            {research_context.description}

            AVOID THESE GENERIC TERMS (they match too many unrelated papers):
            - deterministic, stochastic, optimal, efficient, robust
            - one-pass, single-pass, forward, backward (unless part of specific algorithm name)
            - analysis, optimization, learning (alone)

            Output format:
            {{"queries": [{{"query": "specific algorithm name", "year": null}}, {{"query": "method survey", "year": null}}]}}

            Generate exactly 15 queries now:"""
        )

        print("Generating search queries...")
        
        # Retry up to 3 times if we get empty results
        max_attempts = 3
        search_queries: list[SearchQuery] = []
        
        for attempt in range(max_attempts):
            response = self.model.respond(
                prompt,
                response_format=SearchQueriesResult,
                config={
                    'temperature': 0.0,
                }
            )
            content = remove_thinking_blocks(response.content)
            result = json.loads(content)
            
            # LM Studio returns dicts for structured responses
            queries_list = result.get('queries', [])
            
            search_queries = []
            for q in queries_list:
                query_text = q.get('query', '').strip()
                if query_text:  # Only add non-empty queries
                    search_queries.append(SearchQuery(query=query_text, year=q.get('year')))
            
            if search_queries:
                break  # Got valid queries, exit retry loop
            
            if attempt < max_attempts - 1:
                print(f"  Got empty queries, retrying ({attempt + 2}/{max_attempts})...")
        
        print(f"Generated {len(search_queries)} search queries.")
        
        # Save queries to .json
        self.save_search_queries(search_queries, filename="search_queries.json", output_dir="output")
        
        return search_queries
    

    @staticmethod
    def save_search_queries(queries: list[SearchQuery], filename: Optional[str] = None, output_dir: str = "output"):
        """Save search queries to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_queries_{timestamp}.json"

        queries_data = [{"query": q.query, "year": q.year} for q in queries]
        filepath = save_json(queries_data, filename, output_dir)

        print(f"Saved {len(queries)} search queries to {filepath}")
        return filepath
    

    @staticmethod
    def load_search_queries(filepath: str) -> list[SearchQuery]:
        """Load search queries from JSON file."""
        path_obj = Path(filepath)
        data = load_json(path_obj.name, str(path_obj.parent))

        queries = []
        for q in data:
            # Handle backward compatibility: old format had label/description, new format has query/year
            if "query" in q:
                # New format
                queries.append(SearchQuery(query=q["query"], year=q.get("year")))
            elif "label" in q and "query" in q:
                # Old format - convert to new format
                queries.append(SearchQuery(query=q["query"], year=q.get("year")))
        
        print(f"Loaded {len(queries)} queries from {filepath}")
        return queries


    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison (lowercase, remove special chars, extra spaces)"""
        if not title:
            return ""
        # Lowercase
        normalized = title.lower()
        # Remove special characters except spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def _get_first_author(self, authors: list[str]) -> str:
        """Extract first author name for comparison"""
        if not authors:
            return ""
        first_author = authors[0].strip()
        # Extract last name (handle "Last, First" or "First Last" formats)
        if ',' in first_author:
            return first_author.split(',')[0].strip().lower()
        else:
            parts = first_author.split()
            return parts[-1].lower() if parts else first_author.lower()
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (0-1)"""
        norm1 = self._normalize_title(title1)
        norm2 = self._normalize_title(title2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _is_duplicate(self, paper1: Paper, paper2: Paper) -> bool:
        """
        Check if two papers are duplicates.
        
        Primary: DOI exact match (if both have DOI)
        Secondary: Title + first author similarity (fuzzy match)
        
        Args:
            paper1: First paper
            paper2: Second paper
            
        Returns:
            True if papers are duplicates
        """
        # Primary: DOI exact match
        if paper1.doi and paper2.doi:
            if paper1.doi.lower() == paper2.doi.lower():
                return True
        
        # Secondary: Title + first author similarity
        title_sim = self._title_similarity(paper1.title, paper2.title)
        if title_sim >= 0.9:  # High title similarity threshold
            # Check first author match
            author1 = self._get_first_author(paper1.authors)
            author2 = self._get_first_author(paper2.authors)
            if author1 and author2:
                # Check if author names are similar (fuzzy match)
                author_sim = SequenceMatcher(None, author1, author2).ratio()
                if author_sim >= 0.9:  # High author similarity threshold
                    return True
        
        return False
    
    def remove_duplicates(self, papers: list[Paper]) -> list[Paper]:
        """
        Remove duplicate papers from the list based on paper ID.
        
        Args:
            papers: List of Paper objects that may contain duplicates
            
        Returns:
            List of unique Paper objects (duplicates removed)
        """
        seen_ids = set()
        unique_papers = []
        duplicate_count = 0
        
        for paper in papers:
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                unique_papers.append(paper)
            else:
                duplicate_count += 1
        
        return unique_papers
    
    def detect_and_merge_duplicates(
        self, 
        user_papers: list[Paper], 
        searched_papers: list[Paper]
    ) -> list[Paper]:
        """
        Detect duplicates between user-provided and searched papers, merge them.
        Prefers user-provided papers over searched papers when duplicates are found.
        
        Args:
            user_papers: List of user-provided Paper objects
            searched_papers: List of automatically searched Paper objects
            
        Returns:
            Merged list of unique Paper objects (user papers take priority)
        """
        merged = list(user_papers)  # Start with all user papers
        seen_user_ids = {p.id for p in user_papers}
        duplicate_count = 0
        
        for searched_paper in searched_papers:
            # Skip if already in user papers (by ID)
            if searched_paper.id in seen_user_ids:
                duplicate_count += 1
                print(f"  Duplicate detected (by ID): {searched_paper.title[:60]}... (keeping user version)")
                continue
            
            # Check for duplicates by DOI or title+author
            is_duplicate = False
            for user_paper in user_papers:
                if self._is_duplicate(user_paper, searched_paper):
                    is_duplicate = True
                    duplicate_count += 1
                    print(f"  Duplicate detected: '{searched_paper.title[:60]}...' (keeping user version)")
                    break
            
            if not is_duplicate:
                merged.append(searched_paper)
        
        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate(s) from searched papers (kept user versions)\n")
        
        return merged


    def execute_search(self, query: str, max_results: int = 30, year: Optional[str] = None, fields_of_study: Optional[str] = None, open_access_only: bool = False) -> list[Paper]:
        """
        Execute a single search on Semantic Scholar using the provided query string.
        
        Args:
            query: Search query string
            max_results: Maximum number of results per query
            year: Optional year filter (e.g., "2020-2024" or "2020")
            fields_of_study: Optional comma-separated fields of study filter (e.g., "Computer Science,Mathematics")
            
        Returns:
            List of Paper objects
        """
        year_str = f" (year: {year})" if year else ""
        fields_str = f" (fields: {fields_of_study})" if fields_of_study else ""
        print(f"Searching Semantic Scholar with: {query}{year_str}{fields_str} (max_results={max_results})")
        papers = self.s2_api.search_papers(query, max_results=max_results, year=year, fields_of_study=fields_of_study, open_access_only=open_access_only)
        print(f"Found {len(papers)} papers\n")
        return papers
    

    def search_papers(self, queries: list[SearchQuery], max_results_per_query: int = 30) -> list[Paper]:
        """
        Execute multiple searches on Semantic Scholar using a list of SearchQuery objects.
        Includes delay between queries to respect rate limits.
        Automatically removes duplicate papers.
        Uses default fields of study filter for auto-searched papers: Computer Science, Mathematics, Engineering
        
        Args:
            queries: List of SearchQuery objects
            max_results_per_query: Maximum number of results per individual query
            
        Returns:
            List of unique Paper objects from all queries combined (duplicates removed)
        """
        # Default fields of study for auto-searched papers
        DEFAULT_FIELDS_OF_STUDY = "Computer Science"
        
        all_papers = []
        for i, query_obj in enumerate(queries, 1):
            query_str = query_obj.query[:60] + "..." if len(query_obj.query) > 60 else query_obj.query
            year_str = f" (year: {query_obj.year})" if query_obj.year else ""
            print(f"Executing query {i}/{len(queries)}: {query_str}{year_str}")
            papers = self.execute_search(
                query_obj.query, 
                max_results=max_results_per_query, 
                year=query_obj.year,
                fields_of_study=DEFAULT_FIELDS_OF_STUDY
            )
            all_papers.extend(papers)
            
            # Add delay between queries to respect rate limit
            if i < len(queries):
                time.sleep(2.0)
        
        # Remove duplicates
        unique_papers = self.remove_duplicates(all_papers)
        print(f"Papers found: {len(all_papers)}, unique papers: {len(unique_papers)}")
        
        return unique_papers


    def download_papers_as_pdfs(
        self, 
        papers: list[Paper], 
        base_folder: str = "output/literature/"
    ):
        """
        Download selected papers as PDFs to specified folder.
        
        Args:
            papers: List of Paper objects to download
            base_folder: Base folder for all papers
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        return PDFDownloader.download_papers_as_pdfs(papers, base_folder)
    

    @staticmethod
    def save_papers(papers: list[Paper], filename: Optional[str] = None, output_dir: str = "output"):
        """Save papers to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"papers_{timestamp}.json"

        papers_data = [
            {
                "id": paper.id,
                "title": paper.title,
                "published": paper.published.isoformat() if isinstance(paper.published, datetime) else paper.published,
                "authors": paper.authors,
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "fields_of_study": paper.fields_of_study,
                "venue": paper.venue,
                "citation_count": paper.citation_count,
                "bibtex": paper.bibtex,
                "markdown_text": paper.markdown_text,
                "ranking": asdict(paper.ranking) if paper.ranking else None,
                "citation_key": paper.citation_key,
                "is_open_access": paper.is_open_access,
                "user_provided": paper.user_provided,
                "pdf_path": paper.pdf_path,
                "conclusion": paper.conclusion
            }
            for paper in papers
        ]

        filepath = save_json(papers_data, filename, output_dir)

        print(f"Saved {len(papers)} papers to {filepath}")
        return filepath

    @staticmethod
    def load_papers(filepath: str) -> list[Paper]:
        """Load papers from JSON file."""
        path_obj = Path(filepath)
        data = load_json(path_obj.name, str(path_obj.parent))

        papers = []
        for p in data:
            # Convert ranking dict back to RankingScores object
            if p.get('ranking'):
                p['ranking'] = RankingScores(**p['ranking'])

            # Extract citation_key if present (it won't be passed to constructor due to init=False)
            citation_key = p.pop('citation_key', None)

            # Parse published date if string
            if isinstance(p.get('published'), str):
                try:
                    p['published'] = parser.parse(p['published'])
                except (ValueError, TypeError):
                    pass
            
            # Create Paper object (citation_key will be auto-generated in __post_init__ if None)
            paper = Paper(**p)

            # If citation_key was in JSON, use it (otherwise keep auto-generated one)
            if citation_key:
                paper.citation_key = citation_key

            papers.append(paper)

        print(f"Loaded {len(papers)} papers from {filepath}")
        return papers


    def run_automated_search(
        self,
        research_context: ResearchContext,
        user_papers: List[Paper],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Paper]:
        """
        Execute the full automated literature search pipeline.
        
        Steps:
        1. Generate search queries from research context
        2. Execute search on Semantic Scholar
        3. Rank papers by relevance, citations, and recency
        4. Filter papers using LLM verification
        5. Analyze citation gaps and find missing foundational papers
        6. Check for open access PDFs
        
        Args:
            research_context: Domain context for search and ranking
            user_papers: Existing papers to avoid duplicates
            progress_callback: Optional callback for status updates
            
        Returns:
            List of filtered and ranked Paper objects
        """
        def update_status(msg):
            if progress_callback:
                progress_callback(msg)
                
        # Step 1: Search
        update_status("Building search queries")
        search_queries = self.build_search_queries(research_context)
        
        update_status(f"Searching related papers with {len(search_queries)} queries")
        papers = self.search_papers(search_queries, max_results_per_query=20)
        
        # Filter out papers already in user papers to avoid processing duplicates
        user_paper_ids = {p.id for p in user_papers}
        searched_papers = [p for p in papers if p.id not in user_paper_ids]
        
        if not searched_papers:
            return []
        
        # Step 2: Rank papers
        update_status("Ranking papers for relevance")
        ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
        ranking_context = research_context.description
        ranked_papers = ranker.rank_papers(
            papers=searched_papers,
            context=ranking_context,
            weights={'relevance': 0.8, 'citations': 0.1, 'recency': 0.1}
        )
        
        # Step 3: Filter papers
        update_status("Filtering found papers")
        # Enhance context with open questions for better filtering
        enhanced_context = f"{research_context.description}\n\nOpen Research Questions:\n{research_context.open_questions}"
        filtered_papers = PaperFilter.filter_papers(
            papers=ranked_papers,
            research_context=enhanced_context,
            model_name=self.model_name,
            target_count=40,
            min_relevance=0.5
        )
        
        # Step 4: Citation Gap Analysis
        update_status("Analyzing for missing foundational papers")
        gap_finder = CitationGapFinder()
        suggestions = gap_finder.identify_missing_papers(
            papers=filtered_papers,
            research_context=enhanced_context,
            model_name=self.model_name
        )
        
        if suggestions:
            update_status(f"Searching for {len(suggestions)} suggested foundational papers")
            existing_ids = {p.id for p in filtered_papers} | {p.id for p in user_papers}
            foundational_papers = gap_finder.search_suggested_papers(suggestions, existing_ids)
            
            if foundational_papers:
                update_status("Ranking foundational papers")
                foundational_papers = ranker.rank_papers(
                    papers=foundational_papers,
                    context=ranking_context,
                )
                filtered_papers.extend(foundational_papers)
                print(f"Added {len(foundational_papers)} foundational papers to the collection")
                
                # Re-sort collection by relevance score
                filtered_papers.sort(
                    key=lambda p: p.ranking.relevance_score if p.ranking else 0,
                    reverse=True
                )
        
        # Step 5: Check for open access PDFs
        papers_without_urls = [p for p in filtered_papers if not p.pdf_url]
        if papers_without_urls:
            update_status(f"Finding open access PDFs for {len(papers_without_urls)} papers")
            find_open_access_pdfs(papers_without_urls)  # Updates papers in-place
            
        return filtered_papers
