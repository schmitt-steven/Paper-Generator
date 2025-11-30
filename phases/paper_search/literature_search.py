from typing import List, Optional
import textwrap
import json
import time
import re
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from dataclasses import asdict
from difflib import SequenceMatcher

from phases.paper_search.paper import Paper, RankingScores
from phases.paper_search.semantic_scholar_api import SemanticScholarAPI
from utils.pdf_downloader import PDFDownloader
from utils.lazy_model_loader import LazyModelMixin
from utils.file_utils import save_json, load_json
from phases.context_analysis.paper_conception import PaperConcept


class SearchQuery(BaseModel):
    label: str
    query: str
    description: str


class SearchQueriesResult(BaseModel):
    """Structured response format for multiple search queries"""
    queries: List[SearchQuery]


class LiteratureSearch(LazyModelMixin):
    
    def __init__(self, model_name: str):
        """
        Initialize LiteratureSearch with a language model.
        
        Args:
            model_name: Name of the LLM model to use for query generation
        """
        self.model_name = model_name
        self._model = None
        self.s2_api = SemanticScholarAPI()


    def build_search_queries(self, paper_concept: PaperConcept) -> List[SearchQuery]:
        """
        Generate multiple search queries from paper concept for comprehensive literature search.
        
        Args:
            paper_concept: PaperConcept object containing description, open questions, and code snippets
            
        Returns:
            List of SearchQuery objects with label, query, and description
        """

        system_prompt = textwrap.dedent("""\
            [TASK]
            Generate Semantic Scholar search queries for a comprehensive literature search.

            [SEMANTIC SCHOLAR QUERY SYNTAX]
            - Plain text queries work best.
            - Use + for AND (e.g., "reinforcement learning" + "sparse reward")
            - Use | for OR (e.g., "DQN" | "Q-learning")
            - Use - for NOT (e.g., "transformers" - "vision")
            - Use "exact phrase" for multi-word terms.
            - Filters: year:2020-2024, venue:NeurIPS, fieldsOfStudy:Computer Science

            [CRITICAL CONSTRAINTS]
            - Use standard terminology.
            - Avoid overly complex boolean logic.
            - Labels must be plain text, no URL encoding.
            - Adhere to the SEMANTIC SCHOLAR QUERY SYNTAX.

            [GOOD QUERY EXAMPLES]
            "experience replay" + "Q-learning"
            "reinforcement learning" + "sparse reward"
            "transformer" + "attention" year:2015-2025

            [REQUIREMENTS]
            Generate 15 queries covering:
            - Established methods (Dyna-Q, prioritized sweeping, eligibility traces, etc.)
            - Core concepts (sample efficiency, sparse rewards, value propagation)
            - Use standard terminology
            - Alternative approaches (episodic memory, model-based planning)
            - Surveys/foundational papers

            Cast a wide net. Better to get 100 results and filter than 0 from a "perfect" query.

            [OUTPUT FORMAT]
            Return JSON array with objects containing: label, query, description
        """)

        # Combine all information into a single prompt
        code_snippets_section = paper_concept.code_snippets if paper_concept.code_snippets else "No code snippets available."
        
        user_message = textwrap.dedent(f"""\
            [PAPER CONCEPT DESCRIPTION]
            {paper_concept.description}

            [OPEN QUESTIONS FOR LITERATURE SEARCH]
            {paper_concept.open_questions if paper_concept.open_questions else "No specific questions provided."}

            [CODE SNIPPETS]
            {code_snippets_section}

            Generate comprehensive search queries based on all the above information.
        """)

        print("Generating search queries...")
        full_prompt = system_prompt + "\n\n" + user_message
        
        result = self.model.respond(
            full_prompt,
            response_format=SearchQueriesResult,
            config={
                'temperature': 0.5,
            }
        ).parsed
        
        # Convert dict results to SearchQuery objects
        search_queries = [
            SearchQuery(**query_obj) for query_obj in result['queries']
        ]
        
        print(f"Generated {len(search_queries)} search queries.")
        
        # Automatically save
        self.save_search_queries(search_queries, filename="search_queries.json", output_dir="output")
        
        return search_queries
    

    @staticmethod
    def save_search_queries(queries: List[SearchQuery], filename: str = None, output_dir: str = "output"):
        """Save search queries to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_queries_{timestamp}.json"

        queries_data = [{"label": q.label, "query": q.query, "description": q.description} for q in queries]
        filepath = save_json(queries_data, filename, output_dir)

        print(f"Saved {len(queries)} search queries to {filepath}")
        return filepath
    

    @staticmethod
    def load_search_queries(filepath: str) -> List[SearchQuery]:
        """Load search queries from JSON file."""
        path_obj = Path(filepath)
        data = load_json(path_obj.name, str(path_obj.parent))

        queries = [SearchQuery(**q) for q in data]
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
    
    def _get_first_author(self, authors: List[str]) -> str:
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
    
    def remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
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
        user_papers: List[Paper], 
        searched_papers: List[Paper]
    ) -> List[Paper]:
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


    def execute_search(self, query: str, max_results: int = 20) -> List[Paper]:
        """
        Execute a single search on Semantic Scholar using the provided query string.
        
        Args:
            query: Search query string
            max_results: Maximum number of results per query (default: 50)
            
        Returns:
            List of Paper objects
        """
        print(f"Searching Semantic Scholar with: {query} (max_results={max_results})")
        papers = self.s2_api.search_papers(query, max_results=max_results)
        print(f"Found {len(papers)} papers\n")
        return papers
    

    def search_papers(self, queries: List[SearchQuery], max_results_per_query: int = 30) -> List[Paper]:
        """
        Execute multiple searches on Semantic Scholar using a list of SearchQuery objects.
        Includes delay between queries to respect rate limits.
        Automatically removes duplicate papers.
        
        Args:
            queries: List of SearchQuery objects
            max_results_per_query: Maximum number of results per individual query
            
        Returns:
            List of unique Paper objects from all queries combined (duplicates removed)
        """
        all_papers = []
        for i, query_obj in enumerate(queries, 1):
            print(f"Executing query {i}/{len(queries)} ({query_obj.label})")
            papers = self.execute_search(query_obj.query, max_results=max_results_per_query)
            all_papers.extend(papers)
            
            # Add delay between queries to respect rate limit
            if i < len(queries):
                time.sleep(1.0)
        
        # Remove duplicates
        unique_papers = self.remove_duplicates(all_papers)
        print(f"Papers found: {len(all_papers)}, unique papers: {len(unique_papers)}")
        
        # Automatically save
        self.save_papers(unique_papers, filename="papers.json", output_dir="output")
        
        return unique_papers


    def download_papers_as_pdfs(
        self, 
        papers: List[Paper], 
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
    def save_papers(papers: List[Paper], filename: str = None, output_dir: str = "output"):
        """Save papers to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"papers_{timestamp}.json"

        papers_data = [
            {
                "id": paper.id,
                "title": paper.title,
                "published": paper.published,
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
                "pdf_path": paper.pdf_path
            }
            for paper in papers
        ]

        filepath = save_json(papers_data, filename, output_dir)

        print(f"Saved {len(papers)} papers to {filepath}")
        return filepath
    

    @staticmethod
    def load_papers(filepath: str) -> List[Paper]:
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

            # Create Paper object (citation_key will be auto-generated in __post_init__ if None)
            paper = Paper(**p)

            # If citation_key was in JSON, use it (otherwise keep auto-generated one)
            if citation_key:
                paper.citation_key = citation_key

            papers.append(paper)

        print(f"Loaded {len(papers)} papers from {filepath}")
        return papers
