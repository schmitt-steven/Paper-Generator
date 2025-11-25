from phases.paper_search.arxiv_api import ArxivAPI, Paper, RankingScores
from phases.context_analysis.paper_conception import PaperConcept
from typing import List
from dataclasses import asdict
import textwrap
from pydantic import BaseModel
import lmstudio as lms
import time
import requests
import json
import os
from datetime import datetime
from pathlib import Path
from utils.lazy_model_loader import LazyModelMixin
from utils.file_utils import save_json, load_json


class SearchQuery(BaseModel):
    """Represents a single search query with metadata"""
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
        self._model = None  # Lazy-loaded via LazyModelMixin
        self.arxiv_api = ArxivAPI()


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
            Generate arXiv API search queries for a comprehensive literature search.

            [ARXIV QUERY SYNTAX]
            Field: abs: (abstract)
            Operators: +AND+, +OR+, +ANDNOT+ (uppercase only)
            Quotes: %22phrase%22 (spaces become +)
            Grouping: %28...%29 required when mixing +OR+ with +AND+/+ANDNOT+

            [CRITICAL CONSTRAINTS]
            - Use EXACTLY 2 AND conditions per query. No more, no less.
            - Use ONLY standard RL terminology that commonly appears in paper abstracts.
            - NEVER use invented/novel phrases - if it's not established jargon, don't use it.
            - Avoid ANDNOT entirely.
            - Labels must be plain text, no URL encoding.
            - Adhere to the ARXIV QUERY SYNTAX.


            [GOOD QUERY EXAMPLES]
            abs:%22experience%20replay%22+AND+abs:%22Q-learning%22
            abs:%22prioritized%20sweeping%22+AND+abs:%22reinforcement%20learning%22
            abs:%22eligibility%20traces%22+AND+abs:%22temporal%20difference%22
            abs:%22sparse%20reward%22+AND+abs:%22reinforcement%20learning%22

            [BAD QUERY EXAMPLES - DO NOT GENERATE THESE]
            abs:%22Q-learning%22+AND+abs:%22sample%20efficiency%22+AND+abs:%22sparse%20rewards%22
            (Too many ANDs - will return nothing)
            
            abs:%22backward%20induction%22+AND+abs:%22reward%20propagation%22
            (Phrases too specific - papers don't use these exact terms)

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


    def execute_search(self, query: str, max_results: int = 50) -> List[Paper]:
        """
        Execute a single search on arXiv using the provided query string.
        
        Args:
            query: Formatted search query string for arXiv API
            max_results: Maximum number of results per query (default: 50)
            
        Returns:
            List of Paper objects with metadata (title, authors, abstract, etc.)
        """
        print(f"Searching arXiv with: {query} (max_results={max_results})")
        response = self.arxiv_api.search_papers(query, max_results=max_results)
        papers = self.arxiv_api.parse_response(response)
        print(f"Found {len(papers)} papers\n")
        return papers
    

    def search_papers(self, queries: List[SearchQuery], max_results_per_query: int = 30) -> List[Paper]:
        """
        Execute multiple searches on arXiv using a list of SearchQuery objects.
        Includes 3-second delay between queries to respect arXiv rate limits.
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
            
            # Add delay between queries to respect arXiv rate limit (1 request per 3 seconds recommended)
            if i < len(queries):
                time.sleep(3)
        
        # Remove duplicates
        unique_papers = self.remove_duplicates(all_papers)
        print(f"Papers found: {len(all_papers)}, unique papers: {len(unique_papers)}")
        
        # Automatically save
        self.save_papers(unique_papers, filename="papers.json", output_dir="output")
        
        return unique_papers


    def get_citation_counts(self, papers: List[Paper]) -> List[Paper]:
        """
        Fetch citation counts for papers from Semantic Scholar API.
        Updates papers in-place with citation counts and returns the list.
        
        Args:
            papers: List of Paper objects from arXiv
            
        Returns:
            Same list of Paper objects with citation_count field populated
        """
        if not papers:
            return papers
        
        # Handle if more than 500 papers (Semantic Scholar API limit)
        if len(papers) > 500:
            print(f"Warning: {len(papers)} papers exceeds API limit. Processing first 500 only.")
            papers_to_process = papers[:500]
        else:
            papers_to_process = papers
        
        # Clean arXiv IDs (handle both URL and direct ID formats, remove version numbers)
        arxiv_ids = []
        for p in papers_to_process:
            # Extract ID from URL if present (e.g., "http://arxiv.org/abs/2103.12345v2")
            paper_id = p.id.split('/')[-1] if '/' in p.id else p.id
            # Remove version number (e.g., "2103.12345v2" -> "2103.12345")
            paper_id = paper_id.split('v')[0]
            arxiv_ids.append(paper_id)
        
        # Format for Semantic Scholar API
        paper_ids = [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]
        
        print(f"Fetching citation counts for {len(papers_to_process)} papers...")
        
        try:
            # Single batch API call
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            params = {"fields": "citationCount"}
            
            response = requests.post(url, params=params, json={"ids": paper_ids}, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                
                # Update papers with citation counts
                found_count = 0
                for paper, result in zip(papers_to_process, results):
                    if result and 'citationCount' in result:
                        paper.citation_count = result['citationCount']
                        found_count += 1
                    else:
                        paper.citation_count = None
                
                print(f"Citation counts found for {found_count}/{len(papers_to_process)} papers")
            else:
                print(f"Semantic Scholar API request failed with status {response.status_code}")
                print(f"Response: {response.text[:500]}")  # Show error details
                for paper in papers_to_process:
                    paper.citation_count = None
                    
        except Exception as e:
            print(f"Error fetching citation counts: {e}")
            for paper in papers_to_process:
                paper.citation_count = None
        
        return papers


    def download_papers_as_pdfs(self, papers: List[Paper], base_folder: str = "literature/"):
        """
        Download selected papers as PDFs to specified folder.
        Creates a separate folder for each paper (named by paper ID) containing the PDF.
        The markdown subfolder will be created later by the PDF converter.
        Convenience wrapper around arxiv_api.download_papers_as_pdfs().
        
        Args:
            papers: List of Paper objects to download
            base_folder: Base folder for all papers (will be created if doesn't exist)
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        return self.arxiv_api.download_papers_as_pdfs(papers, base_folder)
    

    def get_bibtex_for_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Get BibTeX citations for multiple papers.
        Uses Semantic Scholar bulk API to retrieve all BibTeX citations once.
        Updates papers in-place with BibTeX and returns the list.
        
        Args:
            papers: List of Paper objects
            
        Returns:
            Same list of Paper objects with bibtex field populated
        """
        if not papers:
            return papers
        
        # Handle if more than 500 papers (Semantic Scholar API limit)
        if len(papers) > 500:
            print(f"Warning: {len(papers)} papers exceeds API limit. Processing first 500 only.")
            papers_to_process = papers[:500]
        else:
            papers_to_process = papers
        
        # Clean arXiv IDs and prepare for Semantic Scholar API
        arxiv_ids = []
        for p in papers_to_process:
            # Extract ID from URL if present (e.g., "http://arxiv.org/abs/2103.12345v2")
            paper_id = p.id.split('/')[-1] if '/' in p.id else p.id
            # Remove version number (e.g., "2103.12345v2" -> "2103.12345")
            paper_id = paper_id.split('v')[0]
            arxiv_ids.append(paper_id)
        
        # Format for Semantic Scholar API
        paper_ids = [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]
        
        print(f"Fetching BibTeX from Semantic Scholar for {len(papers_to_process)} papers...")
        
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            params = {"fields": "citationStyles"}
            
            response = requests.post(url, params=params, json={"ids": paper_ids}, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                
                # Update papers with BibTeX and set citation keys
                found_count = 0
                from phases.paper_search.arxiv_api import _generate_citation_key
                for paper, result in zip(papers_to_process, results):
                    if result and 'citationStyles' in result and result['citationStyles']:
                        bibtex = result['citationStyles'].get('bibtex')
                        if bibtex:
                            paper.bibtex = bibtex
                            # Set citation key from BibTeX
                            paper.citation_key = _generate_citation_key(bibtex, paper.authors, paper.published)
                            found_count += 1
                    else:
                        paper.bibtex = None
                
                print(f"Fetched BibTeX for {found_count}/{len(papers_to_process)} papers")
            else:
                print(f"Semantic Scholar API request failed with status {response.status_code}")
                print(f"Response: {response.text[:500]}")
                for paper in papers_to_process:
                    paper.bibtex = None
                
        except Exception as e:
            print(f"Error fetching BibTeX from Semantic Scholar: {e}")
            for paper in papers_to_process:
                paper.bibtex = None
        
        return papers
    

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
                "updated": paper.updated,
                "authors": paper.authors,
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "comment": paper.comment,
                "journal_ref": paper.journal_ref,
                "citation_count": paper.citation_count,
                "bibtex": paper.bibtex,
                "markdown_text": paper.markdown_text,
                "ranking": asdict(paper.ranking) if paper.ranking else None,
                "citation_key": paper.citation_key
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


if __name__ == "__main__":
    lit_search = LiteratureSearch(model_name='qwen/qwen3-coder-30b')
    
    test_contexts = [
        """Research focus: Deep learning approaches for time series forecasting in financial markets."""
    ]
    
    #print("Building search queries...")
    search_queries = lit_search.build_search_queries(test_contexts)
    
    #print("Executing all searches...")
    all_papers = lit_search.search_papers(search_queries)

    all_papers = lit_search.get_citation_counts(all_papers)

    # print("Downloading papers...")
    # lit_search.arxiv_api.download_papers_as_pdfs(all_papers)