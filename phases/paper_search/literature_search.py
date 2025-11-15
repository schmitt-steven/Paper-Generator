from phases.paper_search.arxiv_api import ArxivAPI, Paper, RankingScores
from typing import List
from dataclasses import asdict
import textwrap
import lmstudio as lms
from lmstudio import BaseModel
import time
import requests
import json
import os
from datetime import datetime


class SearchQuery(BaseModel):
    """Represents a single search query with metadata"""
    label: str
    query: str
    description: str


class SearchQueriesResult(BaseModel):
    """Structured response format for multiple search queries"""
    queries: List[SearchQuery]


class LiteratureSearch:
    
    def __init__(self, model_name: str):
        """
        Initialize LiteratureSearch with a language model.
        
        Args:
            model_name: Name of the LLM model to use for query generation
        """
        self.model = lms.llm(model_name)
        self.arxiv_api = ArxivAPI()


    def build_search_queries(self, contexts: List[str]) -> List[SearchQuery]:
        """
        Generate multiple search queries from contexts for different aspects of academic writing.
        
        Args:
            contexts: List of research context strings, each describing a topic to search for
            
        Returns:
            List of SearchQuery objects with label, query, and description
        """

        system_prompt = """Task: Generate multiple arXiv API search queries usable for a whole and effective literature search.
        Use the provided context to generate the search queries.

        ARXIV QUERY SYNTAX:
        Field: abs: (abstract)
        Operators: +AND+, +OR+, +ANDNOT+ (uppercase only)
        Quotes: %22phrase%22 (spaces become +)
        Grouping: %28...%29 required when mixing +OR+ with +AND+/+ANDNOT+


        GENERATE 6 DIVERSE QUERIES covering different angles:
        - Broad queries (different term combinations)
        - Specific queries (different established methods)
        - Related queries (alternative approaches)  
        - Survey/Review queries (overview papers and theoretical foundations)

        Vary the terms used - don't repeat the same concepts in every query.
        Better to have some redundancy than miss relevant papers.

        OUTPUT: SearchQueriesResult with label, query, description for each SearchQuery."""

        chat = lms.Chat(system_prompt)
        all_search_queries = []
        
        print("Generating search queries, this could take a while...")
        for i, context in enumerate(contexts, 1):
            user_message = textwrap.dedent(f"""\
                RESEARCH CONTEXT:
                {context}
                Now generate the search queries.
            """)

            chat.add_user_message(user_message)
            
            result = self.model.respond(
                chat,
                response_format=SearchQueriesResult,
                config={
                    'temperature': 0.5,
                }
            ).parsed
            
            # Convert dict results to SearchQuery objects
            search_queries = [
                SearchQuery(**query_obj) for query_obj in result['queries']
            ]
            all_search_queries.extend(search_queries)
        
        print(f"Generated {len(all_search_queries)} search queries.")
        
        # Automatically save
        self.save_search_queries(all_search_queries, filename="search_queries.json", output_dir="output")
        
        return all_search_queries
    

    def save_search_queries(self, queries: List[SearchQuery], filename: str = None, output_dir: str = "output"):
        """Save search queries to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_queries_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        queries_data = [{"label": q.label, "query": q.query, "description": q.description} for q in queries]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(queries_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(queries)} search queries to {filepath}")
        return filepath
    

    @staticmethod
    def load_search_queries(filepath: str) -> List[SearchQuery]:
        """Load search queries from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
                
                # Update papers with BibTeX
                found_count = 0
                for paper, result in zip(papers_to_process, results):
                    if result and 'citationStyles' in result and result['citationStyles']:
                        bibtex = result['citationStyles'].get('bibtex')
                        if bibtex:
                            paper.bibtex = bibtex
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
    

    def save_papers(self, papers: List[Paper], filename: str = None, output_dir: str = "output"):
        """Save papers to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"papers_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
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
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(papers)} papers to {filepath}")
        return filepath
    

    @staticmethod
    def load_papers(filepath: str) -> List[Paper]:
        """Load papers from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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