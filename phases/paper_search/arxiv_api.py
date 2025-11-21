from datetime import datetime
import urllib.request as libreq
import ssl
import feedparser
import re
import os
import time
import shutil
import time
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class RankingScores:
    """Stores ranking score components for a paper"""
    relevance_score: float  # 0-1: Semantic similarity to context
    citation_score: float   # 0-1: Age-aware citation impact
    recency_score: float    # 0-1: Publication recency
    final_score: float      # 0-1: Weighted composite score


def _generate_citation_key(bibtex: Optional[str], authors: List[str], published: str) -> str:
    """
    Generate a citation key for a paper.
    
    Priority:
    1. Extract from BibTeX entry key if available (e.g., @article{Diekhoff2024RecursiveBQ, -> Diekhoff2024RecursiveBQ)
    2. Generate from first author last name + year (e.g., diekhoff2024)
    """
    # Try to extract from BibTeX first (use original BibTeX keys)
    if bibtex:
        # Look for pattern: @article{key, or @inproceedings{key, etc.
        match = re.search(r'@\w+\{([^,]+)', bibtex)
        if match:
            return match.group(1).strip()
    
    # Fallback: Generate from first author + year
    if authors and len(authors) > 0:
        first_author = authors[0].strip()
        if ',' in first_author:
            # Format: "Last, First"
            last_name = first_author.split(',')[0].strip()
        else:
            # Format: "First Last"
            parts = first_author.split()
            last_name = parts[-1] if parts else first_author
        
        # Extract year from published date (format: "YYYY-MM-DDTHH:MM:SSZ")
        year_match = re.search(r'(\d{4})', published)
        year = year_match.group(1) if year_match else "unknown"
        
        # Normalize: lowercase, remove special characters
        last_name_normalized = re.sub(r'[^a-zA-Z0-9]', '', last_name.lower())
        return f"{last_name_normalized}{year}"
    
    return "unknown"


@dataclass
class Paper:
    """Represents an academic paper from arXiv"""
    id: str
    title: str
    published: str
    updated: str
    authors: List[str]
    summary: str
    pdf_url: Optional[str]
    doi: Optional[str]
    categories: List[str]
    primary_category: Optional[str]
    comment: Optional[str]
    journal_ref: Optional[str]
    citation_count: Optional[int] = None
    bibtex: Optional[str] = None
    markdown_text: Optional[str] = None
    ranking: Optional[RankingScores] = None
    citation_key: Optional[str] = field(default=None, init=False)



class ArxivAPI:
    def __init__(self):
        pass


    def search_papers(self, search_query: str, max_results: int = 50):
        """
        Search arXiv papers using their API query syntax.
        
        Arguments:
            search_query: Query string using arXiv's search syntax.
            max_results: Maximum number of results to return (default: 50, arXiv max: 2000)
                
                AVAILABLE FIELD PREFIXES:
                - ti:  Search in title (e.g., ti:%22neural+network%22)
                - abs: Search in abstract (e.g., abs:%22transformer%22)
                - au:  Search by author (e.g., au:%22LeCun%22)
                - all: Search all fields (e.g., all:%22quantum%22)

                BOOLEAN OPERATORS (must be UPPERCASE):
                - +AND+: Both terms required
                - +OR+:  Either term
                - +ANDNOT+: Exclude term
                
                FORMATTING RULES:
                - Always wrap search phrases with %22...%22 and use + for spaces (e.g. ti:%22neural+network+architecture%22)
                - Between operators: Use + (e.g. ti:%22bert+AND+abs:%22attention%22)
                - Grouping: Use %28 for ( and %29 for ) (e.g. %28ti:%22bert+OR+ti:%22gpt%22%29+AND+au:%22devlin%22)
                
                EXAMPLES:
                
                Simple searches:
                - ti:%22transformer%22 - Papers with 'transformer' in title
                - au:%22bengio%22 - Papers by anyone named Bengio
                - abs:%22self+supervised+learning%22 
                - all:%22electron+field%22
                
                Boolean searches (note the + around operators):
                - ti:%22BERT%22+AND+abs:%22classification%22 - BERT in title AND classification in abstract
                - au:%22Hinton%22+AND+ti:%22capsule%22 - Papers by Hinton with 'capsule' in title
                - ti:%22gan%22+ANDNOT+au:%22Hinton%22 - Papers about GANs not by Hinton
                - au:%22LeCun%22+OR+au:%22Hinton%22 - Papers by LeCun or Hinton

                Complex queries with grouping:
                - %28ti:bert+OR+ti:gpt%29+AND+au:devlin - BERT or GPT in title by Devlin
                - au:%22goodfellow%22+AND+ti:%22adversarial%22+AND+abs:%22training%22
                - %28ti:%22vision+transformer%22+OR+ti:ViT%29+AND+abs:%22image+classification%22
                
        Returns:
            List of paper dictionaries with id, title, authors, summary, pdf_url, etc.
        """
        BASE_URL = 'http://export.arxiv.org/api/query?'

        START = 0  # Maximum is 30.000
        SORT_BY = 'relevance'  # Options: 'relevance', 'lastUpdatedDate', 'submittedDate'
        SORT_ORDER = 'descending'  # Options: 'ascending', 'descending'; SORT_ORDER requires SORT_BY to work
        SUBMITTED_DATE_FROM = '200001010000'  # Format: YYYYMMDDTTTT
        SUBMITTED_DATE_TO = datetime.now().strftime('%Y%m%d%H%M')  # Today's date, format: YYYYMMDDTTTT

        parameters = (
            f"search_query={search_query}"
            f"&start={START}"
            f"&max_results={max_results}"
            f"&sort_by={SORT_BY}"
            f"&sort_order={SORT_ORDER}"
            f"&submittedDate={SUBMITTED_DATE_FROM}+TO+{SUBMITTED_DATE_TO}"
        )

        query_url = BASE_URL + parameters

        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with libreq.urlopen(query_url, context=ssl_context) as response:
            return response.read().decode('utf-8')
    

    def parse_response(self, response) -> List[Paper]:
        """
        Parse arXiv API response and return list of Paper objects.
        
        Args:
            response: Raw XML response from arXiv API
            
        Returns:
            List of Paper objects
        """
        parsed = feedparser.parse(response)
        papers = []
        for entry in parsed.entries:
            paper = Paper(
                id=entry.id,
                title=entry.title,
                published=entry.published,
                updated=entry.updated,
                authors=[author.name for author in entry.authors],
                summary=entry.summary,
                pdf_url=next((link.href for link in entry.links if link.get('title') == 'pdf'), None),
                doi=getattr(entry, 'arxiv_doi', None),
                categories=[cat.term for cat in entry.get('tags', [])],
                primary_category=getattr(entry, 'arxiv_primary_category', {}).get('term', None),
                comment=getattr(entry, 'arxiv_comment', None),
                journal_ref=getattr(entry, 'arxiv_journal_ref', None),
            )
            papers.append(paper)
        return papers
    

    def download_pdf(self, pdf_url: str, filename: str):
        """
        Download a single PDF from arXiv.
        
        Args:
            pdf_url: URL to the PDF
            filename: Path where PDF should be saved
        """
        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with libreq.urlopen(pdf_url, context=ssl_context) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())


    def download_papers_as_pdfs(self, papers: List[Paper], base_folder: str = "literature/"):
        """
        Download selected papers as PDFs to specified folder.
        Creates a separate folder for each paper (named by paper ID) containing the PDF.
        Includes 1-second rate limiting to be polite to arXiv servers.
        Deletes all existing papers in the base_folder before downloading new ones.
        
        Args:
            papers: List of Paper objects to download
            base_folder: Base folder for all papers (will be created if doesn't exist)
        """
        
        # Delete existing papers folder if it exists
        if os.path.exists(base_folder):
            print(f"Deleting existing papers in '{base_folder}'...")
            shutil.rmtree(base_folder)
        
        # Create base folder
        os.makedirs(base_folder, exist_ok=True)
        
        print(f"Downloading {len(papers)} papers from arXiv to '{base_folder}'...")
        successful = 0
        failed = 0
        
        for i, paper in enumerate(papers, 1):
            if paper.pdf_url:
                # Extract paper ID from URL (e.g., "http://arxiv.org/abs/2001.08317v1" -> "2001.08317v1")
                paper_id = paper.id.split('/')[-1]
                
                # Create folder for this paper
                paper_folder = os.path.join(base_folder, paper_id)
                os.makedirs(paper_folder, exist_ok=True)
                
                # Save PDF in paper's folder
                filename = os.path.join(paper_folder, f"{paper_id}.pdf")
                
                try:
                    self.download_pdf(paper.pdf_url, filename)
                    print(f"  [{i}/{len(papers)}] {paper.title[:50]}...")
                    successful += 1
                except Exception as e:
                    print(f"  [{i}/{len(papers)}] FAILED: {paper.title[:50]}... ({e})")
                    failed += 1
            else:
                print(f"  [{i}/{len(papers)}] NO PDF URL: {paper.title[:50]}...")
                failed += 1
            
            # Add 1-second delay between downloads (except after last one)
            if i < len(papers):
                time.sleep(1)
        
        print(f"Download complete: {successful} successful, {failed} failed\n")


    def get_bibtex(self, paper_id: str) -> str:
        """
        Get BibTeX citation for an arXiv paper.
        
        Args:
            paper_id: arXiv paper ID
            
        Returns:
            BibTeX citation string
        """
        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        url = f'https://arxiv.org/bibtex/{paper_id}'
        with libreq.urlopen(url, context=ssl_context) as response:
            return response.read().decode('utf-8')


    def get_bibtex_for_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Get BibTeX citations for multiple papers.
        Includes 1-second rate limiting to be polite to arXiv servers.
        Updates papers in-place with BibTeX and returns the list.
        
        Args:
            papers: List of Paper objects
            
        Returns:
            Same list of Paper objects with bibtex field populated
        """
        
        print(f"Fetching BibTeX for {len(papers)} papers...")
        found_count = 0
        
        for i, paper in enumerate(papers, 1):
            # Extract paper ID from URL (e.g., "http://arxiv.org/abs/2301.00001" -> "2301.00001")
            paper_id = paper.id.split('/')[-1]
            
            try:
                bibtex = self.get_bibtex(paper_id)
                paper.bibtex = bibtex
                # Set citation key from BibTeX
                paper.citation_key = _generate_citation_key(bibtex, paper.authors, paper.published)
                found_count += 1
                print(f"  [{i}/{len(papers)}] Got BibTeX for {paper_id}")
            except Exception as e:
                paper.bibtex = None
                print(f"  [{i}/{len(papers)}] F for {paper_id}: {e}")
            
            # Add 1-second delay between requests (except after last one)
            if i < len(papers):
                time.sleep(1)
        
        print(f"\nBibTeX retrieval complete: {found_count}/{len(papers)} successful")
        return papers