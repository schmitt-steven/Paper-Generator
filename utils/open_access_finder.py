"""
Open Access PDF finder utility.
Checks multiple sources for free PDF versions of papers:
1. Unpaywall (if DOI available) - covers arXiv, institutional repos, author websites
2. arXiv direct search (fallback for papers without DOI)

Checks papers that are either:
- Marked as closed access (is_open_access=False)
- Marked as open access but missing pdf_url
"""
import requests
import time
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from phases.literature_search.paper import Paper


# Default email for Unpaywall API (used if Settings.UNPAYWALL_EMAIL is not set)
DEFAULT_UNPAYWALL_EMAIL = "paper.generator.app@gmail.com"


def _get_unpaywall_email() -> str:
    """Get email for Unpaywall API from settings, with fallback."""
    try:
        from settings import Settings
        email = getattr(Settings, "UNPAYWALL_EMAIL", "").strip()
        # Basic validation: must contain @ and .
        if email and "@" in email and "." in email:
            return email
    except ImportError:
        pass
    return DEFAULT_UNPAYWALL_EMAIL


def find_open_access_pdfs(papers: List[Paper], delay: float = 0.3) -> List[Paper]:
    """
    Find free PDF versions for papers that need them.
    
    Checks Unpaywall (via DOI) and arXiv (via title search).
    Updates is_open_access and pdf_url if found.
    """
    papers_to_check = [
        p for p in papers
        if not p.user_provided and (not p.is_open_access or not p.pdf_url)
    ]
    
    if not papers_to_check:
        print("Open Access Check: All papers already have PDF URLs")
        return papers
    
    print(f"Open Access Check: Checking {len(papers_to_check)} papers...")
    
    found_count = 0
    for i, paper in enumerate(papers_to_check):
        pdf_url = None
        source = None
        
        # Try arXiv title search first
        if paper.title:
            pdf_url = _search_arxiv_by_title(paper.title)
            if pdf_url:
                source = "arXiv"
        
        # Fallback to Unpaywall if DOI exists
        if not pdf_url and paper.doi:
            pdf_url = _check_unpaywall(paper.doi)
            if pdf_url:
                source = "Unpaywall"
        
        if pdf_url:
            paper.is_open_access = True
            paper.pdf_url = pdf_url
            found_count += 1
            print(f"  [FOUND via {source}] {paper.title[:45]}...")
        else:
            paper.is_open_access = False
            print(f"  [Not Found] {paper.title[:45]}...")
        
        # Rate limiting
        if i < len(papers_to_check) - 1:
            time.sleep(delay)
    
    print(f"Open Access Check: Found {found_count}/{len(papers_to_check)} papers")
    return papers


def _check_unpaywall(doi: str) -> Optional[str]:
    """Check Unpaywall for a free PDF version using DOI."""
    try:
        response = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": _get_unpaywall_email()},
            timeout=10
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Try best_oa_location first
        best_oa = data.get("best_oa_location")
        if best_oa:
            pdf_url = best_oa.get("url_for_pdf")
            if pdf_url:
                return pdf_url
        
        # Check all OA locations
        for location in data.get("oa_locations", []):
            pdf_url = location.get("url_for_pdf")
            if pdf_url:
                return pdf_url
        
        return None
        
    except Exception:
        return None


def _search_arxiv_by_title(title: str) -> Optional[str]:
    """Search arXiv for a paper by title. Tries exact phrase first, then AND query."""
    clean_title = _clean_title_for_search(title)

    if not clean_title or len(clean_title) < 10:
        return None

    # Build queries: try exact phrase first, then AND of key words
    queries = [f'ti:"{clean_title}"']

    # Build AND query from significant words (skip very common ones)
    stop_words = {'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and', 'with', 'by', 'from', 'is', 'are', 'at', 'as', 'or', 'its', 'via'}
    words = [w for w in clean_title.split() if w not in stop_words and len(w) > 1]
    if len(words) >= 2:
        and_query = " AND ".join(f'ti:{w}' for w in words)
        queries.append(and_query)

    for search_query in queries:
        result = _query_arxiv(search_query, clean_title)
        if result:
            return result

    return None


def _query_arxiv(search_query: str, clean_title: str) -> Optional[str]:
    """Execute a single arXiv API query and return PDF URL if a matching paper is found."""
    try:
        response = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": 5
            },
            timeout=10
        )
        
        if response.status_code != 200:
            return None
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = root.findall('atom:entry', ns)
        
        for entry in entries:
            entry_title = entry.find('atom:title', ns)
            if entry_title is not None:
                arxiv_title = _clean_title_for_search(entry_title.text)
                if _titles_match(clean_title, arxiv_title):
                    # Get PDF link
                    for link in entry.findall('atom:link', ns):
                        if link.get('title') == 'pdf':
                            return link.get('href')
                    
                    # Fallback: construct from ID
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None:
                        arxiv_id = id_elem.text.split('/')[-1]
                        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return None
        
    except Exception:
        return None


def _clean_title_for_search(title: str) -> str:
    """Clean title for arXiv search query."""
    if not title:
        return ""
    
    clean = re.sub(r'[^\w\s]', ' ', title)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean.lower()


def _titles_match(title1: str, title2: str, threshold: float = 0.85) -> bool:
    """Check if two titles are similar enough to be the same paper."""
    if not title1 or not title2:
        return False
    
    words1 = set(title1.split())
    words2 = set(title2.split())
    
    if not words1 or not words2:
        return False
    
    intersection = words1 & words2
    union = words1 | words2
    
    jaccard = len(intersection) / len(union)
    return jaccard >= threshold
