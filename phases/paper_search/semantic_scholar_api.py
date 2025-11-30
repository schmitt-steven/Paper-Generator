import requests
import time
import os
from typing import List, Optional
from phases.paper_search.paper import Paper

class SemanticScholarAPI:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = [
        "paperId", "externalIds", "title", "abstract", "authors", 
        "year", "publicationDate", "citationCount", "citationStyles", 
        "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "venue"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce 1 req/sec with API key, 0.5 req/sec without"""
        min_interval = 1.0 if self.api_key else 2.0
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def search_papers(self, query: str, max_results: int = 50) -> List[Paper]:
        """
        Search papers using S2 search endpoint.
        """
        self._rate_limit()
        
        params = {
            "query": query,
            "fields": ",".join(self.FIELDS),
            "limit": min(max_results, 100)
        }
        
        # Retry logic
        max_retries = 3
        backoff = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 429:
                    print(f"Rate limited (429). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 4
                    continue
                
                if response.status_code >= 500:
                    print(f"Server error ({response.status_code}). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 4
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                papers = []
                for item in data.get("data", []):
                    paper = self._to_paper(item)
                    if paper:
                        papers.append(paper)
                
                return papers
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 4
        
        return []
    
    def _to_paper(self, data: dict) -> Optional[Paper]:
        """Convert S2 response to Paper object"""
        # Build PDF URL from open access PDF if available
        pdf_url = None
        if data.get("openAccessPdf"):
            pdf_url = data["openAccessPdf"].get("url")
        
        # Extract author names
        authors = [a.get("name", "") for a in data.get("authors", []) if a.get("name")]
        
        # Extract bibtex
        bibtex = None
        if data.get("citationStyles"):
            bibtex = data["citationStyles"].get("bibtex")
        
        return Paper(
            id=data.get("paperId"),
            title=data.get("title", ""),
            published=data.get("publicationDate") or str(data.get("year", "")),
            authors=authors,
            summary=data.get("abstract") or "",
            pdf_url=pdf_url,
            doi=data.get("externalIds", {}).get("DOI"),
            fields_of_study=[f for f in (data.get("fieldsOfStudy") or []) if f],
            venue=data.get("venue"),
            citation_count=data.get("citationCount"),
            bibtex=bibtex,
            is_open_access=data.get("isOpenAccess", False),
        )
