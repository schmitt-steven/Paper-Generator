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
        """
        Enforce rate limiting.
        Without key: 5,000 requests per 5 minutes (~16/sec) - we use 2 sec to be conservative
        With key: 1 request per second for search endpoint - we use 1 sec
        """
        min_interval = 1.0 if self.api_key else 2.0
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def search_papers(self, query: str, max_results: int = 100, year: Optional[str] = None, fields_of_study: Optional[str] = None, open_access_only: bool = True) -> List[Paper]:
        """
        Search papers using S2 search endpoint.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (max 100)
            year: Optional year filter (e.g., "2020-2024" or "2020")
            fields_of_study: Optional comma-separated fields of study filter (e.g., "Computer Science,Mathematics")
            open_access_only: If True, filter results to only open access papers (client-side)
        """
        self._rate_limit()
        
        params = {
            "query": query,
            "fields": ",".join(self.FIELDS),
            "limit": min(max_results, 100),
        }
        
        # Add year filter if provided
        if year:
            params["year"] = year
        
        # Add fields of study filter if provided
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        
        max_retries = 3
        backoff = 2
        
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
                    backoff *= 3
                    continue
                
                if response.status_code >= 500:
                    print(f"Server error ({response.status_code}). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                
                # Check for other error status codes
                if response.status_code != 200:
                    error_text = response.text[:200] if response.text else "No error message"
                    print(f"API error ({response.status_code}): {error_text}")
                    if attempt == max_retries - 1:
                        return []  # Return empty list instead of raising
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                
                data = response.json()
                
                # Check if we got data
                if "data" not in data:
                    print(f"Warning: No 'data' field in response. Response keys: {list(data.keys())}")
                    if "message" in data:
                        print(f"API message: {data['message']}")
                    return []
                
                papers = []
                for item in data.get("data", []):
                    paper = self._to_paper(item)
                    if paper:
                        # Filter for open access papers if requested (client-side)
                        if open_access_only and not paper.is_open_access:
                            continue
                        papers.append(paper)
                
                # Add delay after successful call to avoid rate limiting on subsequent requests
                time.sleep(0.5)
                return papers
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 3
        
        return []
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """
        Fetch a single paper by its ID.
        Supports Semantic Scholar IDs and external IDs like ARXIV:2404.15822
        
        Args:
            paper_id: Paper ID (S2 ID or external ID like "ARXIV:2404.15822")
            
        Returns:
            Paper object or None if not found
        """
        params = {
            "fields": ",".join(self.FIELDS)
        }
        
        max_retries = 3
        backoff = 2
        
        for attempt in range(max_retries):
            self._rate_limit()
            
            try:
                response = requests.get(
                    f"{self.BASE_URL}/paper/{paper_id}",
                    params=params,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 404:
                    return None
                
                if response.status_code == 429:
                    print(f"    Rate limited (429). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                
                if response.status_code >= 500:
                    print(f"    Server error ({response.status_code}). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                
                response.raise_for_status()
                data = response.json()
                paper = self._to_paper(data)
                
                # Add delay to avoid rate limiting on subsequent requests
                time.sleep(2)
                return paper
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"    Failed to fetch paper {paper_id}: {e}")
                    return None
                time.sleep(backoff)
                backoff *= 3
        
        return None

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
            id=data.get("paperId", "No ID found"),
            title=data.get("title", "No title found"),
            published=data.get("publicationDate") or str(data.get("year", "")),
            authors=authors,
            summary=data.get("abstract", "No abstract found"),
            pdf_url=pdf_url,
            doi=data.get("externalIds", {}).get("DOI"),
            fields_of_study=[f for f in (data.get("fieldsOfStudy") or []) if f],
            venue=data.get("venue"),
            citation_count=data.get("citationCount"),
            bibtex=bibtex,
            is_open_access=data.get("isOpenAccess", False),
            user_provided=False,
        )
