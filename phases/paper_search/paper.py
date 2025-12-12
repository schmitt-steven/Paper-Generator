from typing import List, Optional
from dataclasses import dataclass, field
import re


@dataclass
class RankingScores:
    """Stores ranking score components for a paper"""
    relevance_score: float  # 0-1: Semantic similarity to context
    citation_score: float   # 0-1: Age-aware citation impact
    recency_score: float    # 0-1: Publication recency
    final_score: float      # 0-1: Weighted composite score


@dataclass
class Paper:
    """An academic paper from Semantic Scholar"""
    id: str  # S2 paperId
    title: str
    published: str  # YYYY-MM-DD or YYYY
    authors: list[str]
    summary: str
    pdf_url: Optional[str]
    doi: Optional[str]
    fields_of_study: list[str]
    venue: Optional[str]
    citation_count: Optional[int] = None
    bibtex: Optional[str] = None
    markdown_text: Optional[str] = None
    ranking: Optional[RankingScores] = None
    is_open_access: bool = False
    user_provided: bool = False
    pdf_path: Optional[str] = None
    citation_key: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.citation_key is None:
            self.citation_key = _generate_citation_key(self.bibtex, self.authors, self.published)


def _generate_citation_key(bibtex: Optional[str], authors: list[str], published: str) -> str:
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
            
            # Extract year from published date (format: "YYYY-MM-DD" or "YYYY")
            year_match = re.search(r'(\d{4})', published)
            year = year_match.group(1) if year_match else "unknown"
            
            # Normalize: lowercase, remove special characters
            last_name_normalized = re.sub(r'[^a-zA-Z0-9]', '', last_name.lower())
            return f"{last_name_normalized}{year}"
        
        return "unknown"