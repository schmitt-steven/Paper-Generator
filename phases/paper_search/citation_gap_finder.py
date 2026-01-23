"""
Citation Gap Finder - Identifies missing foundational papers using LLM.

After auto-search returns filtered papers, this class prompts the LLM to identify
well-known/foundational papers that are commonly cited but missing from the collection.
It then searches for those papers and returns them for integration.
"""

import lmstudio as lms
import time
from typing import List, Optional
from pydantic import BaseModel

from phases.paper_search.paper import Paper
from phases.paper_search.semantic_scholar_api import SemanticScholarAPI
from settings import Settings
from difflib import SequenceMatcher


class SuggestedPaper(BaseModel):
    """A paper suggested by the LLM as missing from the collection."""
    title: str              # Approximate/known title
    authors: str            # Key author names (e.g., "Sutton and Barto")
    year: Optional[int]     # Approximate publication year
    reason: str             # Why this paper is important for this research
    search_query: str       # Suggested Semantic Scholar query to find it


class CitationGapResult(BaseModel):
    """Structured response from LLM gap analysis."""
    missing_papers: List[SuggestedPaper]


class CitationGapFinder:
    """
    Identifies missing foundational/highly-cited papers using LLM analysis.
    
    This also addresses the problem of LLM hallucinating citations to well-known papers
    (e.g., Sutton & Barto, Bellman) during paper writing. By proactively identifying
    these papers before writing and actually searching for them, we ensure they're
    in the collection.
    """
    
    def __init__(self):
        self.s2_api = SemanticScholarAPI(api_key=Settings.SEMANTIC_SCHOLAR_API_KEY or None)
    
    def identify_missing_papers(
        self,
        papers: List[Paper],
        research_context: str,
        model_name: str,
        max_suggestions: int = 10
    ) -> List[SuggestedPaper]:
        """Use LLM to identify foundational papers missing from the current collection."""
        if not papers:
            return []
        
        # Build paper list summary for LLM
        papers_summary = self._build_papers_summary(papers)
        
        prompt = f"""You are an expert academic researcher. Analyze this paper collection for a literature review and identify MISSING foundational or highly-cited papers.

RESEARCH TOPIC:
{research_context}

CURRENT PAPER COLLECTION ({len(papers)} papers):
{papers_summary}

YOUR TASK:
Identify up to {max_suggestions} important papers that are commonly cited in this research area but MISSING from the collection above.

Focus on:
1. FOUNDATIONAL WORKS - Seminal papers that introduced key concepts/algorithms
2. HIGHLY-CITED SURVEYS - Major review papers in this field
3. CANONICAL REFERENCES - Papers that are almost always cited in this topic area

IMPORTANT RULES:
- Only suggest papers you are CONFIDENT actually exist
- Suggest papers that would typically be cited in Introduction, Related Work, or Methods sections
- The search_query should be specific enough to find the exact paper on Semantic Scholar
- Do NOT suggest papers that are already in the collection (check titles carefully)

Return your suggestions in the structured format."""

        print(f"\nCitation Gap Analysis: Analyzing {len(papers)} papers for missing foundational works...")
        
        try:
            model = lms.llm(model_name)
            response = model.respond(
                prompt,
                response_format=CitationGapResult,
                config={"temperature": 0.1}
            )
            
            result = response.parsed
            suggestions = result.get("missing_papers", [])
            
            # Convert dicts to SuggestedPaper objects
            suggested_papers = []
            for s in suggestions:
                suggested_papers.append(SuggestedPaper(
                    title=s.get("title", ""),
                    authors=s.get("authors", ""),
                    year=s.get("year"),
                    reason=s.get("reason", ""),
                    search_query=s.get("search_query", s.get("title", ""))
                ))
            
            print(f"Citation Gap Analysis: LLM suggested {len(suggested_papers)} potentially missing papers:")
            for i, paper in enumerate(suggested_papers, 1):
                year_str = f" ({paper.year})" if paper.year else ""
                print(f"  {i}. {paper.title}{year_str} - {paper.authors}")
            
            return suggested_papers
            
        except Exception as e:
            print(f"Citation Gap Analysis: Error during LLM analysis - {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_suggested_papers(
        self,
        suggestions: List[SuggestedPaper],
        existing_paper_ids: set
    ) -> List[Paper]:
        """Search for suggested papers on Semantic Scholar."""
        if not suggestions:
            return []
        
        found_papers = []
        
        print(f"\nSearching for {len(suggestions)} suggested foundational papers...")
        
        for suggestion in suggestions:
            try:
                # Search using the suggested query
                results = self.s2_api.search_papers(
                    suggestion.search_query,
                    max_results=3  # Get top 3 results to find the best match
                )
                
                if not results:
                    print(f"  Not found: {suggestion.title}")
                    continue
                
                # Try to find the best match
                best_match = self._find_best_match(suggestion, results)
                
                if best_match and best_match.id not in existing_paper_ids:
                    found_papers.append(best_match)
                    existing_paper_ids.add(best_match.id)  # Prevent duplicates in this batch
                    print(f"  Found: {best_match.title[:60]}...")
                elif best_match:
                    print(f"  Already in collection: {best_match.title[:60]}...")
                else:
                    print(f"  No good match for: {suggestion.title}")
                    
            except Exception as e:
                print(f"  Error searching for '{suggestion.title}': {e}")
                continue
            
            # delay between searches
            time.sleep(1.0)
        
        print(f"\nCitation Gap Analysis: Found {len(found_papers)} new foundational papers")
        return found_papers
    
    def _build_papers_summary(self, papers: List[Paper], max_papers: int = 50) -> str:
        """Build a concise summary of papers for the LLM prompt."""
        lines = []
        
        # Limit to avoid token overflow
        display_papers = papers[:max_papers]
        
        for paper in display_papers:
            year = "Unknown"
            if paper.published:
                if hasattr(paper.published, "year"):
                    year = str(paper.published.year)
                else:
                    year = str(paper.published)[:4]
            
            first_author = paper.authors[0] if paper.authors else "Unknown"
            lines.append(f"- {paper.title} ({first_author}, {year})")
        
        if len(papers) > max_papers:
            lines.append(f"... and {len(papers) - max_papers} more papers")
        
        return "\n".join(lines)
    
    def _find_best_match(
        self,
        suggestion: SuggestedPaper,
        results: List[Paper]
    ) -> Optional[Paper]:
        """Find the best matching paper from search results."""
        
        suggested_title = suggestion.title.lower()
        suggested_authors = suggestion.authors.lower()
        
        best_paper = None
        best_score = 0.0
        
        for paper in results:
            # Title similarity
            paper_title = paper.title.lower()
            title_sim = SequenceMatcher(None, suggested_title, paper_title).ratio()
            
            # Author matching (check if any suggested author appears)
            author_score = 0.0
            if paper.authors:
                paper_authors_str = " ".join(paper.authors).lower()
                # Check for author name fragments
                for author_part in suggested_authors.split():
                    if len(author_part) > 2 and author_part in paper_authors_str:
                        author_score = 0.5
                        break
            
            # Year matching bonus
            year_score = 0.0
            if suggestion.year and paper.published:
                try:
                    paper_year = None
                    if hasattr(paper.published, "year"):
                        paper_year = paper.published.year
                    else:
                        paper_year = int(str(paper.published)[:4])
                        
                    if paper_year == suggestion.year:
                        year_score = 0.2
                    elif abs(paper_year - suggestion.year) <= 2:
                        year_score = 0.1
                except (ValueError, TypeError):
                    pass
            
            # Combined score
            total_score = title_sim * 0.6 + author_score * 0.25 + year_score * 0.15
            
            if total_score > best_score:
                best_score = total_score
                best_paper = paper
        
        # Require minimum confidence
        if best_score >= 0.7:
            return best_paper
        
        return None
