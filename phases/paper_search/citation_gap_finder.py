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
from utils.open_access_finder import find_open_access_pdfs



class SuggestedPaper(BaseModel):
    """A paper suggested by the LLM as missing from the collection."""
    title: str              # Approximate/known title

    reason: str             # Why paper is important for this research


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

YOUR TASK:
Identify up to {max_suggestions} important papers that are commonly cited in this research area but MISSING from the collection above.

RESEARCH TOPIC:
{research_context}

CURRENT PAPER COLLECTION ({len(papers)} papers):
{papers_summary}

Focus on:
1. FOUNDATIONAL WORKS - Seminal papers that introduced key concepts/algorithms
2. HIGHLY-CITED SURVEYS - Major review papers in this field
3. CANONICAL REFERENCES - Papers that are almost always cited in this topic area

IMPORTANT RULES:
- Only suggest papers you are CONFIDENT actually exist
- Suggest papers that would typically be cited in Introduction, Related Work, or Methods sections
- The title must be EXACT and precise. We use it for strict title matching, so typos or approximate titles will fail.
- Do NOT suggest papers that are already in the collection (check titles carefully)

Return your suggestions in the structured format."""

        print(f"\nCitation Gap Analysis: Analyzing {len(papers)} papers for missing foundational works...")
        
        try:
            model = lms.llm(model_name)
            response = model.respond(
                prompt,
                response_format=CitationGapResult,
                config={"temperature": 0.0}
            )
            
            result = response.parsed
            suggestions = result.get("missing_papers", [])
            
            # Convert dicts to SuggestedPaper objects
            suggested_papers = []
            for s in suggestions:
                suggested_papers.append(SuggestedPaper(
                    title=s.get("title", ""),
                    reason=s.get("reason", "")
                ))
            
            print(f"Citation Gap Analysis: LLM suggested {len(suggested_papers)} potentially missing papers:")
            for i, paper in enumerate(suggested_papers, 1):
                print(f"  {i}. {paper.title}")
            
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
        """Search for suggested papers on Semantic Scholar using direct title match."""
        if not suggestions:
            return []
        
        found_papers = []
        
        print(f"\nSearching for {len(suggestions)} suggested foundational papers...")
        
        for suggestion in suggestions:
            try:
                # Match paper using title only (year is often approximate/incorrect in LLM memory)
                match = self.s2_api.match_paper(query=suggestion.title)
                
                if match:
                    if match.id not in existing_paper_ids:
                        found_papers.append(match)
                        existing_paper_ids.add(match.id)
                        print(f"  Found: {match.title[:60]}...")
                    else:
                        print(f"  Already in collection: {match.title[:60]}...")
                else:
                    print(f"  Not found: {suggestion.title}")
                    
            except Exception as e:
                print(f"  Error searching for '{suggestion.title}': {e}")
                continue
        
        print(f"\nCitation Gap Analysis: Found {len(found_papers)} new foundational papers")
        
        # Check for open access PDFs (Unpaywall/arXiv) for any closed source papers
        if found_papers:
            found_papers = find_open_access_pdfs(found_papers)
            
        return found_papers
    
    def _build_papers_summary(self, papers: List[Paper], max_papers: int = 100) -> str:
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
    

