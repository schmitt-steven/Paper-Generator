"""
Section critic for analyzing draft sections and suggesting improvements.

Uses structured LLM output to identify possible improvements and generate
search queries for additional evidence.
"""

import textwrap
from typing import Optional, Sequence

from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import Section, SectionCritique
from utils.llm_utils import remove_thinking_blocks
from settings import Settings
import lmstudio as lms


class SectionCritic:
    """Analyzes draft sections and outputs structured critique with improvement suggestions."""

    def __init__(self):
        pass

    def critique_section(
        self,
        section_type: Section,
        draft_text: str,
        papers: Sequence[Paper],
        max_queries: int = 5,
    ) -> SectionCritique:
        """Analyze a draft section and return structured critique."""

        model = lms.llm(Settings.EVIDENCE_GATHERING_MODEL)
        prompt = self._build_critique_prompt(section_type, draft_text, papers, max_queries)
        
        response = model.respond(
            prompt,
            response_format=SectionCritique,
            config={"temperature": 0.2},
        )
        
        critique = SectionCritique(**response.parsed)
        
        # Enforce max queries limit
        if len(critique.search_queries) > max_queries:
            critique.search_queries = critique.search_queries[:max_queries]
        
        return critique

    def _build_critique_prompt(
        self,
        section_type: Section,
        draft_text: str,
        papers: Sequence[Paper],
        max_queries: int,
    ) -> str:
        """Build the prompt for section critique."""
        
        paper_catalog = self._format_paper_catalog(papers)
        
        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic reviewer analyzing a draft section of a research paper.
            Your goal is to provide constructive feedback that will improve the section.

            [TASK]
            Analyze the draft {section_type.value} section and identify:
            1. Areas that need improvement (use positive framing - what to add/change, not what's wrong)
            2. Search queries to find additional supporting evidence (max {max_queries} queries)

            [SECTION TYPE]
            {section_type.value}

            [DRAFT TEXT]
            {draft_text}

            [AVAILABLE PAPERS]
            The following papers are available in the corpus for citation:
            {paper_catalog}

            [INSTRUCTIONS]
            1. For improvements: Focus on constructive suggestions. Instead of "lacks evidence", say "add supporting evidence for X".
            2. For citations: Verify that ALL citation keys used in the draft (e.g., [Smith2020]) exist in the [AVAILABLE PAPERS] list.
               - If a key is missing from the list, you MUST add an improvement suggestion: "Remove or correct hallucinated citation [Key]".
               - This is a CRITICAL check.
            3. For search queries: Create specific, focused queries that would retrieve relevant academic content.
               - Queries should target gaps in the current draft
               - Each query should be semantically distinct (no redundant queries)
               - Maximum {max_queries} queries

            [OUTPUT FORMAT]
            Return a JSON object with:
            - improvements: list of constructive improvement suggestions (including citation corrections)
            - search_queries: list of search query strings (max {max_queries})"""
        )

    @staticmethod
    def _format_paper_catalog(papers: Sequence[Paper]) -> str:
        """Format papers as a catalog for the critique prompt."""
        if not papers:
            return "No papers available."
        
        items = []
        for paper in papers:
            citation_key = paper.citation_key or "unknown"
            abstract = paper.summary or "No abstract available."
            conclusion = paper.conclusion or "No conclusion extracted."
            
            items.append(f"""[{citation_key}]
Title: {paper.title}
Abstract: {abstract[:500]}{"..." if len(abstract) > 500 else ""}
Conclusion: {conclusion[:300]}{"..." if len(conclusion) > 300 else ""}""")
        
        return "\n\n".join(items)
