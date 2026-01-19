"""
Section critic for analyzing draft sections and suggesting improvements.

Uses structured LLM output to identify possible improvements and generate
search queries for additional evidence.
"""

import textwrap
from typing import Optional, Sequence

from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import Section, SectionCritique
from phases.paper_writing.section_guidelines import SectionGuidelinesLoader
from phases.context_analysis.user_requirements import UserRequirements
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
        user_requirements: Optional[UserRequirements] = None,
    ) -> SectionCritique:
        """Analyze a draft section and return structured critique."""

        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_critique_prompt(
            section_type, draft_text, papers, max_queries, user_requirements
        )
        
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
        user_requirements: Optional[UserRequirements] = None,
    ) -> str:
        """Build the prompt for section critique."""
        
        paper_catalog = self._format_paper_catalog(papers)
        
        # Get section guidelines
        guidelines = SectionGuidelinesLoader.get_guidelines(section_type)
        
        # Get section-specific user requirements
        user_req_block = ""
        if user_requirements:
            section_to_field = {
                Section.ABSTRACT: "abstract",
                Section.INTRODUCTION: "introduction",
                Section.RELATED_WORK: "related_work",
                Section.METHODS: "methods",
                Section.RESULTS: "results",
                Section.DISCUSSION: "discussion",
                Section.CONCLUSION: "conclusion",
            }
            field = section_to_field.get(section_type)
            if field:
                req_text = getattr(user_requirements, field, None)
                if req_text and req_text.strip():
                    user_req_block = f"""
[USER REQUIREMENTS FOR THIS SECTION]
The user has specified the following requirements for the {section_type.value} section:
{req_text.strip()}

CRITICAL: Check if the draft satisfies these user requirements. If not, add specific improvement suggestions.
"""
        
        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic reviewer analyzing a draft section of a research paper.
            Your goal is to provide constructive feedback that will improve the section.

            [TASK]
            Analyze the draft {section_type.value} section and identify:
            1. Violations of section guidelines (HIGHEST PRIORITY)
            2. Areas that need improvement (use positive framing - what to add/change, not what's wrong)
            3. Search queries to find additional supporting evidence (max {max_queries} queries)

            [SECTION TYPE]
            {section_type.value}

            [SECTION GUIDELINES]
            These are the official guidelines for this section. Check for violations:
            {guidelines}

            {user_req_block}

            [DRAFT TEXT]
            {draft_text}

            [AVAILABLE PAPERS]
            The following papers are available in the corpus for citation:
            {paper_catalog}

            [INSTRUCTIONS]
            1. Check for guideline violations (e.g., citations in Abstract, word count limits)
            2. Verify all citation keys exist in [AVAILABLE PAPERS]; flag missing ones
            3. Suggest improvements: be concise, actionable, 1 sentence each
            4. Generate focused search queries for missing evidence (max {max_queries})

            [OUTPUT FORMAT]
            Return JSON with:
            - improvements: concise text with all suggestions (guideline violations first)
            - search_queries: list of query strings (max {max_queries})""")

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
