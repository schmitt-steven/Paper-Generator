"""
Section critic for analyzing draft sections and suggesting improvements.

Uses structured LLM output to identify possible improvements and generate
search queries for additional evidence.
"""

import json
import textwrap
from typing import Optional, Sequence

from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import Section, SectionCritique
from phases.paper_writing.style_guidelines import SectionGuidelinesLoader
from phases.context_analysis.paper_specification import PaperSpecification
from phases.latex_generation.bibliography import extract_citation_keys_from_markdown, create_paper_mapping
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
        paper_specification: Optional[PaperSpecification] = None,
    ) -> SectionCritique:
        """Analyze a draft section and return structured critique."""

        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_critique_prompt(
            section_type, draft_text, papers, max_queries, paper_specification
        )
        
        response = model.respond(
            prompt,
            response_format=SectionCritique,
            config={"temperature": 0.2},
        )
        
        content = remove_thinking_blocks(response.content)
        parsed = json.loads(content)
        critique = SectionCritique(**parsed)
        
        # --- VERIFY CITATIONS ---
        # Extract all citation keys used in the draft
        used_keys = extract_citation_keys_from_markdown(draft_text)
        
        # Create mapping of all valid keys (citation_key and bibtex keys)
        paper_mapping = create_paper_mapping(papers)
        valid_keys = set(paper_mapping.keys())
        
        # Identify hallucinated keys
        hallucinated_keys = used_keys - valid_keys
        
        # If hallucinations found, force a correction
        if hallucinated_keys:
            bad_keys_str = ", ".join(sorted(hallucinated_keys))
            print(f"[SectionCritic] DETECTED HALLUCINATED CITATIONS: {bad_keys_str}")
            
            critique_msg = (
                f"CRITICAL VIOLATION: The draft contains citations that do not exist in the provided evidence: {bad_keys_str}. "
                f"You MUST remove these citations or replace them with the correct keys from the [AVAILABLE PAPERS] list. "
                "Do NOT include any citations that are not explicitly provided."
            )
            
            # Prepend to improvements
            critique.improvements = f"{critique_msg}\n\n{critique.improvements}"

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
        paper_specification: Optional[PaperSpecification] = None,
    ) -> str:
        """Build the prompt for section critique."""
        
        paper_catalog = self._format_paper_catalog(papers)
        
        # Get style guidelines
        guidelines = SectionGuidelinesLoader.get_guidelines(section_type)
        
        # Get section-specific paper specification
        paper_spec_block = ""
        if paper_specification:
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
                req_text = getattr(paper_specification, field, None)
                if req_text and req_text.strip():
                    paper_spec_block = f"""
[PAPER SPECIFICATION FOR THIS SECTION]
The user has specified the following requirements for the {section_type.value} section:
{req_text.strip()}

CRITICAL: Check if the draft satisfies these paper specifications. If not, add specific improvement suggestions.
"""
        
        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic reviewer analyzing a draft section of a research paper.
            Your goal is to provide constructive feedback that will improve the section.

            [TASK]
            Analyze the draft {section_type.value} section and identify:
            1. Violations of style guidelines (HIGHEST PRIORITY)
            2. Areas that need improvement (use positive framing - what to add/change, not what's wrong)
            3. Search queries to find additional supporting evidence (max {max_queries} queries)

            [SECTION TYPE]
            {section_type.value}

            [STYLE GUIDELINES]
            These are the official guidelines for this section. Check for violations:
            {guidelines}

            {paper_spec_block}

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
Abstract: {abstract[:300]}{"..." if len(abstract) > 300 else ""}
Conclusion: {conclusion[:500]}{"..." if len(conclusion) > 500 else ""}""")
        
        return "\n\n".join(items)
