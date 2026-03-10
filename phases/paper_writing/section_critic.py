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
        max_queries: Optional[int] = None,
        paper_specification: Optional[PaperSpecification] = None,
        previous_sections: Optional[dict[Section, str]] = None,
        section_order: Optional[Sequence[Section]] = None,
    ) -> SectionCritique:
        """Analyze a draft section and return structured critique."""
        if max_queries is None:
            max_queries = getattr(Settings, "CRITIC_MAX_SEARCH_QUERIES", 3)

        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_critique_prompt(
            section_type, draft_text, papers, max_queries, paper_specification,
            previous_sections or {}, section_order,
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
        previous_sections: Optional[dict[Section, str]] = None,
        section_order: Optional[Sequence[Section]] = None,
    ) -> str:
        """Build the prompt for section critique."""
        
        paper_catalog = self._format_paper_catalog(papers)
        paper_structure_block = self._format_paper_structure(section_type, section_order)
        previous_sections_block = self._format_previous_sections(previous_sections or {})

        # Get style guidelines
        guidelines = SectionGuidelinesLoader.get_guidelines(section_type)

        # Build query limit hint for the prompt
        if max_queries == 0:
            query_limit_note = "The query limit of 0 is intentionally set by the user. Return an EMPTY list for search_queries."
        else:
            query_limit_note = "The query limit of {} is intentionally set by the user — respect it strictly.".format(max_queries)

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
            3. Cross-section issues: redundancy, coherence, consistency with previously written sections
            4. Search queries to find additional supporting evidence (max {max_queries} queries)

            [SECTION TYPE]
            {section_type.value}

            [PAPER STRUCTURE]
            {paper_structure_block}

            [PREVIOUSLY WRITTEN SECTIONS]
            {previous_sections_block}

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
            3. Check for duplicate explanations: compare the draft against [PREVIOUSLY WRITTEN SECTIONS] and identify any concept, mechanism, or process that is explained in both. A concept should be explained in detail ONCE (typically in Methods), and other sections should reference it briefly without re-explaining. Quote the duplicate passages from each section.
            4. Check cross-section coherence: flag inconsistent terminology (e.g., same thing called different names across sections), contradictory claims, or mismatched framing between sections
            5. Consider the full paper structure: ensure this section fulfills its role without overlapping with upcoming sections
            6. Suggest improvements: be concise, actionable, 1 sentence each
            7. Generate focused search queries for missing evidence (max {max_queries}).
               {query_limit_note}

            [OUTPUT FORMAT]
            Return JSON with:
            - improvements: concise text with all suggestions (guideline violations first, then duplicate/cross-section issues with quoted passages, then other improvements)
            - search_queries: list of query strings (max {max_queries})""")

    @staticmethod
    def _format_paper_structure(
        current_section: Section,
        section_order: Optional[Sequence[Section]] = None,
    ) -> str:
        """Format the full paper structure, marking the current section and written/upcoming status."""
        if not section_order:
            return "Not available."

        lines = []
        for section in section_order:
            if section == current_section:
                lines.append(f"  → {section.value}  ← (CURRENT - under review)")
            else:
                lines.append(f"  - {section.value}")
        return "\n".join(lines)

    @staticmethod
    def _format_previous_sections(previous_sections: dict[Section, str]) -> str:
        """Format previously written sections for cross-section review."""
        if not previous_sections:
            return "No sections written yet."

        parts = []
        for section, text in previous_sections.items():
            parts.append(f"--- {section.value} ---\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove PDF extraction artifacts like mid-sentence line breaks."""
        import re
        # Replace single newlines (not paragraph breaks) with spaces
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    @staticmethod
    def _format_paper_catalog(papers: Sequence[Paper]) -> str:
        """Format papers as a catalog for the critique prompt."""
        if not papers:
            return "No papers available."

        items = []
        for paper in papers:
            citation_key = paper.citation_key or "unknown"
            abstract = SectionCritic._normalize_text(paper.summary or "No abstract available.")
            conclusion = SectionCritic._normalize_text(paper.conclusion or "No conclusion extracted.")

            lines = [
                f"[{citation_key}]",
                f"Title: {paper.title}",
                f"Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}",
                f"Conclusion: {conclusion[:500]}{'...' if len(conclusion) > 500 else ''}",
            ]
            items.append("\n".join(lines))

        return "\n\n".join(items)
