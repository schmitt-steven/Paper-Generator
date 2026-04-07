import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

from phases.context_analysis.research_context_generator import ResearchContext
from phases.context_analysis.paper_specification import PaperSpecification
from phases.experimentation.experiment_state import ExperimentResult, Plot
from phases.literature_search.paper import Paper
from phases.paper_writing.data_models import Evidence, PaperDraft, Section, SectionCritique
from phases.paper_writing.style_guidelines import SectionGuidelinesLoader
from utils.llm_utils import remove_thinking_blocks
from settings import Settings
import lmstudio as lms


class PaperWriter:
    """Generates research paper sections using a LLM."""
    
    def __init__(self):
        pass

    @staticmethod
    def _format_evidence_for_prompt(evidence: Sequence[Evidence]) -> str:
        if not evidence:
            return ""

        items = []
        for item in evidence:
            citation_key = item.chunk.paper.citation_key or "unknown"
            title = item.chunk.paper.title or "Untitled"
            summary = item.summary or "No summary provided."

            item_lines = [
                f"[{citation_key}]",
                f"Title: {title}",
                summary,
            ]
            items.append("\n".join(item_lines))

        return "\n\n".join(items)

    @staticmethod
    def _format_plots_for_prompt(plots: list[Plot]) -> str:
        """Format plots as figure references for Results section."""
        if not plots:
            return ""
        
        lines = []
        for plot in plots:
            # Convert full path to relative path from output/ directory
            # e.g., "output/experiments/plots/file.png" -> "experiments/plots/file.png"
            filename = plot.filename
            if filename.startswith("output/"):
                filename = filename[len("output/"):]
            elif "/" in filename and not filename.startswith("experiments/"):
                # If it's a full path, extract relative part
                if "experiments" in filename:
                    filename = filename[filename.find("experiments"):]
            
            lines.append(f"Figure:")
            lines.append(f"  Filename: {filename}")
            lines.append(f"  Caption: {plot.caption}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_previous_sections(
        section_type: Section,
        previous_sections: dict[Section, str],
    ) -> str:
        """Format relevant previous sections as context for the current section."""
        
        # Define which previous sections each section should see
        section_dependencies = {
            Section.RESULTS: [Section.METHODS],
            Section.DISCUSSION: [Section.METHODS, Section.RESULTS],
            Section.RELATED_WORK: [Section.METHODS, Section.RESULTS, Section.DISCUSSION],
            Section.INTRODUCTION: [Section.RELATED_WORK, Section.METHODS, Section.RESULTS, Section.DISCUSSION],
            Section.CONCLUSION: [Section.METHODS, Section.RESULTS, Section.DISCUSSION],
            Section.ABSTRACT: [Section.INTRODUCTION, Section.RELATED_WORK, Section.METHODS, Section.RESULTS, Section.DISCUSSION, Section.CONCLUSION],
        }
        
        relevant_sections = section_dependencies.get(section_type, [])
        if not relevant_sections or not previous_sections:
            return ""
        
        parts = []
        for prev_section in relevant_sections:
            if prev_section in previous_sections:
                section_text = previous_sections[prev_section]
                parts.append(f"# {prev_section.value}\n{section_text}")
        
        if not parts:
            return ""
        
        return "\n\n".join(parts)

    @staticmethod
    def _format_context(
        context: ResearchContext,
        experiment: Optional[ExperimentResult],
    ) -> str:
        """Format context and experiment data for prompts."""
        
        def format_if_present(label: str, value: str) -> Optional[str]:
            return f"[{label.upper()}]\n{value.strip()}" if isinstance(value, str) and value.strip() else None
        
        sections = [
            format_if_present("Context description", context.description),
        ]
        
        if experiment:
            # Wrap code in markdown block
            code_block = f"```python\n{experiment.experiment_code}\n```" if experiment.experiment_code else ""
            
            sections.extend([
                format_if_present("Hypothesis", experiment.hypothesis.description),
                format_if_present("Success criteria", experiment.hypothesis.success_criteria),
                format_if_present("Experiment code", code_block),
                format_if_present("Key execution output", experiment.execution_result.stdout),
                format_if_present("Verdict", experiment.hypothesis_evaluation.verdict),
                format_if_present("Verdict reasoning", experiment.hypothesis_evaluation.reasoning),
            ])
        
        return "\n\n".join(s for s in sections if s)

    def get_section_guidelines(
        self,
        section_type: Section,
        experiment: Optional[ExperimentResult] = None,
    ) -> str:
        """
        Specifies style guidelines for each paper section.
        These guidelines are combined with more context and evidence in _build_section_prompt().
        """
        return SectionGuidelinesLoader.get_guidelines(section_type, experiment)

    def generate_title(
        self,
        draft: PaperDraft,
        context: ResearchContext,
        temperature: float = 0.1,
        max_tokens: int = 250,
    ) -> str:
        """Generate a paper title based on the complete paper draft."""

        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Create a concise, informative paper title based on the complete paper draft.

            [REQUIREMENTS]
            - Be punchy, clear, and highly concise
            - Focus on the core methodologies and the specific data or problem
            - Use standard academic title formatting (title case)
            - Avoid unnecessary words like 'A Study of', 'An Investigation into', or 'Approach'
            - Create an engaging title that highlights the specific techniques used (e.g., 'Method A, Method B, and Concept C in Data D')
            - ONLY output the title text, without quotes, additional text or formatting

            [PAPER DRAFT]
            Abstract: {draft.abstract}

            Introduction: {draft.introduction}

            Methods: {draft.methods}

            Results: {draft.results}

            Discussion: {draft.discussion}

            Conclusion: {draft.conclusion}

            Now generate only the title text.
            """)

        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        response = model.respond(
            prompt,
            config={
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        )
        title = remove_thinking_blocks(response.content).strip().strip('"').strip("'")
        return title

    def generate_acknowledgements(self, user_acknowledgements: str, temperature: float = 0.2) -> str:
        """Generate acknowledgements section by formatting/polishing user-provided text."""
        
        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_acknowledgements_prompt(user_acknowledgements)
        response = model.respond(
            prompt,
            config={
                "temperature": temperature,
            },
        )
        return remove_thinking_blocks(response.content).strip()

    def _build_acknowledgements_prompt(self, user_acknowledgements: str) -> str:
        """Create the prompt for generating acknowledgements section."""
        
        guidelines = self.get_section_guidelines(Section.ACKNOWLEDGEMENTS)
        
        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Format and polish the provided acknowledgements text into a professional academic acknowledgements section.

            [USER PROVIDED ACKNOWLEDGEMENTS]
            {user_acknowledgements}

            [STYLE GUIDELINES]
            {guidelines}

            [WRITING REQUIREMENTS]
            - Preserve the original meaning and intent of the user's text
            - Ensure proper grammar, flow, and academic tone
            - Keep it concise and appropriate for an academic paper
            - Do NOT add citations or references
            - Do NOT include section headings (e.g., "## Acknowledgements")
            - Output ONLY the polished acknowledgements text

            [GENERATION RULES]
            - Do NOT reference the guidelines or instructions
            - Output ONLY the final acknowledgements content without any markdown headings
            """)
        
        return prompt


    def generate_initial_section(
        self,
        section_type: Section,
        papers: Sequence[Paper],
        context: ResearchContext,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        paper_specification: Optional[PaperSpecification] = None,
        temperature: float = 0.1,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Generate initial section draft using paper catalog (not chunked evidence)."""
        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_initial_section_prompt(
            section_type,
            papers,
            context,
            experiment,
            previous_sections,
            paper_specification,
            next_section_type,
        )

        response = model.respond(
            prompt,
            config={"temperature": temperature},
        )
        return remove_thinking_blocks(response.content)

    def rewrite_section(
        self,
        section_type: Section,
        text_to_rewrite: str,
        critique: SectionCritique,
        new_evidence: Sequence[Evidence],
        papers: Sequence[Paper],
        context: ResearchContext,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        paper_specification: Optional[PaperSpecification] = None,
        temperature: float = 0.2,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Rewrite a section using the critique feedback and new evidence."""
        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_rewrite_prompt(
            section_type,
            text_to_rewrite,
            critique,
            new_evidence,
            papers,
            context,
            experiment,
            previous_sections,
            paper_specification,
            temperature,
            next_section_type,
        )
        response = model.respond(
            prompt,
            config={"temperature": temperature},
        )
        return remove_thinking_blocks(response.content)

    def _build_initial_section_prompt(
        self,
        section_type: Section,
        papers: Sequence[Paper],
        context: ResearchContext,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        paper_specification: Optional[PaperSpecification] = None,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Build prompt for initial section draft using paper catalog."""

        guidelines = self.get_section_guidelines(section_type, experiment)
        context_block = self._format_context(context, experiment)
        paper_catalog = self._format_paper_catalog(papers)
        previous_sections_block = self._format_previous_sections(section_type, previous_sections or {})
        paper_structure_block = self._format_paper_structure(section_type)

        # Get section-specific paper specification if available
        paper_spec_block = self._get_paper_spec_block(section_type, paper_specification)

        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"

        # Paper structure and forward look
        structure_block = ""
        if paper_structure_block:
            structure_block = f"[PAPER STRUCTURE]\n{paper_structure_block}\n"
            if next_section_type:
                structure_block += (
                    f"\nYou are writing the {section_type.value} section. "
                    f"The NEXT section will be: {next_section_type.value}. "
                    f"Wrap up the current section appropriately, but STOP before you discuss topics reserved for the {next_section_type.value} section. "
                    "Transitions are fine, but do not steal the content of the next section.\n"
                )
        elif next_section_type:
            structure_block = (
                f"[FORWARD LOOK]\n"
                f"You are writing the {section_type.value} section. "
                f"The NEXT section will be: {next_section_type.value}. "
                f"Wrap up the current section appropriately, but STOP before you discuss topics reserved for the {next_section_type.value} section. "
                "Transitions are fine, but do not steal the content of the next section.\n"
            )

        return f"""\
[ROLE]
You are an expert academic writer.

[TASK]
Write the complete {section_type.value} section of the paper based on the provided context and available papers.

[SECTION TYPE]
{section_type.value}

{paper_spec_block}

{title_section}[RESEARCH CONTEXT]
{context_block}

[PREVIOUS SECTIONS]
{previous_sections_block if previous_sections_block else 'None yet.'}

[AVAILABLE PAPERS]
The following papers are available for citation. Use their citation keys in square brackets (e.g. [HintonRL2016]).

{paper_catalog}

[STYLE GUIDELINES]
{guidelines}

{structure_block}

[WRITING REQUIREMENTS — STRICT]
- Produce a cohesive, original, publication-quality academic narrative.
- PAPER SPECIFICATION: You MUST strictly follow the scope and length constraints in [PAPER SPECIFICATION]. Try not to exceed the targeted length.
- CITATION FORMAT: Use square brackets with the EXACT citation keys provided (e.g., [AuthorYear]).
- CRITICAL: Copy citation keys EXACTLY. Do NOT shorten or modify them.
- CRITICAL: NEVER use numeric citations like [1], [2]. These are strictly forbidden.
- Place citations immediately before final punctuation: "[exactKey]."
- For multiple sources: "[key1, key2]."
- Never fabricate evidence, results, or citations.
- Integrate and build upon previous sections to ensure full narrative coherence.
- STRICTLY FORBIDDEN: Do NOT cite papers that are not in the [AVAILABLE PAPERS] list, even if they are seminal works.
- STRICTLY FORBIDDEN: Do NOT generate a bibliography or references section at the end.
- MATHEMATICAL NOTATION: Use LaTeX-compatible notation for all formulas and symbols.
  - Greek letters: Write as *\\alpha*, *\\beta*, *\\gamma*, etc. (NOT Unicode symbols)
  - Formulas: Wrap in single asterisks for inline math: *x = \\alpha + \\beta*
  - Subscripts/superscripts: Use LaTeX syntax: *x_i*, *x^2*, *Q_{{max}}*

[GENERATION RULES — DO NOT VIOLATE]
- Do NOT reference the guidelines or instructions.
- STRICTLY FORBIDDEN: Do NOT start your text with ANY section heading — not "# Methods", not "## Results", not "# Introduction", just nothing! Your first output character must be prose content, NOT a "#" symbol. The heading already exists above your output.
- STRICTLY FORBIDDEN: Do NOT attempt to start or write the next section after finishing this one. Stop writing immediately upon completing the current section.
- You MAY use markdown subsection headings (### and ####) to organize longer sections where sub-topics benefit from labeling.
- SUBSECTION RULES: If you use subsections, you MUST use at least two of the same level. NEVER use just a single subsection.
- SUBSECTION RULES: All subsections MUST be consistently numbered (e.g., "### 1. First Topic", "### 2. Second Topic"). Do NOT mix numbered and non-numbered subsections.
- Output ONLY the final written section content.
"""

    def _build_rewrite_prompt(
        self,
        section_type: Section,
        initial_section: str,
        critique: SectionCritique,
        new_evidence: Sequence[Evidence],
        papers: Sequence[Paper],
        context: ResearchContext,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        paper_specification: Optional[PaperSpecification] = None,
        temperature: float = 0.1,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Build prompt for rewriting a section with critique and new evidence."""

        guidelines = self.get_section_guidelines(section_type, experiment)
        context_block = self._format_context(context, experiment)
        paper_catalog = self._format_paper_catalog(papers)
        evidence_block = self._format_evidence_for_prompt(new_evidence)
        previous_sections_block = self._format_previous_sections(section_type, previous_sections or {})
        paper_spec_block = self._get_paper_spec_block(section_type, paper_specification)
        paper_structure_block = self._format_paper_structure(section_type)

        improvements_text = critique.improvements

        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"

        # Paper structure and forward look
        structure_block = ""
        if paper_structure_block:
            structure_block = f"[PAPER STRUCTURE]\n{paper_structure_block}\n"
            if next_section_type:
                structure_block += (
                    f"\nYou are writing the {section_type.value} section. "
                    f"The NEXT section will be: {next_section_type.value}. "
                    f"Wrap up the current section appropriately, but STOP before you discuss topics reserved for the {next_section_type.value} section. "
                    "Transitions are fine, but do not steal the content of the next section.\n"
                )
        elif next_section_type:
            structure_block = (
                f"[FORWARD LOOK]\n"
                f"You are writing the {section_type.value} section. "
                f"The NEXT section will be: {next_section_type.value}. "
                f"Wrap up the current section appropriately, but STOP before you discuss topics reserved for the {next_section_type.value} section. "
                "Transitions are fine, but do not steal the content of the next section.\n"
            )

        return f"""\
[ROLE]
You are an expert academic writer revising a section based on feedback.

[TASK]
Rewrite the original {section_type.value} section, addressing the suggested improvements and incorporating new evidence.

[SECTION TYPE]
{section_type.value}

{paper_spec_block}

[ORIGINAL SECTION]
{initial_section}

[IMPROVEMENTS TO MAKE]
{improvements_text}

[NEW EVIDENCE]
{self._format_new_evidence_block(evidence_block)}

{title_section}[RESEARCH CONTEXT]
{context_block}

[PREVIOUS SECTIONS]
{previous_sections_block if previous_sections_block else 'None available.'}

[AVAILABLE PAPERS]
{paper_catalog}

[STYLE GUIDELINES]
{guidelines}

{structure_block}

[WRITING REQUIREMENTS — STRICT]
- Address ALL suggested improvements.
- Incorporate the new evidence naturally with proper citations.
- CITATION FORMAT: Use square brackets with the EXACT citation keys provided.
- CRITICAL: Copy citation keys EXACTLY. Do NOT shorten or modify them.
- CRITICAL: NEVER use numeric citations like [1], [2]. These are strictly forbidden.
- Maintain the strengths of the original draft.
- Produce a cohesive, publication-quality narrative.
- STRICTLY FORBIDDEN: Do NOT cite papers that are not in the [NEW EVIDENCE] or [AVAILABLE PAPERS] lists, even if they are seminal works.
- STRICTLY FORBIDDEN: Do NOT generate a bibliography or references section at the end.
- MATHEMATICAL NOTATION: Use LaTeX-compatible notation for all formulas and symbols.
  - Greek letters: Write as *\\alpha*, *\\beta*, *\\gamma*, etc. (NOT Unicode symbols)
  - Formulas: Wrap in single asterisks for inline math: *x = \\alpha + \\beta*
  - Subscripts/superscripts: Use LaTeX syntax: *x_i*, *x^2*, *Q_{{max}}*

[GENERATION RULES — DO NOT VIOLATE]
- Do NOT reference the critique or instructions.
- You MUST output the ENTIRE rewritten section from start to finish, not just the parts you changed.
- STRICTLY FORBIDDEN: Do NOT start your text with ANY section heading — not "# Methods", not "## Results", not "# Introduction", nothing. Your first output character must be prose content, not a "#" symbol. The heading already exists above your output.
- STRICTLY FORBIDDEN: Do NOT attempt to start or write the next section after finishing this one. Stop writing immediately upon completing the current section.
- You MAY use markdown subsection headings (### and ####) to organize longer sections where sub-topics benefit from labeling.
- SUBSECTION RULES: If you use subsections, you MUST use at least two of the same level. NEVER use just a single subsection.
- SUBSECTION RULES: All subsections MUST be consistently numbered (e.g., "### 1. First Topic", "### 2. Second Topic"). Do NOT mix numbered and non-numbered subsections.
- Output ONLY the final rewritten section content.
"""

    @staticmethod
    def _format_paper_structure(current_section: Section) -> str:
        """Format the full paper structure in order and mark the current section."""
        order = [
            Section.ABSTRACT,
            Section.INTRODUCTION,
            Section.RELATED_WORK,
            Section.METHODS,
            Section.RESULTS,
            Section.DISCUSSION,
            Section.CONCLUSION,
        ]

        lines = []
        for section in order:
            if section == current_section:
                lines.append(f"  → {section.value}  ← (CURRENT)")
            else:
                lines.append(f"  - {section.value}")
        return "\n".join(lines)

    @staticmethod
    def _format_new_evidence_block(evidence_block: str) -> str:
        """Format the new evidence block for the rewrite prompt."""
        if not evidence_block:
            return "No additional evidence retrieved."
        return (
            "The following evidence was retrieved based on the critic's analysis of your initial draft. "
            "The critic identified gaps or areas that could benefit from additional supporting material, "
            "and these passages were found in response to those needs. "
            "Incorporate them where they strengthen the narrative.\n\n"
            + evidence_block
        )

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
        """Format papers as a catalog for prompts."""
        if not papers:
            return "No papers available."

        items = []
        for paper in papers:
            citation_key = paper.citation_key or "unknown"
            abstract = PaperWriter._normalize_text(paper.summary or "No abstract available.")
            conclusion = PaperWriter._normalize_text(paper.conclusion or "")

            # Truncate long abstracts
            abstract_truncated = abstract[:1000] + "..." if len(abstract) > 1000 else abstract

            lines = [
                f"[{citation_key}]",
                f"Title: {paper.title}",
                f"Abstract: {abstract_truncated}",
            ]

            if conclusion:
                # Truncate long conclusions
                conclusion_truncated = conclusion[:1000] + "..." if len(conclusion) > 1000 else conclusion
                lines.append(f"Conclusion: {conclusion_truncated}")

            items.append("\n".join(lines))

        return "\n\n".join(items)

    def _get_paper_spec_block(
        self,
        section_type: Section,
        paper_specification: Optional[PaperSpecification],
    ) -> str:
        """Get section-specific paper specification as a formatted block."""
        if not paper_specification:
            return ""
        
        section_to_requirement = {
            Section.ABSTRACT: "abstract",
            Section.INTRODUCTION: "introduction",
            Section.RELATED_WORK: "related_work",
            Section.METHODS: "methods",
            Section.RESULTS: "results",
            Section.DISCUSSION: "discussion",
            Section.CONCLUSION: "conclusion",
        }
        
        requirement_field = section_to_requirement.get(section_type)
        if requirement_field:
            requirement_text = getattr(paper_specification, requirement_field, None)
            if requirement_text and requirement_text.strip():
                return f"[PAPER SPECIFICATION]\n{requirement_text.strip()}"
        
        return ""
