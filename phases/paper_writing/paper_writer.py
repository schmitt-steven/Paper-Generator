import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

from phases.context_analysis.paper_conception import PaperConcept
from phases.context_analysis.user_requirements import UserRequirements
from phases.experimentation.experiment_state import ExperimentResult, Plot
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import Evidence, PaperDraft, Section, SectionCritique
from phases.paper_writing.section_guidelines import SectionGuidelinesLoader
from utils.llm_utils import remove_thinking_blocks
from settings import Settings
import lmstudio as lms


class PaperWriter:
    """Generates research paper sections using a LLM."""
    
    def __init__(self):
        pass
    
    def generate_paper_sections(
        self,
        context: PaperConcept,
        experiment: ExperimentResult,
        evidence_by_section: dict[Section, Sequence[Evidence]],

        user_requirements: Optional[UserRequirements] = None,
        writing_prompts: Optional[dict[str, str]] = None,
    ) -> tuple[PaperDraft, dict[str, str]]:
        """Generate all paper sections using provided evidence. Returns (draft, prompts_by_section)."""

        section_order = (
            Section.METHODS, Section.RESULTS, Section.DISCUSSION,
            Section.INTRODUCTION, Section.RELATED_WORK, Section.CONCLUSION, Section.ABSTRACT
        )
        
        sections = {}
        prompts_by_section = {}
        for section_type in section_order:
            print(f"Writing {section_type.value} section...")
            prompt = self._build_section_prompt(
                section_type=section_type,
                context=context,
                experiment=experiment,
                evidence=evidence_by_section.get(section_type, []),
                previous_sections=sections,
                user_requirements=user_requirements,
            )
            
            # Use existing prompt if available (override the built one if needed, or just use it)
            # Actually, if we have a pre-loaded prompt, we should probably use THAT one for generation.
            # But we also want to return it.
            if writing_prompts and section_type.value in writing_prompts:
                 prompt = writing_prompts[section_type.value]
            
            prompts_by_section[section_type.value] = prompt
            sections[section_type] = self.generate_section(
                section_type=section_type,
                context=context,
                experiment=experiment,
                evidence=evidence_by_section.get(section_type, []),
                previous_sections=sections,
                user_requirements=user_requirements,
                existing_prompt=prompt,
            )

        # Generate acknowledgements if enabled and user provided content
        acknowledgements = None
        if Settings.GENERATE_ACKNOWLEDGEMENTS and user_requirements and user_requirements.acknowledgements:
            print("Writing Acknowledgements section...")
            acknowledgements = self.generate_acknowledgements(user_requirements.acknowledgements)
            prompts_by_section[Section.ACKNOWLEDGEMENTS.value] = self._build_acknowledgements_prompt(user_requirements.acknowledgements)

        # Create draft (title will be set below)
        draft = PaperDraft(
            title="",
            abstract=sections[Section.ABSTRACT],
            introduction=sections[Section.INTRODUCTION],
            related_work=sections[Section.RELATED_WORK],
            methods=sections[Section.METHODS],
            results=sections[Section.RESULTS],
            discussion=sections[Section.DISCUSSION],
            conclusion=sections[Section.CONCLUSION],
            acknowledgements=acknowledgements,
        )

        # Use settings title if provided, otherwise generate one
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            draft.title = Settings.LATEX_TITLE
        else:
            draft.title = self.generate_title(draft=draft, context=context)
        return draft, prompts_by_section

    def generate_section(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        evidence: Sequence[Evidence],

        previous_sections: Optional[dict[Section, str]] = None,
        temperature: float = 0.2,
        user_requirements: Optional[UserRequirements] = None,
        existing_prompt: Optional[str] = None,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Generate a single section given context and evidence."""

        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        if existing_prompt:
             prompt = existing_prompt
        else:
             prompt = self._build_section_prompt(
                 section_type, context, experiment, evidence, previous_sections, user_requirements, next_section_type
             )
        response = model.respond(
            prompt,
            config={
                "temperature": temperature,
            },
        )
        return remove_thinking_blocks(response.content)

    def _build_section_prompt(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        evidence: Sequence[Evidence],
        previous_sections: Optional[dict[Section, str]] = None,
        user_requirements: Optional[UserRequirements] = None,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Create the generation prompt for a specific section."""

        guidelines = self.get_section_guidelines(section_type, experiment)
        context_block = self._format_context(context, experiment)
        evidence_block = self._format_evidence_for_prompt(evidence)
        previous_sections_block = self._format_previous_sections(section_type, previous_sections or {})
        
        # Map Section enum to UserRequirements field
        section_to_requirement = {
            Section.ABSTRACT: "abstract",
            Section.INTRODUCTION: "introduction",
            Section.RELATED_WORK: "related_work",
            Section.METHODS: "methods",
            Section.RESULTS: "results",
            Section.DISCUSSION: "discussion",
            Section.CONCLUSION: "conclusion",
        }
        
        # Get section-specific user requirements if available
        user_requirements_block = ""
        if user_requirements:
            requirement_field = section_to_requirement.get(section_type)
            if requirement_field:
                requirement_text = getattr(user_requirements, requirement_field, None)
                if requirement_text and requirement_text.strip():
                    user_requirements_block = f"""[USER REQUIREMENTS]\n{requirement_text.strip()}"""
        
        # Add plots block for Results section
        plots_block = ""
        if section_type == Section.RESULTS and experiment and experiment.plots:
            plots_block = f"""[FIGURES TO INCLUDE]
                You MUST include ALL of the following figures in your Results section using markdown image syntax.

                {self._format_plots_for_prompt(experiment.plots)}

                FIGURE REQUIREMENTS:
                - **INTERLEAVED PLACEMENT**: You MUST place the figure markdown immediately after the paragraph where it is discussed. DO NOT dump all figures at the end.
                - **DYNAMIC NUMBERING**: You MUST assign Figure numbers (Figure 1, Figure 2...) sequentially based on the order you introduce them in the text.
                - **START AT 1**: The first figure you discuss MUST be "Figure 1", the second "Figure 2", and so on.
                - **ALL INCLUDED**: You MUST include ALL available plots listed above.
                - **SYNTAX**: Use markdown: ![Brief description](experiments/plots/file_name.png)
                - **CAPTION**: Add a caption line below each figure: *Figure N: Full caption text*"""
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n            {Settings.LATEX_TITLE}\n\n            "
        
        # Forward Look Instruction
        forward_look_block = ""
        if next_section_type:
            forward_look_block = f"""
            [FORWARD LOOK]
            You are writing the {section_type.value} section.
            The NEXT section will be: {next_section_type.value}.
            INSTRUCTION: wrap up the current section appropriately, but STOP before you discuss the topics reserved for the {next_section_type.value} section.
            Transitions are fine, but do not steal the content of the next section.
            """
        
        # Special handling for abstract: No citations allowed
        if section_type == Section.ABSTRACT:
            prompt = f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Write the complete Abstract section of the paper based on the provided context and previous sections.

            [SECTION TYPE]
            Abstract

            {title_section}[RESEARCH CONTEXT]
            {context_block}

            [PREVIOUS SECTIONS]
            {previous_sections_block if previous_sections_block else 'None'}

            [SECTION GUIDELINES]
            {guidelines}

            {user_requirements_block}

            [WRITING REQUIREMENTS — STRICT]
            - Write a publication-quality abstract that summarizes the key contributions.
            - CRITICAL: DO NOT INCLUDE ANY CITATIONS OR REFERENCES.
            - DO NOT use square brackets [] anywhere in the text.
            - DO NOT reference other papers or authors by name.
            - The abstract must be self-contained without external references.
            - Never fabricate evidence or results.
            - Draw from the previous sections to write an accurate summary.
            - MATHEMATICAL NOTATION: Use LaTeX-compatible notation for all formulas and symbols.
            - Greek letters: Write as *\\alpha*, *\\beta*, *\\gamma*, etc. (NOT Unicode symbols like α, β, γ)
            - Formulas: Wrap in single asterisks for inline math: *x = \\alpha + \\beta*
            - Subscripts/superscripts: Use LaTeX syntax: *x_i*, *x^2*, *Q_{max}*

            [GENERATION RULES — DO NOT VIOLATE]
            - Do NOT include ANY citations like [AuthorYear], [1], or any bracketed references.
            - Do NOT reference the guidelines or instructions.
            - Do NOT include section headings (e.g., "## Abstract") in your output.
            - Output ONLY the final abstract content without any markdown headings.

            Your output must be a polished academic abstract with ZERO citations."""
        else:
            prompt = f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Write the complete {section_type.value} section of the paper based on the provided context.

            [SECTION TYPE]
            {section_type.value}

            {title_section}[RESEARCH CONTEXT]
            {context_block}

            [PREVIOUS SECTIONS]
            {previous_sections_block if previous_sections_block else ''}

            [EVIDENCE]
            {evidence_block if evidence_block else 'No evidence available.'}

            [SECTION GUIDELINES]
            {guidelines}

            {user_requirements_block}

            {user_requirements_block}
            {forward_look_block}
            {plots_block}

            [WRITING REQUIREMENTS]
            - Produce a cohesive, original, publication-quality academic narrative.
            - CITATION FORMAT: Use square brackets with the EXACT, COMPLETE citation keys provided in the <citation_key> tags in the evidence section.
            - Copy the citation keys EXACTLY as they appear in <citation_key> tags. Do NOT shorten them, do NOT change them, do NOT generate simplified versions.
            - NEVER use numeric citations like [1], [2], [30]. These are strictly forbidden.
            - Do NOT invent citation keys. Do NOT generate "nameYear" format. Use ONLY the exact keys found in the <citation_key> tags.
            - Example: If evidence shows <citation_key>Hoppe2019QgraphboundedQS</citation_key>, use [Hoppe2019QgraphboundedQS] exactly, NOT [Hoppe2019].
            - Place citations immediately before final punctuation: "[exactKeyFromEvidence]."
            - For multiple sources: "[exactKey1, exactKey2]."
            - If a source in the evidence has "unknown" or "n.d." as a key, do NOT cite it.
            - Cite external papers ONLY using the exact citation keys from the evidence in square brackets.
            - Never fabricate evidence, results, or citations.
            - Integrate and build upon previous sections to ensure full narrative coherence.
            - Do NOT cite papers that are not in the [EVIDENCE] list, even if they are seminal works (e.g. by Sutton, Pearl, Bellman). If you must discuss them, do so without generating a citation key.
            - Do NOT generate a bibliography or references section at the end.
            - MATHEMATICAL NOTATION: Use LaTeX-compatible notation for all formulas and symbols.
            - Greek letters: Write as *\\alpha*, *\\beta*, *\\gamma*, etc. (NOT Unicode symbols like α, β, γ)
            - Formulas: Wrap in single asterisks for inline math: *x = \\alpha + \\beta*
            - Subscripts/superscripts: Use LaTeX syntax: *x_i*, *x^2*, *Q_{max}*
            - Common symbols: *\\leq*, *\\geq*, *\\neq*, *\\approx*, *\\infty*, *\\sum*, *\\prod*

            [GENERATION RULES — DO NOT VIOLATE]
            - Do NOT reference the guidelines or instructions.
            - Do NOT comment on the evidence structure.
            - Do NOT include section headings (e.g., "## Introduction", "# Abstract", etc.) in your output.
            - Output ONLY the final written section content without any markdown headings.

            Your output must strictly follow the requirements and produce a polished academic section.
            """
        return prompt

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
                "<item>",
                f"  <citation_key>{citation_key}</citation_key>",
                f"  <title>{title}</title>",
                f"  <summary>{summary}</summary>",
                "</item>"
            ]
            items.append("\n".join(item_lines))

        # Indent the joined items by two spaces for the <evidence> block
        indented_items = textwrap.indent("\n".join(items), "  ")
        return f"<evidence>\n{indented_items}\n</evidence>"

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
            Section.DISCUSSION: [Section.RESULTS, Section.METHODS],
            Section.CONCLUSION: [Section.METHODS,Section.RESULTS, Section.DISCUSSION],
            Section.ABSTRACT: [
                Section.METHODS,
                Section.RESULTS,
                Section.DISCUSSION,
                Section.INTRODUCTION,
                Section.RELATED_WORK,
                Section.CONCLUSION,
            ],
        }
        
        relevant_sections = section_dependencies.get(section_type, [])
        if not relevant_sections or not previous_sections:
            return ""
        
        parts = []
        for prev_section in relevant_sections:
            if prev_section in previous_sections:
                section_text = previous_sections[prev_section]
                parts.append(f"{prev_section.value}:\n{section_text}")
        
        if not parts:
            return ""
        
        return "\n\n".join(parts)

    @staticmethod
    def _format_context(
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> str:
        """Format context and experiment data for prompts."""
        
        def format_if_present(label: str, value: str) -> Optional[str]:
            return f"[{label.upper()}]\n{value.strip()}" if isinstance(value, str) and value.strip() else None
        
        sections = [
            format_if_present("Concept description", context.description),
            format_if_present("Open questions", context.open_questions),
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
        Specifies writing guidelines for each paper section.
        These guidelines are combined with more context and evidence in _build_section_prompt().
        """
        return SectionGuidelinesLoader.get_guidelines(section_type, experiment)

    def generate_title(
        self,
        draft: PaperDraft,
        context: PaperConcept,
        temperature: float = 0.2,
        max_tokens: int = 200,
    ) -> str:
        """Generate a paper title based on the complete paper draft."""

        prompt = f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Create a concise, informative paper title based on the complete paper draft.

            [REQUIREMENTS]
            - Be clear, concise and descriptive
            - Use standard academic title formatting (title case)
            - Avoid unnecessary words like 'A Study of' or 'An Investigation into'
            - ONLY output the title text, without quotes, additional text or formatting

            [PAPER DRAFT]
            Abstract: {draft.abstract}
            
            Introduction: {draft.introduction}
            
            Methods: {draft.methods}
            
            Results: {draft.results}
            
            Discussion: {draft.discussion}
            
            Conclusion: {draft.conclusion}

            Now generate only the title text.
            """

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
        
        prompt = f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Format and polish the provided acknowledgements text into a professional academic acknowledgements section.

            [USER PROVIDED ACKNOWLEDGEMENTS]
            {user_acknowledgements}

            [SECTION GUIDELINES]
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
            """
        
        return prompt


    def generate_initial_section(
        self,
        section_type: Section,
        papers: Sequence[Paper],
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        user_requirements: Optional[UserRequirements] = None,
        temperature: float = 0.2,
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
            user_requirements,
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
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        user_requirements: Optional[UserRequirements] = None,
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
            user_requirements,
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
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        user_requirements: Optional[UserRequirements] = None,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Build prompt for initial section draft using paper catalog."""
        
        guidelines = self.get_section_guidelines(section_type, experiment)
        context_block = self._format_context(context, experiment)
        paper_catalog = self._format_paper_catalog(papers)
        previous_sections_block = self._format_previous_sections(section_type, previous_sections or {})
        
        # Get section-specific user requirements if available
        user_requirements_block = self._get_user_requirements_block(section_type, user_requirements)
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"
        
        # Forward Look Instruction
        forward_look_block = ""
        if next_section_type:
            forward_look_block = f"""
            [FORWARD LOOK]
            You are writing the {section_type.value} section.
            The NEXT section will be: {next_section_type.value}.
            INSTRUCTION: wrap up the current section appropriately, but STOP before you discuss the topics reserved for the {next_section_type.value} section.
            Transitions are fine, but do not steal the content of the next section.
            """
        
        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Write the complete {section_type.value} section of the paper based on the provided context and available papers.

            [SECTION TYPE]
            {section_type.value}

            {title_section}[RESEARCH CONTEXT]
            {context_block}

            [PREVIOUS SECTIONS]
            {previous_sections_block if previous_sections_block else 'None yet.'}

            [AVAILABLE PAPERS]
            The following papers are available for citation. Use their citation keys in square brackets (e.g. [HintonRL2016]).
            {paper_catalog}

            [SECTION GUIDELINES]
            {guidelines}

            {user_requirements_block}
            {user_requirements_block}
            {forward_look_block}

            [WRITING REQUIREMENTS — STRICT]
            - Produce a cohesive, original, publication-quality academic narrative.
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
              - Subscripts/superscripts: Use LaTeX syntax: *x_i*, *x^2*, *Q_{max}*

            [GENERATION RULES — DO NOT VIOLATE]
            - Do NOT reference the guidelines or instructions.
            - Do NOT include section headings (e.g., "## Introduction") in your output.
            - Output ONLY the final written section content.
        """)

    def _build_rewrite_prompt(
        self,
        section_type: Section,
        initial_section: str,
        critique: SectionCritique,
        new_evidence: Sequence[Evidence],
        papers: Sequence[Paper],
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        previous_sections: Optional[dict[Section, str]] = None,
        user_requirements: Optional[UserRequirements] = None,
        temperature: float = 0.2,
        next_section_type: Optional[Section] = None,
    ) -> str:
        """Build prompt for rewriting a section with critique and new evidence."""
        
        guidelines = self.get_section_guidelines(section_type, experiment)
        context_block = self._format_context(context, experiment)
        paper_catalog = self._format_paper_catalog(papers)
        evidence_block = self._format_evidence_for_prompt(new_evidence)
        previous_sections_block = self._format_previous_sections(section_type, previous_sections or {})
        user_requirements_block = self._get_user_requirements_block(section_type, user_requirements)
        
        improvements_text = critique.improvements
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"
        
        # Forward Look Instruction
        forward_look_block = ""
        if next_section_type:
            forward_look_block = f"""
            [FORWARD LOOK]
            You are writing the {section_type.value} section.
            The NEXT section will be: {next_section_type.value}.
            INSTRUCTION: wrap up the current section appropriately, but STOP before you discuss the topics reserved for the {next_section_type.value} section.
            Transitions are fine, but do not steal the content of the next section.
            """
        
        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer revising a section based on feedback.

            [TASK]
            Rewrite the {section_type.value} section, addressing the suggested improvements and incorporating new evidence.

            [SECTION TYPE]
            {section_type.value}

            [ORIGINAL SECTION]
            {initial_section}

            [IMPROVEMENTS TO MAKE]
            {improvements_text}

            [NEW EVIDENCE]
            {evidence_block if evidence_block else 'No additional evidence retrieved.'}

            {title_section}[RESEARCH CONTEXT]
            {context_block}

            [PREVIOUS SECTIONS]
            {previous_sections_block if previous_sections_block else 'None available.'}

            [AVAILABLE PAPERS]
            {paper_catalog}

            [SECTION GUIDELINES]
            {guidelines}

            {user_requirements_block}
            {forward_look_block}

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
              - Subscripts/superscripts: Use LaTeX syntax: *x_i*, *x^2*, *Q_{max}*

            [GENERATION RULES — DO NOT VIOLATE]
            - Do NOT reference the critique or instructions.
            - Do NOT include section headings in your output.
            - Output ONLY the final rewritten section content.
        """)

    @staticmethod
    def _format_paper_catalog(papers: Sequence[Paper]) -> str:
        """Format papers as a catalog for prompts."""
        if not papers:
            return "No papers available."
        
        items = []
        for paper in papers:
            citation_key = paper.citation_key or "unknown"
            abstract = paper.summary or "No abstract available."
            conclusion = paper.conclusion or ""
            
            # Truncate long abstracts
            abstract_truncated = abstract[:500] + "..." if len(abstract) > 500 else abstract
            
            entry = textwrap.dedent(f"""\
                [{citation_key}]
                Title: {paper.title}
                Abstract: {abstract_truncated}""")
            
            if conclusion:
                conclusion_truncated = conclusion[:500] + "..." if len(conclusion) > 500 else conclusion
                entry += f"\nConclusion: {conclusion_truncated}"
            
            items.append(entry)
        
        return "\n\n".join(items)

    def _get_user_requirements_block(
        self,
        section_type: Section,
        user_requirements: Optional[UserRequirements],
    ) -> str:
        """Get section-specific user requirements as a formatted block."""
        if not user_requirements:
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
            requirement_text = getattr(user_requirements, requirement_field, None)
            if requirement_text and requirement_text.strip():
                return f"[USER REQUIREMENTS]\n{requirement_text.strip()}"
        
        return ""
