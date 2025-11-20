import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult, Plot
from phases.paper_writing.data_models import Evidence, PaperDraft, Section
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
        evidence_by_section: Dict[Section, Sequence[Evidence]],
    ) -> Tuple[PaperDraft, Dict[str, str]]:
        """Generate all paper sections using provided evidence. Returns (draft, prompts_by_section)."""

        section_order = (
            Section.METHODS, Section.RESULTS, Section.DISCUSSION,
            Section.INTRODUCTION, Section.RELATED_WORK, Section.CONCLUSION, Section.ABSTRACT
        )
        
        sections = {}
        prompts_by_section = {}
        for section_type in section_order:
            prompt = self._build_section_prompt(
                section_type=section_type,
                context=context,
                experiment=experiment,
                evidence=evidence_by_section.get(section_type, []),
                previous_sections=sections,
            )
            prompts_by_section[section_type.value] = prompt
            sections[section_type] = self.generate_section(
                section_type=section_type,
                context=context,
                experiment=experiment,
                evidence=evidence_by_section.get(section_type, []),
                previous_sections=sections,
            )

        # Use settings title if provided, otherwise generate one
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title = Settings.LATEX_TITLE
        else:
            title = self.generate_title(
                abstract=sections[Section.ABSTRACT],
                introduction=sections[Section.INTRODUCTION],
                conclusion=sections[Section.CONCLUSION],
                context=context,
            )

        draft = PaperDraft(
            title=title,
            abstract=sections[Section.ABSTRACT],
            introduction=sections[Section.INTRODUCTION],
            related_work=sections[Section.RELATED_WORK],
            methods=sections[Section.METHODS],
            results=sections[Section.RESULTS],
            discussion=sections[Section.DISCUSSION],
            conclusion=sections[Section.CONCLUSION],
        )
        return draft, prompts_by_section

    def generate_section(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        evidence: Sequence[Evidence],
        previous_sections: Optional[Dict[Section, str]] = None,
        temperature: float = 0.5,
    ) -> str:
        """Generate a single section given context and evidence."""

        model = lms.llm(Settings.PAPER_WRITING_MODEL)
        prompt = self._build_section_prompt(section_type, context, experiment, evidence, previous_sections)
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
        previous_sections: Optional[Dict[Section, str]] = None,
    ) -> str:
        """Create the generation prompt for a specific section."""

        guidelines = self.get_section_guidelines(section_type, experiment)
        context_block = self._format_context(context, experiment)
        evidence_block = self._format_evidence_for_prompt(evidence)
        previous_sections_block = self._format_previous_sections(section_type, previous_sections or {})
        
        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer.

            [TASK]
            Write the complete {section_type.value} section of the paper based on the provided context.

            [SECTION TYPE]
            {section_type.value}

            [RESEARCH CONTEXT]
            {context_block}

            [PREVIOUS SECTIONS]
            {previous_sections_block if previous_sections_block else ''}

            [EVIDENCE]
            {evidence_block if evidence_block else 'No evidence available.'}

            [SECTION GUIDELINES]
            {guidelines}

           [WRITING REQUIREMENTS — STRICT]
            - Produce a cohesive, original, publication-quality academic narrative.
            - CITATION FORMAT: Use square brackets with the EXACT keys provided in the evidence section (e.g., [smith2024]).
            - CRITICAL: NEVER use numeric citations like [1], [2], [30]. These are strictly forbidden.
            - CRITICAL: Do NOT invent citation keys. Use ONLY the keys found in the <citation_key> tags in the evidence.
            - Place citations immediately before final punctuation: "[smith2024]."
            - For multiple sources: "[smith2024, jones2023]."
            - If a source in the evidence has "unknown" or "n.d." as a key, do NOT cite it.
            - Cite external papers ONLY using citation keys from the evidence in square brackets.
            - Never fabricate evidence, results, or citations.
            - Integrate and build upon previous sections to ensure full narrative coherence.

            [GENERATION RULES — DO NOT VIOLATE]
            - Do NOT reference the guidelines or instructions.
            - Do NOT comment on the evidence structure.
            - Do NOT include section headings (e.g., "## Introduction", "# Abstract", etc.) in your output.
            - Output ONLY the final written section content without any markdown headings.

            [FINAL PRIORITY]
            Your output must strictly follow the requirements and produce a polished academic section.
        """)
        
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
    def _format_plots_for_prompt(plots: List[Plot]) -> str:
        """Format plots as figure references for Results section."""
        if not plots:
            return ""
        
        lines = []
        for idx, plot in enumerate(plots, 1):
            lines.append(f"Figure {idx}:")
            lines.append(f"  Filename: {plot.filename}")
            lines.append(f"  Caption: {plot.caption}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_previous_sections(
        section_type: Section,
        previous_sections: Dict[Section, str],
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
            sections.extend([
                format_if_present("Hypothesis", experiment.hypothesis.description),
                format_if_present("Expected improvement", experiment.hypothesis.expected_improvement),
                format_if_present("Experimental plan", experiment.experimental_plan),
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
        
        section_guidelines = {
            Section.ABSTRACT: textwrap.dedent("""\
                150-250 words. Structure: (1) problem/gap, (2) approach, (3) key result with metrics, (4) main implication. 
                Be specific. NO citations."""),
            
            Section.INTRODUCTION: textwrap.dedent("""\
                Open with the problem and its concrete impact.
                Identify what's missing in current solutions using evidence.
                State your contribution as specific, falsifiable claims.
                End with brief paper roadmap.
                Justify claims with evidence, don't just assert."""),
            
            Section.RELATED_WORK: textwrap.dedent("""\
                Group by approach/theme, not chronologically. For each cluster:
                - What they did (method + reported results)
                - Limitations relative to this work
                - Direct comparison where applicable
                Avoid generic praise. Be precise about differences. Cite liberally."""),
            
            Section.METHODS: textwrap.dedent("""\
                Reproducibility is the goal. If possible and relevant, include:
                - Architecture/algorithm with justification for key choices
                - Hyperparameters, dataset details, compute resources
                - Baseline comparisons (what and why)
                - Evaluation metrics with rationale
                Use present tense. Avoid implementation details unless critical."""),
            
            Section.RESULTS: 
            self._get_results_guidelines(experiment),
            
            Section.DISCUSSION: textwrap.dedent("""\
                Open by restating main finding in context of hypothesis.
                Explain why it worked/failed using specific evidence and results. Acknowledge limitations honestly.
                Compare to related work quantitatively where possible.
                Speculation allowed but label it clearly.
                End with concrete future directions, not vague "explore further."""),
            
            Section.CONCLUSION: textwrap.dedent("""\
                Summarize: what you did, what you found (with key metrics), broader implications (realistic, not grandiose), one actionable next step.
                No new information. No citations."""),
        }

        return section_guidelines.get(section_type, "")

    def _get_results_guidelines(self, experiment: Optional[ExperimentResult]) -> str:
        """Get Results section guidelines, including figure integration if plots are available."""

        section_guidelines = """Present experimental outcomes with relevant metrics or observations.
        Compare results against expected improvements or baselines if available.
        Never fabricate data or results."""

        if experiment and experiment.plots:
            plots_block = self._format_plots_for_prompt(experiment.plots)
            
            section_guidelines += "\n\n" + textwrap.dedent(f"""
                [FIGURE INTEGRATION]
                The following figures were generated from the experiment. You MUST integrate all of them into your Results section.

                {plots_block}

                For each figure:
                1. Reference it naturally in the text (e.g., "As shown in Figure 1..." or "Figure 2 demonstrates...")
                2. Include the markdown image syntax: ![Brief alt text](filename.png)
                3. Add a visible caption line immediately below: *Figure N: Full caption text*
                4. Use the exact caption text provided above for each figure
                5. Place figures at appropriate points in the narrative where they support your discussion

                Example:
                As shown in Figure 1, our method...

                ![Figure 1](output/experiments/experiment_hyp_001/learning_curves.png)
                *Figure 1: Learning curves comparing the ...*"""
            )

        return section_guidelines

    def generate_title(
        self,
        abstract: str,
        introduction: str,
        conclusion: str,
        context: PaperConcept,
        temperature: float = 0.4,
        max_tokens: int = 100,
    ) -> str:
        """Generate a paper title based on abstract, introduction, and conclusion."""

        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer.
            
            [TASK]
            Create a concise, informative paper title based on the abstract and conclusion.

            [REQUIREMENTS]
            - Be clear, concise and descriptive
            - Use standard academic title formatting (title case)
            - Avoid unnecessary words like 'A Study of' or 'An Investigation into'
            - ONLY output the title text, without quotes, additional text or formatting

            [ABSTRACT]
            {abstract}

            [CONCLUSION]
            {conclusion}

            Now generate only the title text."""
        )

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

