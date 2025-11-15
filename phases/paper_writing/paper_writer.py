import textwrap
import lmstudio as lms
from typing import Dict, List, Optional, Sequence

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult, Plot
from phases.paper_writing.data_models import Evidence, PaperDraft, Section


class PaperWriter:
    """Generates research paper sections using a LLM."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = lms.llm(model_name)
    
    def generate_paper_sections(
        self,
        context: PaperConcept,
        experiment: ExperimentResult,
        evidence_by_section: Dict[Section, Sequence[Evidence]],
    ) -> PaperDraft:
        """Generate all paper sections using provided evidence."""

        sections = {}
        # Generate sections in order: Methods, Results, Discussion, Introduction, Related Work, Conclusion, Abstract
        for section_type in (
            Section.METHODS,
            Section.RESULTS,
            Section.DISCUSSION,
            Section.INTRODUCTION,
            Section.RELATED_WORK,
            Section.CONCLUSION,
            Section.ABSTRACT,
        ):
            evidence = evidence_by_section.get(section_type, [])
            sections[section_type] = self.generate_section(
                section_type=section_type,
                context=context,
                experiment=experiment,
                evidence=evidence,
                previous_sections=sections,
            )

        # Generate title after all sections are complete, using abstract, introduction, and conclusion
        title = self.generate_title(
            abstract=sections[Section.ABSTRACT],
            introduction=sections[Section.INTRODUCTION],
            conclusion=sections[Section.CONCLUSION],
            context=context,
        )

        return PaperDraft(
            title=title,
            abstract=sections[Section.ABSTRACT],
            introduction=sections[Section.INTRODUCTION],
            related_work=sections[Section.RELATED_WORK],
            methods=sections[Section.METHODS],
            results=sections[Section.RESULTS],
            discussion=sections[Section.DISCUSSION],
            conclusion=sections[Section.CONCLUSION],
        )

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

        prompt = self._build_section_prompt(section_type, context, experiment, evidence, previous_sections)
        response = self.model.respond(
            prompt,
            config={
                "temperature": temperature,
            },
        )
        return self._extract_response_text(response)

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
            - Use citations strictly via the provided keys in parentheses, placed immediately before final punctuation.
            - Combine multiple supporting evidence items using semicolons within a single set of parentheses.
            - Never fabricate evidence, results, or citations.
            - Integrate and build upon previous sections to ensure full narrative coherence.

            [GENERATION RULES — DO NOT VIOLATE]
            - Do NOT reference the guidelines or instructions.
            - Do NOT comment on the evidence structure.
            - Output ONLY the final written section.

            [FINAL PRIORITY]
            Your output must strictly follow the requirements and produce a polished academic section.
        """)
        
        return prompt

    @staticmethod
    def _format_evidence_for_prompt(evidence: Sequence[Evidence]) -> str:
        lines: List[str] = []
        for idx, item in enumerate(evidence, 1):
            citation_key = getattr(item.chunk.paper, "citation_key", "unknown")
            source_info = (
                f"{item.chunk.paper.title} "
                f"({citation_key}; {item.chunk.paper.published or 'n.d.'})"
            )
            lines.append(f"{idx}. Summary: {item.summary}")
            lines.append(f"   Source: {source_info}")
            lines.append(
                f"   Scores → vector: {item.vector_score:.3f}, LLM: {item.llm_score:.3f}, combined: {item.combined_score:.3f}"
            )
            lines.append("")
        return "\n".join(lines).strip()

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
                # Truncate very long sections to avoid token limits
                if len(section_text) > 4000:
                    section_text = section_text[:4000] + "..."
                parts.append(f"{prev_section.value}:\n{section_text}")
        
        if not parts:
            return ""
        
        return "\n\n".join(parts)

    @staticmethod
    def _format_context(
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> str:
        parts = [
            ("Concept description", context.description),
            ("Open questions", context.open_questions),
        ]

        if experiment:
            parts.extend(
                [
                    ("Hypothesis", getattr(experiment.hypothesis, "description", "")),
                    ("Expected improvement", getattr(experiment.hypothesis, "expected_improvement", "")),
                    ("Experimental plan", experiment.experimental_plan),
                    ("Key execution output", getattr(experiment.execution_result, "stdout", "")),
                    ("Verdict", getattr(experiment.hypothesis_evaluation, "verdict", "")),
                    ("Verdict reasoning", getattr(experiment.hypothesis_evaluation, "reasoning", "")),
                ]
            )

        formatted = [
            f"[{label.upper()}]\n{value.strip()}"
            for label, value in parts
            if isinstance(value, str) and value.strip()
        ]
        return "\n\n".join(formatted)

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
            Section.ABSTRACT: 
            """Summarize the purpose, methodology, key findings, and implications succinctly.
            Include major contributions and outcomes in 3-5 sentences.""",
            Section.INTRODUCTION: 
            """Establish context and motivation for the research.
            Introduce the problem and hypothesis derived from open questions.""",
            Section.RELATED_WORK:
            """Review existing work in the field and position this research relative to prior contributions.
            Identify gaps, limitations, and how this work addresses them.
            Organize by themes or approaches, comparing and contrasting related methods.""",
            Section.METHODS: 
            """Detail the experimental setup, methodology, and implementation choices.
            Contrast the approach with comparable methods or baselines.""",
            Section.RESULTS: 
            self._get_results_guidelines(experiment),
            Section.DISCUSSION: 
            """Interpret findings, discuss limitations, and relate to prior work.
            Highlight implications and potential future directions.""",
            Section.CONCLUSION: 
            """Summarize overall contributions and lessons learned.
            Outline broader impact and suggested future work.""",
        }

        return section_guidelines.get(section_type, "")

    def _get_results_guidelines(self, experiment: Optional[ExperimentResult]) -> str:
        """Get Results section guidelines, including figure integration if plots are available."""
        section_guidelines = """Present experimental outcomes with relevant metrics or observations.
        Compare results against expected improvements or baselines if available.
        Never fabricate data or results.
        """

        if experiment and experiment.plots:
            plots_block = self._format_plots_for_prompt(experiment.plots)
            section_guidelines += textwrap.dedent(f"""
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
                *Figure 1: Learning curves comparing the ...*
            """)

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

            Now generate only the title text.
        """)
        response = self.model.respond(
            prompt,
            config={
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        )
        title = self._extract_response_text(response)
        # Clean up title - remove quotes if present, ensure proper capitalization
        title = title.strip().strip('"').strip("'")
        return title

    @staticmethod
    def _extract_response_text(response) -> str:
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()

