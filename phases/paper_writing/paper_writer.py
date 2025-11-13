import lmstudio as lms
from typing import Dict, List, Optional, Sequence

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult
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
        # Generate sections in order: Methods, Results, Discussion, Introduction, Conclusion, Abstract
        for section_type in (
            Section.METHODS,
            Section.RESULTS,
            Section.DISCUSSION,
            Section.INTRODUCTION,
            Section.CONCLUSION,
            Section.ABSTRACT,
        ):
            evidence = evidence_by_section.get(section_type, [])
            sections[section_type] = self.generate_section(
                section_type=section_type,
                context=context,
                experiment=experiment,
                evidence=evidence,
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
        temperature: float = 0.35,
        max_tokens: int = 900,
    ) -> str:
        """Generate a single section given context and evidence."""

        prompt = self._build_section_prompt(section_type, context, experiment, evidence)
        response = self.model.respond(
            prompt,
            config={
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        )
        return self._extract_response_text(response)

    def _build_section_prompt(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        evidence: Sequence[Evidence],
    ) -> str:
        """Create the generation prompt for a specific section."""

        guidelines = self.get_section_guidelines(section_type)
        context_block = self._format_context(context, experiment)
        evidence_block = self._format_evidence_for_prompt(evidence)

        return f"""[ROLE]
You are an expert academic writer tasked with drafting a section of a research paper.
Write in a formal academic tone and integrate evidence smoothly with indirect citations (e.g., (smith2024reinforcement)). Ensure the narrative is cohesive and original.

[SECTION TYPE]
{section_type.value}

[SECTION GUIDELINES]
{guidelines}

[RESEARCH CONTEXT]
{context_block}

[EVIDENCE]
{evidence_block if evidence_block else 'No evidence available.'}

[REQUIREMENTS]
- Emphasize the most relevant evidence for the section objectives.
- Refer to evidence using the provided citation keys in parentheses.
- Do not fabricate data or citations.
- Maintain logical flow and avoid bullet lists unless necessary.
"""

    @staticmethod
    def _format_evidence_for_prompt(evidence: Sequence[Evidence]) -> str:
        lines: List[str] = []
        for idx, item in enumerate(evidence, 1):
            citation_key = getattr(item.chunk.paper, "citation_key", "unknown")
            source_info = (
                f"{item.chunk.paper.title} "
                f"({citation_key}; {item.chunk.paper.published or 'n.d.'}; "
                f"{item.chunk.section_type.value})"
            )
            lines.append(f"{idx}. Summary: {item.summary}")
            lines.append(f"   Source: {source_info}")
            lines.append(
                f"   Scores → vector: {item.vector_score:.3f}, LLM: {item.llm_score:.3f}, combined: {item.combined_score:.3f}"
            )
            lines.append("")
        return "\n".join(lines).strip()

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

    @staticmethod
    def get_section_guidelines(section_type: Section) -> str:
        """
        Specifies writing guidelines for each paper section.
        These guidelines are combined with more context and evidence in _build_section_prompt().
        """
        section_guidelines = {
            Section.ABSTRACT: 
            """Summarize the purpose, methodology, key findings, and implications succinctly.
            Include major contributions and outcomes in 3-5 sentences.""",
            Section.INTRODUCTION: 
            """Establish context and related work leading to the research gap.
            Clarify the hypothesis and motivation derived from open questions.""",
            Section.METHODS: 
            """Detail the experimental setup, methodology, and implementation choices.
            Contrast the approach with comparable methods or baselines.""",
            Section.RESULTS: 
            """Present experimental outcomes with relevant metrics or observations.
            Compare results against expected improvements or baselines.""",
            Section.DISCUSSION: 
            """Interpret findings, discuss limitations, and relate to prior work.
            Highlight implications and potential future directions.""",
            Section.CONCLUSION: 
            """Summarize overall contributions and lessons learned.
            Outline broader impact and suggested future work.""",
        }

        return section_guidelines.get(section_type, "")

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

        prompt = f"""[ROLE]
You are an expert academic writer tasked with creating a concise, informative paper title.
The title should:
- Be clear and descriptive of the main contribution
- Use standard academic title formatting (title case)
- Be concise (typically 8-15 words)
- Capture the key innovation or finding
- Avoid unnecessary words like 'A Study of' or 'An Investigation into'

[ABSTRACT]
{abstract}

[INTRODUCTION] (key excerpts)
{introduction[:1000]}

[CONCLUSION] (key excerpts)
{conclusion[:1000]}

Generate only the title text, without quotes or additional formatting.
"""
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

