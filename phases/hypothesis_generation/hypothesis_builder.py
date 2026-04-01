import json
from utils.llm_utils import remove_thinking_blocks
import textwrap
import numpy as np
from typing import List, Tuple
from pathlib import Path
from phases.context_analysis.research_context_generator import ResearchContext
from pydantic import BaseModel
from dataclasses import dataclass
from settings import Settings
from typing import List, Dict, Tuple, Optional
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from phases.context_analysis.paper_specification import PaperSpecification
from utils.lazy_model_loader import LazyModelMixin, LazyEmbeddingMixin
from utils.file_utils import save_markdown, load_markdown

class Hypothesis(BaseModel):
    """A testable research hypothesis"""
    id: str
    description: str
    rationale: str
    success_criteria: str
    selected_for_experimentation: bool = True  # Always true for single hypothesis flow

    def to_markdown(self) -> str:
        """Convert hypothesis to markdown format."""
        return textwrap.dedent(f"""\
            # Research Hypothesis

            ## Description
            {self.description}

            ## Rationale
            {self.rationale}

            ## Success Criteria
            {self.success_criteria}
            """)

    @classmethod
    def from_markdown(cls, content: str, hyp_id: str = "user_hypothesis") -> "Hypothesis":
        """Parse hypothesis from markdown content."""
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].lower().replace(' ', '_')
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return cls(
            id=hyp_id,
            description=sections.get('description', ''),
            rationale=sections.get('rationale', ''),
            success_criteria=sections.get('success_criteria', ''),
            selected_for_experimentation=True
        )


class HypothesisBuilder(LazyModelMixin):
    """Generates and validates research hypotheses"""
    
    def __init__(self, model_name: str, research_context: ResearchContext, top_limitations: list[tuple[str, float]], num_papers_analyzed: int):
        self.model_name = model_name
        self._model = None  # Lazy-loaded via LazyModelMixin
        self.research_context = research_context
        self.top_limitations = top_limitations
        self.num_papers_analyzed = num_papers_analyzed
        
    def create_hypothesis_from_user_input(self, paper_specification) -> Hypothesis:
        """
        Create a Hypothesis object from a user-provided string.
        Uses LLM to structure the raw text into a proper Hypothesis object.
        """
        user_hypothesis_text = paper_specification.hypothesis
        print(f"\nProcessing user-provided hypothesis...")
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"Paper Title: {Settings.LATEX_TITLE}\n\n"
        
        prompt = textwrap.dedent(f"""\
            You are a research assistant helping to structure a user's research hypothesis.
                        
            Task: Convert this raw hypothesis into a structured format.
            
            REQUIREMENTS:
            1. Extract/Infer a clear description, rationale, and success criteria.
            2. If information is missing, infer reasonable defaults based on the context or mark as "Not specified".
            3. Ensure the output is a valid Hypothesis object.
            4. Use the additional paper specification to better understand the context and intent of the hypothesis.
            5. CRITICAL for success_criteria: Criteria must be concrete and testable. Use one or more of the following patterns depending on what fits the hypothesis:
               - Relative comparison: "achieves lower mean error than baseline Y", "converges in fewer epochs than the baseline"
               - Statistical significance: "the improvement over baseline X is statistically significant"
               - Ablation: "removing component Y from the method degrades performance" (use when the method has distinct components)
               - Existence/capability: "the method successfully performs X where baseline Y fails to" (use when the baseline cannot do the task at all)
               Do NOT invent specific numeric thresholds — no made-up percentages, multipliers, or quantitative targets (e.g., "10x faster", "50% improvement").
               However, if the user's paper specification already contains specific numeric targets, you may use those as-is.
               Avoid vague, unfalsifiable language (e.g., "shows improved convergence", "demonstrates better efficiency").
               Pick the 1-2 pattern(s) that best fit(s) the hypothesis — do not combine all of them.
            6. Zero Bullshit Policy / No AI Meta-Commentary: Write directly in the third person about the scientific subject.
               NEVER use phrases like "The user wants me to structure...", "This hypothesis addresses...", "Based on the provided context...".
               Start the description directly with the scientific phenomenon (e.g. "Validating official statistics integrity using Benford's Law...").
               Eliminate entirely AI filler words ("robust", "seamless", "comprehensive", "leverage", "vital").
            
            For the structured hypothesis, provide:
            - id: unique identifier (e.g., "user_hypothesis_01")
            - description: Clear, testable scientific statement extracted from the user's input (NO meta-commentary)
            - rationale: The scientific justification for this hypothesis (NO meta-commentary about the user or prompt)
            - success_criteria: Concrete, testable criteria using one or more patterns: relative comparison against a baseline,
              statistical significance, ablation (component necessity), or existence/capability (binary pass/fail).
              Do NOT invent numeric thresholds — but if the paper specification provides specific numbers, use them.
            
            Research Context:
            {title_section}{self.research_context.description}

            User's raw hypothesis:
            "{user_hypothesis_text}"
            
            Additional Paper Specification/Context:
            Topic: {paper_specification.topic}
            Methods: {paper_specification.methods}
            Results: {paper_specification.results}
            Discussion: {paper_specification.discussion}
            
            Generate the structured hypothesis now."""
        )

        try:
            # Generate hypothesis using structured response
            result = self.model.respond(
                prompt,
                response_format=Hypothesis,
                config={"temperature": 0.2}
            )
            
            content = remove_thinking_blocks(result.content)
            response_data = json.loads(content)
            hypothesis = Hypothesis(
                id="user_hypothesis",
                description=response_data.get("description", user_hypothesis_text),
                rationale=response_data.get("rationale", "User provided hypothesis"),
                success_criteria=response_data.get("success_criteria", "As specified by user"),
                selected_for_experimentation=True
            )

            # Save it
            HypothesisBuilder.save_hypothesis(hypothesis, "output/hypothesis.md")
            
            return hypothesis

        except Exception as e:
            print(f"Error processing user hypothesis: {e}")
            # Fallback
            hyp = Hypothesis(
                id="user_hypothesis",
                description=user_hypothesis_text,
                rationale="User provided hypothesis (Error in processing)",
                success_criteria="Unknown",
                selected_for_experimentation=True
            )
            HypothesisBuilder.save_hypothesis(hyp, "output/hypothesis.md")
            return hyp

        except Exception as e:
            print(f"Error processing user hypothesis: {e}")
            # Fallback
            return [Hypothesis(
                id="user_hypothesis",
                description=user_hypothesis_text,
                rationale="User provided hypothesis (Error in processing)",
                success_criteria="Unknown",
                selected_for_experimentation=True
            )]

    @staticmethod
    def save_hypothesis(hypothesis: Hypothesis, filepath: str):
        """Save single hypothesis to Markdown file."""
        try:
            path_obj = Path(filepath)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(hypothesis.to_markdown(), encoding='utf-8')
            print(f"\nSaved hypothesis to {filepath}")
        except Exception as e:
            print(f"Error saving hypothesis: {e}")

    @staticmethod
    def load_hypothesis(filepath: str) -> Optional[Hypothesis]:
        """Load single hypothesis from Markdown file."""
        try:
            path_obj = Path(filepath)
            if not path_obj.exists():
                return None
            
            content = path_obj.read_text(encoding='utf-8')
            return Hypothesis.from_markdown(content)

        except Exception as e:
            print(f"Error loading hypothesis: {e}")
            return None

    @staticmethod
    def generate_new_hypothesis(status_callback: callable = None) -> Hypothesis:
        """
        Generate hypothesis from paper specification.
        
        This handles:
        1. Loading research context
        2. Loading paper specification
        3. Generating hypothesis using LLM
        4. Saving the result
        
        Args:
            status_callback: Optional callback function(str) for progress updates.
            
        Returns:
            The generated Hypothesis object.
        """
        
        if status_callback:
            status_callback("Loading research context")
        research_context = ResearchContextGenerator.load_research_context("output/research_context.md")
        
        if status_callback:
            status_callback("Loading paper specification")
        paper_specification = PaperSpecification.load("user_files/paper_specification.md")
        
        if status_callback:
            status_callback("Generating hypothesis")
        builder = HypothesisBuilder(
            model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
            research_context=research_context,
            top_limitations=[],
            num_papers_analyzed=0
        )
        return builder.create_hypothesis_from_user_input(paper_specification)


