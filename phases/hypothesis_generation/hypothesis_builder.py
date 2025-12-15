import json
import textwrap
import numpy as np
from typing import List, Tuple
from pathlib import Path
from phases.context_analysis.paper_conception import PaperConcept
from phases.context_analysis.paper_conception import PaperConcept
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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



from utils.lazy_model_loader import LazyModelMixin, LazyEmbeddingMixin
from utils.file_utils import save_markdown, load_markdown


class HypothesisBuilder(LazyModelMixin, LazyEmbeddingMixin):
    """Generates and validates research hypotheses"""
    
    def __init__(self, model_name: str, embedding_model_name: str, paper_concept: PaperConcept, top_limitations: list[tuple[str, float]], num_papers_analyzed: int):
        self.model_name = model_name
        self._model = None  # Lazy-loaded via LazyModelMixin
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None  # Lazy-loaded via LazyEmbeddingMixin
        self.paper_concept = paper_concept
        self.top_limitations = top_limitations
        self.num_papers_analyzed = num_papers_analyzed
        
    def create_hypothesis_from_user_input(self, user_requirements) -> Hypothesis:
        """
        Create a Hypothesis object from a user-provided string.
        Uses LLM to structure the raw text into a proper Hypothesis object.
        """
        user_hypothesis_text = user_requirements.hypothesis
        print(f"\nProcessing user-provided hypothesis...")
        
        prompt = textwrap.dedent(f"""\
            You are a research assistant helping to structure a user's research hypothesis.
                        
            Task: Convert this raw hypothesis into a structured format.
            
            REQUIREMENTS:
            1. Extract/Infer a clear description, rationale, and success criteria.
            2. If information is missing, infer reasonable defaults based on the context or mark as "Not specified".
            3. Ensure the output is a valid Hypothesis object.
            4. Use the additional user requirements to better understand the context and intent of the hypothesis.
            5. CRITICAL for success_criteria: Do NOT include specific numbers, percentages, multipliers, or quantitative targets (e.g., "10x faster", "50% improvement", "reduces error by 20%"). 
               These are impossible to know before running experiments and are pure speculation. 
               Instead, use qualitative, observable criteria (e.g., "shows improved convergence", "demonstrates better sample efficiency", "exhibits reduced memory usage", "achieves stable performance")
            
            For the structured hypothesis, provide:
            - id: unique identifier (e.g., "user_hypothesis_01")
            - description: Clear, testable statement extracted from the user's input
            - rationale: Why this hypothesis is relevant (reference the research context if available)
            - success_criteria: Clear, measurable criterion or criteria for determining if the hypothesis is validated.
              CRITICAL: Do NOT include specific numbers, percentages, multipliers, or quantitative targets.
              Use qualitative, observable criteria instead.
            
            Research Context:
            {self.paper_concept.description}

            User's raw hypothesis:
            "{user_hypothesis_text}"
            
            Additional User Requirements/Context:
            Topic: {user_requirements.topic}
            Methods: {user_requirements.methods}
            Results: {user_requirements.results}
            Discussion: {user_requirements.discussion}
            
            Generate the structured hypothesis now."""
        )

        try:
            # Generate hypothesis using structured response
            # We reuse HypothesesList to keep it consistent, even if it's just one
            result = self.model.respond(
                prompt,
                response_format=HypothesesList,
                config={"temperature": 0.3, "maxTokens": 1000}
            )
            
            response_data = result.parsed
            hypotheses_data = response_data.get("hypotheses", [])
            
            # Take the first one or default
            if hypotheses_data:
                hyp_data = hypotheses_data[0]
                hypothesis = Hypothesis(
                    id="user_hypothesis",
                    description=hyp_data.get("description", user_hypothesis_text),
                    rationale=hyp_data.get("rationale", "User provided hypothesis"),
                    success_criteria=hyp_data.get("success_criteria", "As specified by user"),
                    selected_for_experimentation=True
                )
            else:
                raise ValueError("No hypothesis returned by LLM")

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


