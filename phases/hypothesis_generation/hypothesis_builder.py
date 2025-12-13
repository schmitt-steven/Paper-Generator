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
    selected_for_experimentation: bool = False


class HypothesesList(BaseModel):
    """Simple container for LLM-generated hypotheses"""
    hypotheses: list[Hypothesis]


class HypothesesResult(BaseModel):
    """Container for generated hypotheses with metadata"""
    paper_concept_file: str
    num_papers_analyzed: int
    top_limitations_addressed: list[dict[str, float]]
    hypotheses: list[Hypothesis]


class SelectedHypotheses(BaseModel):
    """Selected hypothesis IDs from a selection process"""
    selected_ids: list[str]
from utils.lazy_model_loader import LazyModelMixin, LazyEmbeddingMixin
from utils.file_utils import save_json, load_json


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
        
    def create_hypothesis_from_user_input(self, user_requirements) -> list[Hypothesis]:
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
            
            hypotheses = []
            for i, hyp_data in enumerate(hypotheses_data, 1):
                hypothesis = Hypothesis(
                    id=hyp_data.get("id", f"user_hyp_{i:03d}"),
                    description=hyp_data.get("description", user_hypothesis_text), # Fallback to raw text
                    rationale=hyp_data.get("rationale", "User provided hypothesis"),
                    success_criteria=hyp_data.get("success_criteria", "As specified by user"),
                    selected_for_experimentation=True # Auto-select user hypothesis
                )
                hypotheses.append(hypothesis)
            
            if not hypotheses:
                # Fallback if LLM fails to return a list
                print("Warning: LLM failed to structure user hypothesis. Creating basic object.")
                hypotheses = [Hypothesis(
                    id="user_hyp_001",
                    description=user_hypothesis_text,
                    rationale="User provided hypothesis",
                    methods="User specified",
                    success_criteria="Unknown",
                    selected_for_experimentation=True
                )]

            # Save it
            HypothesisBuilder.save_hypotheses(hypotheses, "output/hypotheses.json", num_papers_analyzed=self.num_papers_analyzed)
            
            return hypotheses

        except Exception as e:
            print(f"Error processing user hypothesis: {e}")
            # Fallback
            return [Hypothesis(
                id="user_hyp_001",
                description=user_hypothesis_text,
                rationale="User provided hypothesis (Error in processing)",
                methods="User specified",
                success_criteria="Unknown",
                selected_for_experimentation=True
            )]

    @staticmethod
    def save_hypotheses(hypotheses: list[Hypothesis], filepath: str, num_papers_analyzed: int = 0):
        """Save hypotheses to JSON file."""
        path_obj = Path(filepath)

        result_dict = {
            "paper_concept_file": "output/paper_concept.md",
            "num_papers_analyzed": num_papers_analyzed,
            "hypotheses": [
                {
                    "id": h.id,
                    "description": h.description,
                    "rationale": h.rationale,
                    "success_criteria": h.success_criteria,
                    "selected_for_experimentation": h.selected_for_experimentation
                }
                for h in hypotheses
            ]
        }

        save_json(result_dict, path_obj.name, str(path_obj.parent))

        print(f"\nSaved {len(hypotheses)} hypotheses to {filepath}")
    
    @staticmethod
    def load_hypotheses(filepath: str) -> list[Hypothesis]:
        """
        Load hypotheses from a JSON file.

        Args:
            filepath: Path to the JSON file containing saved hypotheses

        Returns:
            List of Hypothesis objects
        """
        try:
            path_obj = Path(filepath)
            data = load_json(path_obj.name, str(path_obj.parent))

            hypotheses = []
            for hyp_data in data.get("hypotheses", []):
                hypothesis = Hypothesis(
                    id=hyp_data.get("id", ""),
                    description=hyp_data.get("description", ""),
                    rationale=hyp_data.get("rationale", ""),
                    success_criteria=hyp_data.get("success_criteria", ""),
                    selected_for_experimentation=hyp_data.get("selected_for_experimentation", False)
                )
                hypotheses.append(hypothesis)

            print(f"Loaded {len(hypotheses)} hypotheses from {filepath}")
            return hypotheses

        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {filepath}: {e}")
            return []
        except Exception as e:
            print(f"Error loading hypotheses: {e}")
            return []


