import json
import textwrap
import numpy as np
from typing import List, Tuple
from pathlib import Path
from phases.context_analysis.paper_conception import PaperConcept
from phases.hypothesis_generation.hypothesis_models import Hypothesis, HypothesesList, HypothesesResult, SelectedHypotheses
from utils.lazy_model_loader import LazyModelMixin, LazyEmbeddingMixin
from utils.file_utils import save_json, load_json


class HypothesisBuilder(LazyModelMixin, LazyEmbeddingMixin):
    """Generates and validates research hypotheses"""
    
    def __init__(self, model_name: str, embedding_model_name: str, paper_concept: PaperConcept, top_limitations: List[Tuple[str, float]], num_papers_analyzed: int):
        self.model_name = model_name
        self._model = None  # Lazy-loaded via LazyModelMixin
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None  # Lazy-loaded via LazyEmbeddingMixin
        self.paper_concept = paper_concept
        self.top_limitations = top_limitations
        self.num_papers_analyzed = num_papers_analyzed
        
    def validate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Validate a list of hypotheses and filter to only valid ones.
        
        Returns: List of valid hypotheses
        """
        valid_hypotheses = []
        
        print("\nValidating hypotheses...")
        for hypothesis in hypotheses:
            is_valid = True
            reason = ""
            
            # Has clear description
            if not hypothesis.description or len(hypothesis.description) < 20:
                is_valid = False
                reason = "Description too short or missing"
            
            # Has rationale
            elif not hypothesis.rationale or len(hypothesis.rationale) < 20:
                is_valid = False
                reason = "Rationale too short or missing"
            
            if is_valid:
                valid_hypotheses.append(hypothesis)
                print(f"{hypothesis.id}: {hypothesis.description[:60]}...")
            else:
                print(f"{hypothesis.id}: Rejected - {reason}")
        
        return valid_hypotheses
    
    def generate_hypotheses(self, n_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Generate hypotheses and validate them.
        
        May generate more than N initially to ensure N valid hypotheses.
        """
        print(f"\nGenerating {n_hypotheses} hypotheses...")
        
        # Format limitations for prompt (use top 5)
        limitations_text = "\n".join([f"- {limitation} (score: {score:.2f})" for limitation, score in self.top_limitations[:5]])
        
        # Get available code snippets
        code_snippets = self.paper_concept.code_snippets if self.paper_concept.code_snippets else "No code provided"
        
        prompt = textwrap.dedent(f"""\
            You are a research hypothesis generator.
            Task: Generate {n_hypotheses} testable research hypotheses.

            REQUIREMENTS for each hypothesis:
            1. Can be tested programmatically!
            2. Is testable and measurable with clear success criteria
            3. Does NOT suggest incompatible method combinations (e.g., don't mix deterministic with stochastic)
            4. Focuses on realistic improvements - DO NOT include specific percentages, multipliers, or numeric improvements
            5. Use qualitative descriptions instead (e.g., "improved convergence", "better sample efficiency", "reduced memory usage")

            For each hypothesis provide:
            - id: unique identifier (e.g., "hyp_001", "hyp_002", etc.) - REQUIRED, must NOT be empty
            - description: Clear, testable statement (NO percentages or specific numbers unless preliminary results exist)
            - rationale: Why this hypothesis addresses the limitation (reference literature limitations)
            - success_criteria: Clear, measurable criterion or criteria for determining if the hypothesis is validated. 
              CRITICAL: Do NOT include specific numbers, percentages, multipliers, or quantitative targets (e.g., "10x faster", "50% improvement", "reduces error by 20%"). 
              These are impossible to know before running experiments and are pure speculation. 
              Instead, use qualitative, observable criteria (e.g., "shows improved convergence", "demonstrates better sample efficiency", "exhibits reduced memory usage", "achieves stable performance")

            Research Context:
            {self.paper_concept.description}

            User provided implementations/code:
            {code_snippets}

            Some found research limitations (that could be inspiration, only use if it is relevant to the paper concept):
            {limitations_text}

            Generate exactly the {n_hypotheses} hypotheses now.
        """)

        try:
            # Generate hypotheses using structured response
            result = self.model.respond(
                prompt,
                response_format=HypothesesList,
                config={"temperature": 0.7, "maxTokens": 2000}
            )
            
            # LLM returns dict with "hypotheses" key containing list of hypothesis dicts
            response_data = result.parsed
            hypotheses_data = response_data.get("hypotheses", [])
            
            # Convert dict data to Hypothesis objects
            hypotheses = []
            for i, hyp_data in enumerate(hypotheses_data[:n_hypotheses], 1):
                # hyp_data is a dict from the structured response
                hypothesis = Hypothesis(
                    id=hyp_data.get("id", f"hyp_{i:03d}"),
                    description=hyp_data.get("description", ""),
                    rationale=hyp_data.get("rationale", ""),
                    success_criteria=hyp_data.get("success_criteria", "")
                )
                hypotheses.append(hypothesis)
            
            # Validate and filter
            valid_hypotheses = self.validate_hypotheses(hypotheses)
            
            # Return top N valid
            final_hypotheses = valid_hypotheses[:n_hypotheses]
            
            # Automatically save
            if final_hypotheses:
                HypothesisBuilder.save_hypotheses(final_hypotheses, filepath="output/hypotheses.json", num_papers_analyzed=self.num_papers_analyzed)
            
            return final_hypotheses
        
        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            # Return empty list on error
            return []
    def create_hypothesis_from_user_input(self, user_requirements) -> List[Hypothesis]:
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
    def save_hypotheses(hypotheses: List[Hypothesis], filepath: str, num_papers_analyzed: int = 0):
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
    def load_hypotheses(filepath: str) -> List[Hypothesis]:
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
    
    def select_best_hypotheses(self, hypotheses: List[Hypothesis], max_n: int) -> List[Hypothesis]:
        """
        Select the most feasible and best hypotheses from a list, prioritizing testability via Python code.
        
        Args:
            hypotheses: List of hypothesis objects to choose from
            max_n: Maximum number of hypotheses to select
        
        Returns:
            List of selected hypothesis objects (may be fewer than max_n if not enough feasible ones)
        """
        if not hypotheses:
            return []
        
        # Check if any hypotheses are already selected
        already_selected = [h for h in hypotheses if h.selected_for_experimentation]
        if already_selected:
            print(f"\nFound {len(already_selected)} already selected hypotheses.")
            # If we have more than max_n, just take the first max_n (or could add logic to re-select)
            return already_selected[:max_n]
        
        if len(hypotheses) <= max_n:
            # If few hypotheses, select all of them
            for h in hypotheses:
                h.selected_for_experimentation = True
            HypothesisBuilder.save_hypotheses(hypotheses, "output/hypotheses.json", num_papers_analyzed=self.num_papers_analyzed)
            return hypotheses
        
        # Format hypotheses for prompt
        hypotheses_text = "\n\n".join([
            f"ID: {h.id}\n"
            f"Description: {h.description}\n"
            f"Rationale: {h.rationale}\n"
            f"Success Criteria: {h.success_criteria}"
            for h in hypotheses
        ])
        
        prompt = textwrap.dedent(f"""\
            You are selecting the most feasible and best hypotheses for experiment testing.

            CRITICAL REQUIREMENT: Each selected hypothesis MUST be testable via Python code! 

            A hypothesis is testable via Python code if:
            - It can be implemented and tested programmatically
            - It has clear, measurable success criteria
            - The success can be quantified through code execution
            - It does NOT require human evaluation, surveys, or manual analysis
            - It does NOT require proprietary data or external APIs that cannot be simulated

            Selection Criteria (in order of importance):
            1. TESTABILITY VIA PYTHON CODE (MOST IMPORTANT - reject if not testable programmatically!)
            2. Clear baseline to beat (measurability)
            3. Scientific rigor and potential impact
            4. Alignment with the paper concept

            Instructions:
            - You MUST select EXACTLY {max_n} hypothesis/hypotheses (or fewer if not enough are feasible)
            - Do NOT select more than {max_n} hypotheses - this is a strict limit
            - Select the {max_n} hypotheses that are MOST FEASIBLE to test via Python code
            - If fewer than {max_n} hypotheses are feasible to test programmatically, select only the feasible ones
            - Prioritize hypotheses that can be tested with clear metrics and comparisons
            - Return ONLY the ID(s) of selected hypothesis/hypotheses

            Paper concept:
            {self.paper_concept.description}

            Hypotheses to choose from:
            {hypotheses_text}
        """)

        try:
            result = self.model.respond(
                prompt,
                response_format=SelectedHypotheses,
                config={"temperature": 0.2, "maxTokens": 500}
            )
            
            selected_data = result.parsed
            selected_ids = selected_data.get("selected_ids", [])
            
            # Enforce max_n limit - take only the first max_n IDs
            if len(selected_ids) > max_n:
                print(f"Warning: LLM selected {len(selected_ids)} hypotheses, limiting to {max_n}")
                selected_ids = selected_ids[:max_n]
            
            # Filter hypotheses by selected IDs and update status
            selected_hypotheses = []
            id_to_hypothesis = {h.id: h for h in hypotheses}
            
            for hyp_id in selected_ids:
                if hyp_id in id_to_hypothesis:
                    h = id_to_hypothesis[hyp_id]
                    h.selected_for_experimentation = True
                    selected_hypotheses.append(h)

            # Save the updated state (with selected flags)
            HypothesisBuilder.save_hypotheses(hypotheses, "output/hypotheses.json", num_papers_analyzed=self.num_papers_analyzed)
            
            print(f"\nSelected {len(selected_hypotheses)} {'hypothesis' if len(selected_hypotheses) == 1 else 'hypotheses'} from {len(hypotheses)} candidates:")
            for h in selected_hypotheses:
                print(f"  - {h.id}: {h.description[:60]}...")
            
            return selected_hypotheses
        
        except Exception as e:
            print(f"Error selecting hypotheses: {e}")
            # Fallback: return first max_n hypotheses if selection fails
            fallback = hypotheses[:max_n]
            for h in fallback:
                h.selected_for_experimentation = True
            HypothesisBuilder.save_hypotheses(hypotheses, "output/hypotheses.json", num_papers_analyzed=self.num_papers_analyzed)
            return fallback

