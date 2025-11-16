import json
import textwrap
import numpy as np
from typing import List, Tuple
from phases.context_analysis.paper_conception import PaperConcept
from phases.hypothesis_generation.hypothesis_models import Hypothesis, HypothesesList, HypothesesResult, SelectedHypotheses
from utils.lazy_model_loader import LazyModelMixin, LazyEmbeddingMixin


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
        
        # Known incompatible pairs
        self.incompatible_pairs = [
            ("deterministic", "stochastic"),
            ("model-free", "model-based"),
            ("online", "offline"),
            ("on-policy", "off-policy"),
        ]
    
    def check_method_compatibility(self, method_a: str, method_b: str) -> Tuple[bool, str]:
        """
        Check if two methods are compatible for combination.
        
        Returns: (is_compatible, reason)
        """
        method_a_lower = method_a.lower()
        method_b_lower = method_b.lower()
        
        # First: Check known incompatible pairs
        for incompat_a, incompat_b in self.incompatible_pairs:
            if ((incompat_a in method_a_lower and incompat_b in method_b_lower) or
                (incompat_b in method_a_lower and incompat_a in method_b_lower)):
                return False, f"Incompatible: {incompat_a} vs {incompat_b}"
        
        # Second: Check redundancy with embeddings
        emb_a = np.array(self.embedding_model.embed(method_a))
        emb_b = np.array(self.embedding_model.embed(method_b))
        
        similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        
        if similarity > 0.95:
            return False, f"Too similar (redundant): similarity={similarity:.2f}"
        
        # Methods are compatible
        return True, "Compatible"
    
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
            
            # Has baseline (measurability)
            elif not hypothesis.baseline_to_beat:
                is_valid = False
                reason = "No clear baseline to beat (not measurable)"
            
            # Check method compatibility if combination mentioned
            elif "+" in hypothesis.method_combination or "and" in hypothesis.method_combination.lower():
                methods = [m.strip() for m in hypothesis.method_combination.replace("+", " and ").split(" and ")]
                if len(methods) >= 2:
                    compatible, compat_reason = self.check_method_compatibility(methods[0], methods[1])
                    if not compatible:
                        is_valid = False
                        reason = f"Method incompatibility: {compat_reason}"
            
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
            - id: unique identifier (e.g., "hyp_001")
            - description: Clear, testable statement (NO percentages or specific numbers unless preliminary results exist)
            - rationale: Why this hypothesis addresses the limitation (reference literature limitations)
            - method_combination: What methods/approaches to combine
            - expected_improvement: Qualitative improvement expected (avoid percentages)
            - baseline_to_beat: What baseline to compare against (if applicable)

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
                    method_combination=hyp_data.get("method_combination", ""),
                    expected_improvement=hyp_data.get("expected_improvement", ""),
                    baseline_to_beat=hyp_data.get("baseline_to_beat")
                )
                hypotheses.append(hypothesis)
            
            # Validate and filter
            valid_hypotheses = self.validate_hypotheses(hypotheses)
            
            # Return top N valid
            final_hypotheses = valid_hypotheses[:n_hypotheses]
            
            # Automatically save
            if final_hypotheses:
                self.save_hypotheses(final_hypotheses, filepath="output/hypotheses.json")
            
            return final_hypotheses
        
        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            # Return empty list on error
            return []
    
    def save_hypotheses(self, hypotheses: List[Hypothesis], filepath: str):
        """Save hypotheses to JSON file."""
        result_dict = {
            "paper_concept_file": "output/paper_concept.md",
            "num_papers_analyzed": self.num_papers_analyzed,
            "hypotheses": [
                {
                    "id": h.id,
                    "description": h.description,
                    "rationale": h.rationale,
                    "method_combination": h.method_combination,
                    "expected_improvement": h.expected_improvement,
                    "baseline_to_beat": h.baseline_to_beat
                }
                for h in hypotheses
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(hypotheses)} hypotheses to {filepath}")
    
    def load_hypotheses(self, filepath: str) -> List[Hypothesis]:
        """
        Load hypotheses from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing saved hypotheses
        
        Returns:
            List of Hypothesis objects
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            hypotheses = []
            for hyp_data in data.get("hypotheses", []):
                hypothesis = Hypothesis(
                    id=hyp_data.get("id", ""),
                    description=hyp_data.get("description", ""),
                    rationale=hyp_data.get("rationale", ""),
                    method_combination=hyp_data.get("method_combination", ""),
                    expected_improvement=hyp_data.get("expected_improvement", ""),
                    baseline_to_beat=hyp_data.get("baseline_to_beat")
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
        
        if len(hypotheses) <= max_n:
            return hypotheses
        
        # Format hypotheses for prompt
        hypotheses_text = "\n\n".join([
            f"ID: {h.id}\n"
            f"Description: {h.description}\n"
            f"Rationale: {h.rationale}\n"
            f"Method Combination: {h.method_combination}\n"
            f"Expected Improvement: {h.expected_improvement}\n"
            f"Baseline to Beat: {h.baseline_to_beat or 'N/A'}"
            for h in hypotheses
        ])
        
        prompt = textwrap.dedent(f"""\
            You are selecting the most feasible and best hypotheses for experimental testing.

            CRITICAL REQUIREMENT: Each selected hypothesis MUST be testable via Python code! 

            A hypothesis is testable via Python code if:
            - It can be implemented and tested programmatically
            - It has clear, measurable success criteria
            - It has a baseline to compare against (unless explicitly stated as exploratory)
            - The expected improvement can be quantified through code execution
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
            
            # Filter hypotheses by selected IDs
            selected_hypotheses = []
            id_to_hypothesis = {h.id: h for h in hypotheses}
            
            for hyp_id in selected_ids:
                if hyp_id in id_to_hypothesis:
                    selected_hypotheses.append(id_to_hypothesis[hyp_id])
            
            print(f"\nSelected {len(selected_hypotheses)} {'hypothesis' if len(selected_hypotheses) == 1 else 'hypotheses'} from {len(hypotheses)} candidates:")
            for h in selected_hypotheses:
                print(f"  - {h.id}: {h.description[:60]}...")
            
            return selected_hypotheses
        
        except Exception as e:
            print(f"Error selecting hypotheses: {e}")
            # Fallback: return first max_n hypotheses if selection fails
            return hypotheses[:max_n]

