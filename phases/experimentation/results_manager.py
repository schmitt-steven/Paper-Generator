import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from phases.experimentation.experiment_state import HypothesisEvaluation


class ResultsManager:
    """Manages storage and loading of experiment metadata."""
    
    def __init__(self, base_output_dir: str = "output/experiments"):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
    
    def save_hypothesis_evaluation(
        self,
        evaluation: HypothesisEvaluation
    ) -> str:
        """Save hypothesis evaluation (proven/disproven/inconclusive)."""
        
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        eval_data = {
            "hypothesis_id": evaluation.hypothesis_id,
            "verdict": evaluation.verdict,
            "reasoning": evaluation.reasoning
        }
        
        eval_path = os.path.join(self.base_output_dir, f"hypothesis_evaluation_{evaluation.hypothesis_id}.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        return eval_path
    
    def load_previous_results(
        self,
        hypothesis_id: str,
        run_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load previous experiment results for comparison."""
        
        result_data = {}
        eval_path = os.path.join(self.base_output_dir, f"hypothesis_evaluation_{hypothesis_id}.json")
        if os.path.exists(eval_path):
            with open(eval_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        
        return result_data

