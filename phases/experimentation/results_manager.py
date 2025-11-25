import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from phases.experimentation.experiment_state import HypothesisEvaluation
from utils.file_utils import save_json, load_json


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

        eval_data = {
            "hypothesis_id": evaluation.hypothesis_id,
            "verdict": evaluation.verdict,
            "reasoning": evaluation.reasoning
        }

        filename = f"hypothesis_evaluation_{evaluation.hypothesis_id}.json"
        eval_path = save_json(eval_data, filename, self.base_output_dir)

        return eval_path
    
    @staticmethod
    def load_previous_results(
        hypothesis_id: str,
        run_id: Optional[int] = None,
        base_dir: str = "output/experiments"
    ) -> Dict[str, Any]:
        """Load previous experiment results for comparison."""

        result_data = {}
        eval_path = os.path.join(base_dir, f"hypothesis_evaluation_{hypothesis_id}.json")
        if os.path.exists(eval_path):
            path_obj = Path(eval_path)
            result_data = load_json(path_obj.name, str(path_obj.parent))

        return result_data

