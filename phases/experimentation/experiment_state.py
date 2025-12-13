from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from phases.hypothesis_generation.hypothesis_builder import Hypothesis

class VerdictResult(BaseModel):
    """Structured verdict result for hypothesis evaluation."""
    verdict: str
    reasoning: str

@dataclass
class ExecutionResult:
    """Result from executing Python code."""
    stdout: str
    stderr: str
    return_code: int
    plot_files: list[str] = field(default_factory=list)
    result_files: list[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if execution had errors."""
        return self.return_code != 0 or bool(self.stderr and self.stderr.strip())
    
    @property
    def succeeded(self) -> bool:
        """Check if execution succeeded."""
        return self.return_code == 0 and not self.has_errors


@dataclass
class CodeGenerationResult:
    """Result from generating experiment code."""
    code_file_path: Optional[str]
    execution_result: ExecutionResult


@dataclass
class ExperimentFiles:
    """Loaded experiment files."""
    experiment_plan: str
    experiment_code: str
    plan_file_path: str
    code_file_path: str

class ValidationResult(BaseModel):
    """Structured validation result for experiment results."""
    is_valid: bool
    reasoning: str
    issues: Optional[str] = None


@dataclass
class HypothesisEvaluation:
    """Final verdict on whether hypothesis is proven/disproven/inconclusive."""
    hypothesis_id: str
    verdict: str  # "proven", "disproven", or "inconclusive"
    reasoning: str


@dataclass
class Plot:
    """Plot file with caption for paper generation."""
    filename: str
    caption: str


@dataclass
class ExperimentResult:
    """Aggregation of all experiment data."""
    
    hypothesis: Hypothesis

    experiment_plan: str
    experiment_code: str
    
    execution_result: ExecutionResult
    validation_result: ValidationResult
    hypothesis_evaluation: HypothesisEvaluation
    
    plots: list[Plot] = field(default_factory=list)
    
    fix_attempts: int = 0
    validation_attempts: int = 0
    execution_time: Optional[float] = None