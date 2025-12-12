from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from pydantic import BaseModel


class FindingsExtractionResult(BaseModel):
    """LLM extraction result containing paper findings (without metadata)"""
    methods_used: list[str]
    test_setup: str
    main_limitations: str


class PaperFindings(BaseModel):
    """Extracted key findings from a single paper"""
    paper_id: str
    title: str
    methods_used: list[str]
    test_setup: str
    main_limitations: str


class LiteratureGaps(BaseModel):
    """Aggregated analysis of research gaps from literature"""
    method_frequency: dict[str, int]
    common_limitations: dict[str, int]


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

