from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel
from phases.paper_search.paper import Paper

class Section(str, Enum):
    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    RELATED_WORK = "Related Work"
    METHODS = "Methods"
    RESULTS = "Results"
    DISCUSSION = "Discussion"
    CONCLUSION = "Conclusion"
    ACKNOWLEDGEMENTS = "Acknowledgements"


@dataclass
class PaperDraft:
    """Container for generated paper sections."""
    title: str
    abstract: str
    introduction: str
    related_work: str
    methods: str
    results: str
    discussion: str
    conclusion: str
    acknowledgements: Optional[str] = None


@dataclass
class PaperChunk:
    """Indexed chunk of a source paper used for retrieval."""
    chunk_id: str
    paper: Paper
    chunk_text: str
    chunk_index: int
    embedding: list[float] = field(default_factory=list)


class ScoreResult(BaseModel):
    """Structured response for evidence relevance scoring."""
    score: float
    reason: str


class SummaryItem(BaseModel):
    summary: str


class SummaryBatchResult(BaseModel):
    results: list[SummaryItem]


class ScoreItem(BaseModel):
    score: float
    reason: str


class ScoreBatchResult(BaseModel):
    results: list[ScoreItem]


@dataclass
class Evidence:
    """Scored evidence chunk returned by retrieval pipeline."""
    chunk: PaperChunk
    summary: str
    vector_score: float
    llm_score: float
    combined_score: float
    source_query: str



