from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult
from phases.paper_writing.data_models import Section


@dataclass
class QueryBuilder:
    """Builds section-specific default queries for evidence gathering."""

    def build_default_queries(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult] = None,
    ) -> List[str]:
        builder_map = {
            Section.ABSTRACT: self._build_abstract_queries,
            Section.INTRODUCTION: self._build_introduction_queries,
            Section.RELATED_WORK: self._build_related_work_queries,
            Section.METHODS: self._build_methods_queries,
            Section.RESULTS: self._build_results_queries,
            Section.DISCUSSION: self._build_discussion_queries,
            Section.CONCLUSION: self._build_conclusion_queries,
        }

        builder = builder_map.get(section_type)
        if not builder:
            return []

        queries = builder(context, experiment)
        return [query for query in queries if query]

    def _build_abstract_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        return [
            self._combine_segments(
                ("Research concept overview", context.description),
                ("Hypothesis summary", self._safe_get(experiment, "hypothesis.description")),
                ("Experiment outcomes", self._safe_get(experiment, "execution_result.stdout")),
                ("Validation insights", self._safe_get(experiment, "validation_result.reasoning")),
                ("Success criteria", self._safe_get(experiment, "hypothesis.success_criteria")),
            )
        ]

    def _build_introduction_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        return [
            self._combine_segments(
                ("Research context", context.description),
                ("Open questions", context.open_questions),
                ("Hypothesis overview", self._safe_get(experiment, "hypothesis.description")),
                ("Success criteria", self._safe_get(experiment, "hypothesis.success_criteria")),
            )
        ]

    def _build_related_work_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        return [
            self._combine_segments(
                ("Research domain", context.description),
                ("Research gap", context.open_questions),
                ("Success criteria", self._safe_get(experiment, "hypothesis.success_criteria")),
                ("Key techniques", context.code_snippets),
            )
        ]

    def _build_methods_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        return [
            self._combine_segments(
                ("Experiment plan", self._safe_get(experiment, "experiment_plan")),
                ("Implementation details", self._safe_get(experiment, "experiment_code")),
                ("Key code snippets", context.code_snippets),
            )
        ]

    def _build_results_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        stdout = self._safe_get(experiment, "execution_result.stdout")
        validation = self._safe_get(experiment, "validation_result.reasoning")
        plot_captions = self._join_plot_captions(self._safe_get(experiment, "plots"))

        return [
            self._combine_segments(
                ("Experiment stdout", stdout),
                ("Validation reasoning", validation),
                ("Result files", self._safe_join(self._safe_get(experiment, "execution_result.result_files"))),
                ("Key metrics and observations", stdout),
                ("Plot summaries", plot_captions),
            )
        ]

    def _build_discussion_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        return [
            self._combine_segments(
                ("Hypothesis verdict", self._safe_get(experiment, "hypothesis_evaluation.verdict")),
                ("Verdict reasoning", self._safe_get(experiment, "hypothesis_evaluation.reasoning")),
                ("Validation issues", self._safe_get(experiment, "validation_result.issues")),
                ("Observed limitations", self._safe_get(experiment, "validation_result.reasoning")),
                ("Future work ideas", context.open_questions),
            )
        ]

    def _build_conclusion_queries(
        self,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> List[str]:
        return [
            self._combine_segments(
                ("Final verdict", self._safe_get(experiment, "hypothesis_evaluation.verdict")),
                ("Supporting reasoning", self._safe_get(experiment, "hypothesis_evaluation.reasoning")),
                ("Research description", context.description),
                ("Success criteria met", self._safe_get(experiment, "hypothesis.success_criteria")),
                ("Implications", self._safe_get(experiment, "validation_result.reasoning")),
                ("Future directions", context.open_questions),
            )
        ]

    @staticmethod
    def _combine_segments(*segments: Tuple[str, Optional[str]]) -> str:
        parts = []
        for label, value in segments:
            if value:
                cleaned = QueryBuilder._clean_text(value)
                if cleaned:
                    parts.append(f"{label}:\n{cleaned}")
        return "\n\n".join(parts)

    @staticmethod
    def _clean_text(value: str) -> str:
        return " ".join(value.strip().split())

    @staticmethod
    def _safe_get(obj: Optional[object], path: str) -> Optional[str]:
        if obj is None:
            return None

        current = obj
        for attr in path.split("."):
            if current is None or not hasattr(current, attr):
                return None
            current = getattr(current, attr)
        if current is None:
            return None
        if isinstance(current, str):
            return current
        if isinstance(current, (list, tuple)):
            return "\n".join(str(item) for item in current if item)
        return str(current)

    @staticmethod
    def _safe_join(items: Optional[Sequence[str]], max_items: int = 5) -> Optional[str]:
        if not items:
            return None
        subset = list(items)[:max_items]
        return "\n".join(subset)

    @staticmethod
    def _join_plot_captions(plots) -> Optional[str]:
        if not plots:
            return None
        captions = [getattr(plot, "caption", "") for plot in plots if getattr(plot, "caption", "")]
        return "\n".join(captions) if captions else None

