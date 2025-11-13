from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult
from phases.literature_review.arxiv_api import Paper
from phases.paper_writing.evidence_gatherer import EvidenceGatherer
from phases.paper_writing.data_models import Evidence, PaperChunk, PaperDraft, Section
from phases.paper_writing.paper_indexer import PaperIndexer
from phases.paper_writing.paper_writer import PaperWriter
from phases.paper_writing.query_builder import QueryBuilder


class PaperWritingPipeline:
    """Orchestrates indexing, evidence gathering, and section writing."""

    def __init__(
        self,
        writer_model_name: str,
        embedding_model_name: str,
        evidence_model_name: Optional[str] = None,
        top_k_initial: int = 20,
        top_k_final: int = 8,
        agentic_iterations: int = 5,
    ) -> None:
        self.llm_model_name = writer_model_name
        self.embedding_model_name = embedding_model_name
        self.evidence_model_name = evidence_model_name or writer_model_name
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.agentic_iterations = agentic_iterations

        self.indexer = PaperIndexer(embedding_model_name=embedding_model_name)
        self.query_builder = QueryBuilder()
        self.writer = PaperWriter(model_name=writer_model_name)

        self._indexed_corpus: Optional[List[PaperChunk]] = None

    def index_papers(self, papers: Sequence[Paper]) -> List[PaperChunk]:
        """Index papers into chunk embeddings and cache the result."""

        self._indexed_corpus = self.indexer.index_papers(papers)
        return self._indexed_corpus

    def generate_paper(
        self,
        context: PaperConcept,
        experiment: ExperimentResult,
        papers: Sequence[Paper],
    ) -> PaperDraft:
        """Run the full pipeline and return generated paper sections."""

        if not self._indexed_corpus:
            self.index_papers(papers)

        gatherer = EvidenceGatherer(
            llm_model_name=self.evidence_model_name,
            embedding_model_name=self.embedding_model_name,
            indexed_corpus=self._indexed_corpus or [],
        )

        evidence_by_section: Dict[Section, Sequence[Evidence]] = {}
        prompts_by_section: Dict[str, str] = {}
        # Generate sections in order: Methods -> Results -> Discussion -> Introduction -> Conclusion -> Abstract
        for section_type in (
            Section.METHODS,
            Section.RESULTS,
            Section.DISCUSSION,
            Section.INTRODUCTION,
            Section.CONCLUSION,
            Section.ABSTRACT,
        ):
            default_queries = self.query_builder.build_default_queries(section_type, context, experiment)

            evidence, final_prompt = gatherer.gather_evidence(
                section_type=section_type,
                context=context,
                experiment=experiment,
                default_queries=default_queries,
                max_iterations=self.agentic_iterations,
                top_k_initial=self.top_k_initial,
                top_k_final=self.top_k_final,
            )

            evidence_by_section[section_type] = evidence
            prompts_by_section[section_type.value] = final_prompt

        # Automatically save prompts to output directory
        self._save_prompts(prompts_by_section)

        return self.writer.generate_paper_sections(
            context=context,
            experiment=experiment,
            evidence_by_section=evidence_by_section,
        )

    @staticmethod
    def _save_prompts(
        prompts_by_section: Dict[str, str],
    ) -> None:
        """Save section prompts to a JSON file for debugging."""

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "section_prompts.json")

        output_data = {
            "sections": {
                section_name: {"prompt": prompt}
                for section_name, prompt in prompts_by_section.items()
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[PaperWritingPipeline] Saved prompts to {output_path}")

    def reset_index(self) -> None:
        """Reset the cached indexed corpus."""

        self._indexed_corpus = None

