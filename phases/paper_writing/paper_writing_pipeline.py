from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Dict, List, Optional, Sequence

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult
from phases.paper_search.arxiv_api import Paper
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

    def write_paper(
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
        # Generate sections in order: Methods -> Results -> Discussion -> Introduction -> Related Work -> Conclusion -> Abstract
        for section_type in (
            Section.METHODS,
            Section.RESULTS,
            Section.DISCUSSION,
            Section.INTRODUCTION,
            Section.RELATED_WORK,
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

        paper_draft = self.writer.generate_paper_sections(
            context=context,
            experiment=experiment,
            evidence_by_section=evidence_by_section,
        )
        self._save_paper_draft(paper_draft)

        return paper_draft

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

    @staticmethod
    def _save_paper_draft(
        paper_draft: PaperDraft,
        output_dir: str = "output",
        filename: str = "paper_draft.md",
    ) -> None:
        """Save the paper draft as a markdown file."""

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        markdown_content = textwrap.dedent(f"""\
            # {paper_draft.title}

            ## Abstract

            {paper_draft.abstract}

            ## Introduction

            {paper_draft.introduction}

            ## Related Work

            {paper_draft.related_work}

            ## Methods

            {paper_draft.methods}

            ## Results

            {paper_draft.results}

            ## Discussion

            {paper_draft.discussion}

            ## Conclusion

            {paper_draft.conclusion}
        """)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"[PaperWritingPipeline] Saved paper draft to {output_path}")

    @staticmethod
    def load_paper_draft(
        filepath: str = "output/paper_draft.md",
    ) -> PaperDraft:
        """Load a paper draft from a markdown file."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Paper draft file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract title (first # header)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if not title_match:
            raise ValueError("Could not find title in paper draft file")
        title = title_match.group(1).strip()
        
        # Extract sections using regex
        # Pattern matches: ## Section Name followed by content until next ## or end
        section_pattern = r'##\s+(\w+(?:\s+\w+)*)\s*\n\n(.*?)(?=\n##\s+|$)'
        sections = {}
        
        for match in re.finditer(section_pattern, content, re.DOTALL):
            section_name = match.group(1).strip()
            section_content = match.group(2).strip()
            sections[section_name.lower().replace(' ', '_')] = section_content
        
        # Build PaperDraft with extracted sections
        draft_data = {'title': title}
        for field_name in ['abstract', 'introduction', 'related_work', 'methods', 'results', 'discussion', 'conclusion']:
            draft_data[field_name] = sections.get(field_name, '')
        
        paper_draft = PaperDraft(**draft_data)
        print(f"[PaperWritingPipeline] Loaded paper draft from {filepath}")
        
        return paper_draft

    def reset_index(self) -> None:
        """Reset the cached indexed corpus."""

        self._indexed_corpus = None

