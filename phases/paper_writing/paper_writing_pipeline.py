from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Callable
from pathlib import Path
import re
from phases.context_analysis.paper_conception import PaperConcept
from phases.context_analysis.user_requirements import UserRequirements
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft, PaperChunk, Section, Evidence
from phases.paper_writing.paper_indexer import PaperIndexer
from phases.paper_writing.query_builder import QueryBuilder
from phases.paper_writing.paper_writer import PaperWriter
from phases.paper_writing.evidence_gatherer import EvidenceGatherer
from phases.paper_writing.evidence_manager import save_evidence, load_evidence
from phases.experimentation.experiment_state import ExperimentResult
from utils.lms_settings import LMSJITSettings
from utils.file_utils import save_json, load_json, save_markdown, load_markdown
from settings import Settings


class PaperWritingPipeline:
    """Orchestrates indexing, evidence gathering, and section writing."""

    def __init__(self) -> None:
        self.indexer = PaperIndexer()
        self.query_builder = QueryBuilder()
        self.writer = PaperWriter()

        self._indexed_corpus: Optional[list[PaperChunk]] = None

    def index_papers(self, papers: Sequence[Paper]) -> list[PaperChunk]:
        """Index papers into chunk embeddings and cache the result."""

        self._indexed_corpus = self.indexer.index_papers(papers)
        return self._indexed_corpus

    def write_paper(
        self,
        paper_concept: PaperConcept,
        experiment_result: ExperimentResult,
        papers: Sequence[Paper],
        user_requirements: Optional[UserRequirements] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> PaperDraft:
        """Run the full pipeline and return generated paper sections."""

        if not self._indexed_corpus:
            if status_callback:
                status_callback("Generating embeddings for papers...")
            self.index_papers(papers)

        print(f"\n{'='*80}")
        print(f"GATHERING EVIDENCE FOR PAPER SECTIONS")
        print(f"{'='*80}\n")
        
        gatherer = EvidenceGatherer(
            indexed_corpus=self._indexed_corpus or [],
        )

        evidence_by_section: dict[Section, Sequence[Evidence]] = {}
        # Generate sections in order: Methods -> Results -> Discussion -> Introduction -> Related Work -> Conclusion -> Abstract
        
        # Use context manager to ensure multiple models can be loaded for ALL sections
        with LMSJITSettings():
            for section_type in (
                Section.METHODS,
                Section.RESULTS,
                Section.DISCUSSION,
                Section.INTRODUCTION,
                Section.RELATED_WORK,
                Section.CONCLUSION,
                # Section.ABSTRACT,
            ):
                print(f"[{section_type.value}] Gathering evidence for {section_type.value} section")
                if status_callback:
                    status_callback(f"Gathering evidence for {section_type.value} section")
                default_queries = self.query_builder.build_default_queries(section_type, paper_concept, experiment_result)

                evidence, _ = gatherer.gather_evidence(
                    section_type=section_type,
                    context=paper_concept,
                    experiment=experiment_result,
                    default_queries=default_queries,
                    max_iterations=Settings.EVIDENCE_AGENTIC_ITERATIONS,
                    initial_chunks=Settings.EVIDENCE_INITIAL_CHUNKS,
                    filtered_chunks=Settings.EVIDENCE_FILTERED_CHUNKS,
                    user_requirements=user_requirements,
                )

                evidence_by_section[section_type] = evidence

        # Save evidence for the Evidence Manager screen
        save_evidence(evidence_by_section)

        print(f"\n{'='*80}")
        print(f"WRITING PAPER SECTIONS")
        print(f"{'='*80}\n")
        
        # Load prompts if setting is enabled
        # Try to load existing prompts to avoid re-generation
        writing_prompts = None
        try:
            writing_prompts = self.load_section_writing_prompts()
            print(f"[PaperWritingPipeline] Using loaded writing prompts for {len(writing_prompts)} sections")
        except FileNotFoundError:
            print(f"[PaperWritingPipeline] Prompts file not found. Generating new prompts.")
            writing_prompts = None
        
        if status_callback:
            status_callback("Drafting paper sections...")

        paper_draft, generated_prompts = self.writer.generate_paper_sections(
            context=paper_concept,
            experiment=experiment_result,
            evidence_by_section=evidence_by_section,
            user_requirements=user_requirements,
            writing_prompts=writing_prompts,
        )
        
        # Save writing prompts to output directory (use generated if not loaded)
        self._save_prompts(prompts_by_section=generated_prompts if writing_prompts is None else writing_prompts)
        self._save_paper_draft(paper_draft=paper_draft)

        return paper_draft

    @staticmethod
    def _save_prompts(
        prompts_by_section: dict[str, str],
        filename: str = "section_writing_prompts.md",
        output_dir: str = "output"
    ) -> None:
        """Save section writing prompts to a Markdown file."""
        
        content_parts = []
        
        # Sort by section order logic if possible, otherwise alphabetical or just iteration order
        # Iteration order is usually preserving insertion order in modern Python, which works for us
        
        for section_name, prompt in prompts_by_section.items():
             content_parts.append(f"# {section_name}\n\n{prompt.strip()}\n")
             
        markdown_content = "\n".join(content_parts)
        
        output_path = save_markdown(markdown_content, filename, output_dir)

        print(f"[PaperWritingPipeline] Saved section writing prompts to {output_path}")

    @staticmethod
    def load_section_writing_prompts(
        filepath: str = "output/section_writing_prompts.md",
    ) -> dict[str, str]:
        """Load section writing prompts from a Markdown file."""

        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"Section writing prompts file not found: {filepath}")

        content = load_markdown(path_obj.name, str(path_obj.parent))

        prompts = {}
        # Pattern: # Section Name followed by content until next # or end
        # Note: Section headers are Level 1 (#)
        section_pattern = r'^#\s+(.+?)\s*\n(.*?)(?=\n#\s+|$)'
        
        for match in re.finditer(section_pattern, content, re.DOTALL | re.MULTILINE):
            section_name = match.group(1).strip()
            prompt_content = match.group(2).strip()
            prompts[section_name] = prompt_content

        print(f"[PaperWritingPipeline] Loaded {len(prompts)} section writing prompts from {filepath}")
        return prompts

    @staticmethod
    def _save_paper_draft(
        paper_draft: PaperDraft,
        output_dir: str = "output",
        filename: str = "paper_draft.md",
    ) -> None:
        """Save the paper draft as a markdown file."""

        markdown_content = f"# {paper_draft.title}\n\n"
        markdown_content += f"## Abstract\n\n{paper_draft.abstract}\n\n"
        markdown_content += f"## Introduction\n\n{paper_draft.introduction}\n\n"
        markdown_content += f"## Related Work\n\n{paper_draft.related_work}\n\n"
        markdown_content += f"## Methods\n\n{paper_draft.methods}\n\n"
        markdown_content += f"## Results\n\n{paper_draft.results}\n\n"
        markdown_content += f"## Discussion\n\n{paper_draft.discussion}\n\n"
        markdown_content += f"## Conclusion\n\n{paper_draft.conclusion}\n"
        
        if paper_draft.acknowledgements:
            markdown_content += f"\n## Acknowledgements\n\n{paper_draft.acknowledgements}\n"

        output_path = save_markdown(markdown_content, filename, output_dir)

        print(f"[PaperWritingPipeline] Saved paper draft to {output_path}")

    @staticmethod
    def load_paper_draft(
        filepath: str = "output/paper_draft.md",
    ) -> PaperDraft:
        """Load a paper draft from a markdown file."""

        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"Paper draft file not found: {filepath}")

        content = load_markdown(path_obj.name, str(path_obj.parent))

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
        
        # Handle acknowledgements (optional field)
        acknowledgements_content = sections.get('acknowledgements', '')
        if acknowledgements_content:
            draft_data['acknowledgements'] = acknowledgements_content

        paper_draft = PaperDraft(**draft_data)
        print(f"[PaperWritingPipeline] Loaded paper draft from {filepath}")

        return paper_draft

    def reset_index(self) -> None:
        """Reset the cached indexed corpus."""

        self._indexed_corpus = None

    def write_paper_from_evidence(
        self,
        paper_concept: PaperConcept,
        experiment_result: ExperimentResult,
        user_requirements: Optional[UserRequirements] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        evidence_file: str = "output/evidence.json",
    ) -> PaperDraft:
        """
        Write paper using pre-edited evidence from JSON file.
        
        This method skips evidence gathering and uses evidence that was
        previously gathered and potentially edited by the user via the
        Evidence Manager screen.
        """
        print(f"\n{'='*80}")
        print(f"LOADING EDITED EVIDENCE")
        print(f"{'='*80}\n")
        
        if status_callback:
            status_callback("Loading edited evidence...")
        
        # Load evidence from file (user may have added/removed chunks)
        try:
            evidence_by_section = load_evidence(evidence_file)
            total_chunks = sum(len(ev) for ev in evidence_by_section.values())
            print(f"[PaperWritingPipeline] Loaded {total_chunks} evidence chunks from {evidence_file}")
        except FileNotFoundError:
            raise ValueError(f"Evidence file not found: {evidence_file}. Please run evidence gathering first.")
        
        print(f"\n{'='*80}")
        print(f"WRITING PAPER SECTIONS")
        print(f"{'='*80}\n")
        
        # Load prompts if setting is enabled
        writing_prompts = None
        try:
            writing_prompts = self.load_section_writing_prompts()
            print(f"[PaperWritingPipeline] Using loaded writing prompts for {len(writing_prompts)} sections")
        except FileNotFoundError:
            print(f"[PaperWritingPipeline] Prompts file not found. Generating new prompts.")
            writing_prompts = None
        
        if status_callback:
            status_callback("Drafting paper sections...")

        paper_draft, generated_prompts = self.writer.generate_paper_sections(
            context=paper_concept,
            experiment=experiment_result,
            evidence_by_section=evidence_by_section,
            user_requirements=user_requirements,
            writing_prompts=writing_prompts,
        )
        
        # Save writing prompts to output directory
        self._save_prompts(prompts_by_section=generated_prompts if writing_prompts is None else writing_prompts)
        self._save_paper_draft(paper_draft=paper_draft)

        return paper_draft
