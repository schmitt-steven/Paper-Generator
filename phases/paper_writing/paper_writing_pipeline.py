from __future__ import annotations
from typing import List, Sequence, Optional
from phases.context_analysis.paper_conception import PaperConcept
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft, PaperChunk
from phases.paper_writing.paper_indexer import PaperIndexer
from phases.paper_writing.query_builder import QueryBuilder
from phases.paper_writing.paper_writer import PaperWriter
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

        self._indexed_corpus: Optional[List[PaperChunk]] = None

    def index_papers(self, papers: Sequence[Paper]) -> List[PaperChunk]:
        """Index papers into chunk embeddings and cache the result."""

        self._indexed_corpus = self.indexer.index_papers(papers)
        return self._indexed_corpus

    def write_paper(
        self,
        paper_concept: PaperConcept,
        experiment_result: ExperimentResult,
        papers: Sequence[Paper],
    ) -> PaperDraft:
        """Run the full pipeline and return generated paper sections."""

        if not self._indexed_corpus:
            self.index_papers(papers)

        print(f"\n{'='*80}")
        print(f"GATHERING EVIDENCE FOR PAPER SECTIONS")
        print(f"{'='*80}\n")
        
        gatherer = EvidenceGatherer(
            indexed_corpus=self._indexed_corpus or [],
        )

        evidence_by_section: Dict[Section, Sequence[Evidence]] = {}
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
                print(f"[{section_type.value}] Gathering evidence for {section_type.value} section...")
                default_queries = self.query_builder.build_default_queries(section_type, paper_concept, experiment_result)

                evidence, _ = gatherer.gather_evidence(
                    section_type=section_type,
                    context=paper_concept,
                    experiment=experiment_result,
                    default_queries=default_queries,
                    max_iterations=Settings.EVIDENCE_AGENTIC_ITERATIONS,
                    initial_chunks=Settings.EVIDENCE_INITIAL_CHUNKS,
                    filtered_chunks=Settings.EVIDENCE_FILTERED_CHUNKS,
                )

                evidence_by_section[section_type] = evidence

        print(f"\n{'='*80}")
        print(f"WRITING PAPER SECTIONS")
        print(f"{'='*80}\n")
        
        # Load prompts if setting is enabled
        if Settings.LOAD_PAPER_WRITING_PROMPTS:
            writing_prompts = self.load_section_writing_prompts()
            print(f"[PaperWritingPipeline] Using loaded writing prompts for {len(writing_prompts)} sections")
        else:
            writing_prompts = None
        
        paper_draft, generated_prompts = self.writer.generate_paper_sections(
            context=paper_concept,
            experiment=experiment_result,
            evidence_by_section=evidence_by_section,
        )
        
        # Save writing prompts to output directory (use generated if not loaded)
        self._save_prompts(prompts_by_section=generated_prompts if writing_prompts is None else writing_prompts)
        self._save_paper_draft(paper_draft=paper_draft)

        return paper_draft

    @staticmethod
    def _save_prompts(
        prompts_by_section: Dict[str, str],
    ) -> None:
        """Save section writing prompts to a JSON file."""

        output_data = {
            "sections": {
                section_name: {"prompt": prompt}
                for section_name, prompt in prompts_by_section.items()
            }
        }

        output_path = save_json(output_data, "section_writing_prompts.json", "output")

        print(f"[PaperWritingPipeline] Saved section writing prompts to {output_path}")

    @staticmethod
    def load_section_writing_prompts(
        filepath: str = "output/section_writing_prompts.json",
    ) -> Dict[str, str]:
        """Load section writing prompts from a JSON file."""

        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"Section writing prompts file not found: {filepath}")

        data = load_json(path_obj.name, str(path_obj.parent))

        prompts = {
            section_name: section_data["prompt"]
            for section_name, section_data in data.get("sections", {}).items()
        }

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

        paper_draft = PaperDraft(**draft_data)
        print(f"[PaperWritingPipeline] Loaded paper draft from {filepath}")

        return paper_draft

    def reset_index(self) -> None:
        """Reset the cached indexed corpus."""

        self._indexed_corpus = None

