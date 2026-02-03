from __future__ import annotations
from typing import Sequence, Optional, Callable
from pathlib import Path
import re
from phases.context_analysis.research_context_generator import ResearchContext
from phases.context_analysis.paper_specification import PaperSpecification
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft, PaperChunk, Section, Evidence
from phases.paper_writing.paper_indexer import PaperIndexer
from phases.paper_writing.paper_writer import PaperWriter
from phases.paper_writing.evidence_gatherer import EvidenceGatherer
from phases.paper_writing.section_critic import SectionCritic
from phases.experimentation.experiment_state import ExperimentResult
from utils.lms_settings import LMSJITSettings
from utils.file_utils import save_markdown, load_markdown, save_json, load_json
from settings import Settings
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.paper_search.literature_search import LiteratureSearch


class PaperWritingPipeline:
    """Orchestrates the entire paper writing process."""

    def __init__(self) -> None:
        self.indexer = PaperIndexer()
        self.writer = PaperWriter()

        self._indexed_corpus: Optional[list[PaperChunk]] = None

    def index_papers(self, papers: Sequence[Paper]) -> list[PaperChunk]:
        """Index papers into chunk embeddings and cache the result."""

        self._indexed_corpus = self.indexer.index_papers(papers)
        return self._indexed_corpus

    @staticmethod
    def _save_prompts(
        prompts_by_section: dict[str, str],
        filename: str = "section_writing_prompts.json",
        output_dir: str = "output"
    ) -> None:
        """Save section writing prompts to a JSON file."""
        
        # Save as JSON directly (preserving dictionary structure)
        output_path = save_json(prompts_by_section, filename, output_dir)

        print(f"[PaperWritingPipeline] Saved section writing prompts to {output_path}")

    @staticmethod
    def load_section_writing_prompts(
        filepath: str = "output/section_writing_prompts.json",
    ) -> dict[str, str]:
        """Load section writing prompts from a JSON file."""

        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"Section writing prompts file not found: {filepath}")

        # Load from JSON
        prompts = load_json(path_obj.name, str(path_obj.parent))
        
        # Ensure it's a dict
        if not isinstance(prompts, dict):
             # If it wrapped in a list or something, try to handle or fail
             raise ValueError(f"Expected dict from prompts file, got {type(prompts)}")

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

    def write_paper(
        self,
        research_context: ResearchContext,
        experiment_result: ExperimentResult,
        papers: Sequence[Paper],
        paper_specification: Optional[PaperSpecification] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        max_critique_queries: int = 5,  # Num of search suggestions/queries the critique generates
        chunks_per_query: int = 5,  # Num of kept chunks per query
        max_chunks_per_paper: int = 2,  # Max num of chunks from the same paper for a query
    ) -> PaperDraft:
        """
        Writes a paper in markdown format.
        
        Flow per section:
        1. Draft v1 using paper catalog (title + abstract + conclusion)
        2. Critique: identify improvements and generate search queries
        3. Search: execute queries for additional evidence
        4. Rewrite: incorporate critique and new evidence
        """
        # Index papers for critique-based evidence search
        if not self._indexed_corpus:
            if status_callback:
                status_callback("Generating embeddings for papers")
            self.index_papers(papers)

        print(f"\n{'='*80}")
        print(f"PAPER WRITING PIPELINE")
        print(f"{'='*80}\n")

        critic = SectionCritic()
        gatherer = EvidenceGatherer(indexed_corpus=self._indexed_corpus or [])
        
        section_order = (
            Section.METHODS, Section.RESULTS, Section.DISCUSSION,
            Section.INTRODUCTION, Section.RELATED_WORK, Section.CONCLUSION, Section.ABSTRACT
        )
        
        sections: dict[Section, str] = {}
        evidence_by_section: dict[Section, Sequence[Evidence]] = {}
        prompts_by_section: dict[str, str] = {}
        
        with LMSJITSettings():
            for idx, section_type in enumerate(section_order):
                # Identify next section for forward look
                next_section_type = section_order[idx + 1] if idx + 1 < len(section_order) else None

                print(f"\n{'─'*60}")
                print(f"[{section_type.value}] Processing section...")
                print(f"{'─'*60}")
                
                # Step 1: Draft v1 using paper catalog
                if status_callback:
                    status_callback(f"Drafting {section_type.value} section")
                print(f"  [Step 1] Writing initial draft using paper catalog...")
                
                # Build and save the prompt
                prompt = self.writer._build_initial_section_prompt(
                    section_type=section_type,
                    papers=papers,
                    context=research_context,
                    experiment=experiment_result,
                    previous_sections=sections,
                    paper_specification=paper_specification,
                    next_section_type=next_section_type,
                )
                prompts_by_section[section_type.value] = prompt
                
                section_draft_v1 = self.writer.generate_initial_section(
                    section_type=section_type,
                    papers=papers,
                    context=research_context,
                    experiment=experiment_result,
                    previous_sections=sections,
                    paper_specification=paper_specification,
                    next_section_type=next_section_type,
                )
                print(f"    Draft complete ({len(section_draft_v1)} chars)")
                
                # Step 2: Critique the draft
                if status_callback:
                    status_callback(f"Critiquing {section_type.value} section")
                print(f"  [Step 2] Analyzing draft for improvements...")
                
                critique = critic.critique_section(
                    section_type=section_type,
                    draft_text=section_draft_v1,
                    papers=papers,
                    max_queries=max_critique_queries,
                    paper_specification=paper_specification,
                )
                print(f"    Critique: {len(critique.improvements)} chars, {len(critique.search_queries)} queries")
                
                # Step 3: Batch search for additional evidence
                new_evidence: list[Evidence] = []
                
                # Skip search for sections that don't need external evidence (Abstract, Conclusion, Acknowledgements)
                skip_search = section_type in [Section.ABSTRACT, Section.CONCLUSION, Section.ACKNOWLEDGEMENTS]
                
                if critique.search_queries and not skip_search:
                    if status_callback:
                        status_callback(f"Searching evidence for {section_type.value}")
                    print(f"  [Step 3] Searching for additional evidence...")
                    
                    new_evidence = gatherer.batch_search(
                        queries=critique.search_queries,
                        section_type=section_type,
                        chunks_per_query=chunks_per_query,
                        max_chunks_per_paper=max_chunks_per_paper,
                    )
                else:
                    reason = "skipped by policy" if skip_search else "no queries suggested"
                    print(f"  [Step 3] Skipping search ({reason})")
                
                evidence_by_section[section_type] = new_evidence
                
                # Step 4: Rewrite with critique and new evidence
                if status_callback:
                    status_callback(f"Rewriting {section_type.value} section")
                print(f"  [Step 4] Rewriting section with improvements and evidence...")
                
                final_section = self.writer.rewrite_section(
                    section_type=section_type,
                    text_to_rewrite=section_draft_v1,
                    critique=critique,
                    new_evidence=new_evidence,
                    papers=papers,
                    context=research_context,
                    experiment=experiment_result,
                    previous_sections=sections,
                    paper_specification=paper_specification,
                    next_section_type=next_section_type,
                )
                
                sections[section_type] = final_section
                print(f"    Section complete ({len(final_section)} chars)")

        # Generate acknowledgements if enabled
        acknowledgements = None
        if Settings.GENERATE_ACKNOWLEDGEMENTS and paper_specification and paper_specification.acknowledgements:
            print("\nWriting Acknowledgements section...")
            acknowledgements = self.writer.generate_acknowledgements(paper_specification.acknowledgements)

        # Create draft (title will be set below)
        paper_draft = PaperDraft(
            title="",
            abstract=sections[Section.ABSTRACT],
            introduction=sections[Section.INTRODUCTION],
            related_work=sections[Section.RELATED_WORK],
            methods=sections[Section.METHODS],
            results=sections[Section.RESULTS],
            discussion=sections[Section.DISCUSSION],
            conclusion=sections[Section.CONCLUSION],
            acknowledgements=acknowledgements,
        )

        # Use settings title if provided, otherwise generate one
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            paper_draft.title = Settings.LATEX_TITLE
        else:
            paper_draft.title = self.writer.generate_title(draft=paper_draft, context=research_context)
        
        self._save_paper_draft(paper_draft=paper_draft)
        self._save_prompts(prompts_by_section)
        
        print(f"\n{'='*80}")
        print(f"PAPER WRITING COMPLETE")
        print(f"{'='*80}\n")

        return paper_draft

    @staticmethod
    def generate_new_draft(status_callback: callable = None) -> "PaperDraft":
        """
        Generate paper draft.
        
        This handles:
        1. Loading research context
        2. Loading experiment result
        3. Loading indexed papers
        4. Loading paper specification (optional)
        5. Running the paper writing pipeline
        6. Saving the result
        
        Args:
            status_callback: Optional callback function(str) for progress updates.
            
        Returns:
            The generated PaperDraft object.
        """
        
        if status_callback:
            status_callback("Loading resources")
        research_context = ResearchContextGenerator.load_research_context("output/research_context.md")
        
        # Check experiment result exists
        experiment_result_file = "output/experiments/experiment_result.json"
        from pathlib import Path
        if not Path(experiment_result_file).exists():
            raise ValueError("No experiment results found. Please run experiments first.")
        experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
        
        papers = LiteratureSearch.load_papers("output/papers.json")
        
        paper_specification = None
        try:
            paper_specification = PaperSpecification.load("user_files/paper_specification.md")
        except:
            pass
        
        pipeline = PaperWritingPipeline()
        return pipeline.write_paper(
            research_context=research_context,
            experiment_result=experiment_result,
            papers=papers,
            paper_specification=paper_specification,
            status_callback=status_callback
        )

