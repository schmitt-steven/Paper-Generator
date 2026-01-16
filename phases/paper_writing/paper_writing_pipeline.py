from __future__ import annotations
from typing import Sequence, Optional, Callable
from pathlib import Path
import re
from phases.context_analysis.paper_conception import PaperConcept
from phases.context_analysis.user_requirements import UserRequirements
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft, PaperChunk, Section, Evidence
from phases.paper_writing.paper_indexer import PaperIndexer
from phases.paper_writing.paper_writer import PaperWriter
from phases.paper_writing.evidence_gatherer import EvidenceGatherer
from phases.paper_writing.section_critic import SectionCritic
from phases.experimentation.experiment_state import ExperimentResult
from utils.lms_settings import LMSJITSettings
from utils.file_utils import save_markdown, load_markdown
from settings import Settings


class PaperWritingPipeline:
    """Orchestrates critique-based paper writing."""

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
        pattern = r'^# (.+)$'
        parts = re.split(pattern, content, flags=re.MULTILINE)
        
        # parts[0] is content before first header
        # Then alternating: header, content, header, content...
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                section_name = parts[i].strip()
                section_content = parts[i + 1].strip()
                if section_content:  # Only add if there's actual content
                    prompts[section_name] = section_content

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

    def write_paper_with_critique(
        self,
        paper_concept: PaperConcept,
        experiment_result: ExperimentResult,
        papers: Sequence[Paper],
        user_requirements: Optional[UserRequirements] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        max_critique_queries: int = 5,
        chunks_per_query: int = 3,
    ) -> PaperDraft:
        """
        Write paper using the critique-based pipeline.
        
        Flow per section:
        1. Draft v1 using paper catalog (title + abstract + conclusion)
        2. Critique: identify improvements and search queries
        3. Search: batch execute queries for additional evidence
        4. Rewrite: incorporate critique and new evidence
        
        Args:
            paper_concept: The paper concept/context
            experiment_result: Experiment results to incorporate
            papers: Selected papers for citation
            user_requirements: Optional user requirements per section
            status_callback: Callback for status updates
            max_critique_queries: Max search queries from critique (default 5)
            chunks_per_query: Evidence chunks per query (default 3)
            
        Returns:
            PaperDraft with all sections
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
            for section_type in section_order:
                print(f"\n{'─'*60}")
                print(f"[{section_type.value}] Processing section...")
                print(f"{'─'*60}")
                
                # Step 1: Draft v1 using paper catalog
                if status_callback:
                    status_callback(f"Drafting {section_type.value} section")
                print(f"  [Step 1] Writing initial draft using paper catalog...")
                
                # Build and save the prompt
                prompt = self.writer._build_catalog_prompt(
                    section_type=section_type,
                    papers=papers,
                    context=paper_concept,
                    experiment=experiment_result,
                    previous_sections=sections,
                    user_requirements=user_requirements,
                )
                prompts_by_section[section_type.value] = prompt
                
                draft_v1 = self.writer.generate_section_from_catalog(
                    section_type=section_type,
                    papers=papers,
                    context=paper_concept,
                    experiment=experiment_result,
                    previous_sections=sections,
                    user_requirements=user_requirements,
                )
                print(f"    Draft complete ({len(draft_v1)} chars)")
                
                # Step 2: Critique the draft
                if status_callback:
                    status_callback(f"Critiquing {section_type.value} section")
                print(f"  [Step 2] Analyzing draft for improvements...")
                
                critique = critic.critique_section(
                    section_type=section_type,
                    draft_text=draft_v1,
                    papers=papers,
                    max_queries=max_critique_queries,
                    user_requirements=user_requirements,
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
                    original_draft=draft_v1,
                    critique=critique,
                    new_evidence=new_evidence,
                    papers=papers,
                    context=paper_concept,
                    experiment=experiment_result,
                    previous_sections=sections,
                    user_requirements=user_requirements,
                )
                
                sections[section_type] = final_section
                print(f"    Section complete ({len(final_section)} chars)")
                
                # Intermediate save:
                current_title = "Draft in Progress"
                if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
                     current_title = Settings.LATEX_TITLE
                     
                partial_draft = PaperDraft(
                    title=current_title,
                    abstract=sections.get(Section.ABSTRACT, ""),
                    introduction=sections.get(Section.INTRODUCTION, ""),
                    related_work=sections.get(Section.RELATED_WORK, ""),
                    methods=sections.get(Section.METHODS, ""),
                    results=sections.get(Section.RESULTS, ""),
                    discussion=sections.get(Section.DISCUSSION, ""),
                    conclusion=sections.get(Section.CONCLUSION, ""),
                    acknowledgements=None
                )
                self._save_paper_draft(paper_draft=partial_draft)

        # Generate acknowledgements if enabled
        acknowledgements = None
        if Settings.GENERATE_ACKNOWLEDGEMENTS and user_requirements and user_requirements.acknowledgements:
            print("\nWriting Acknowledgements section...")
            acknowledgements = self.writer.generate_acknowledgements(user_requirements.acknowledgements)

        # Generate or use provided title
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title = Settings.LATEX_TITLE
        else:
            title = self.writer.generate_title(
                abstract=sections[Section.ABSTRACT],
                introduction=sections[Section.INTRODUCTION],
                conclusion=sections[Section.CONCLUSION],
                context=paper_concept,
            )

        paper_draft = PaperDraft(
            title=title,
            abstract=sections[Section.ABSTRACT],
            introduction=sections[Section.INTRODUCTION],
            related_work=sections[Section.RELATED_WORK],
            methods=sections[Section.METHODS],
            results=sections[Section.RESULTS],
            discussion=sections[Section.DISCUSSION],
            conclusion=sections[Section.CONCLUSION],
            acknowledgements=acknowledgements,
        )
        
        self._save_paper_draft(paper_draft=paper_draft)
        self._save_prompts(prompts_by_section)
        
        print(f"\n{'='*80}")
        print(f"PAPER WRITING COMPLETE")
        print(f"{'='*80}\n")

        return paper_draft

