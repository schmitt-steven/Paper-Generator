"""Convert PaperDraft to LaTeX project."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Any, List, Set, Callable
from pathlib import Path
from settings import Settings
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft, Section
from utils.lazy_model_loader import LazyModelMixin
from phases.latex_generation.bibliography import generate_literature_bib
from phases.latex_generation.markdown_to_latex import MarkdownToLaTeX
from phases.experimentation.experiment_state import ExperimentResult
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.paper_search.literature_search import LiteratureSearch
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder


@dataclass
class LaTeXMetadata:
    """Metadata for LaTeX document generation (IEEEtran format)."""

    title: str
    authors: list[dict[str, str]]  # List of author dictionaries

    @classmethod
    def from_settings(cls, generated_title: str) -> "LaTeXMetadata":
        """Create LaTeXMetadata from settings"""
        return cls(
            title=generated_title,
            authors=Settings.LATEX_AUTHORS,
        )


class PaperConverter(LazyModelMixin):
    """Converts PaperDraft to compilable LaTeX project."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize PaperConverter."""
       
        self.model_name = model_name or Settings.LATEX_GENERATION_MODEL
        self._model: Optional[Any] = None  # Lazy-loaded via LazyModelMixin

    def convert_to_latex(
        self,
        paper_draft: PaperDraft,
        metadata: LaTeXMetadata,
        indexed_papers: list[Paper],
        experiment_result: Optional[ExperimentResult] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """Convert PaperDraft to LaTeX project.
        
        Args:
            progress_callback: Optional function(str) to report progress
        """

        output_dir = Path("output/latex")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[PaperConverter] Converting PaperDraft to LaTeX")
        
        latex_dir = self._setup_latex_directory(output_dir)
        
        self._populate_metadata(latex_dir, metadata)
        
        self._inject_sections_into_tex(latex_dir, paper_draft, progress_callback)
        
        if experiment_result:
            self._copy_plot_images(latex_dir, experiment_result)
        
        self._generate_bibliography(latex_dir, paper_draft, indexed_papers)
        
        print(f"[PaperConverter] LaTeX project generated at {latex_dir}")
        return latex_dir

    @staticmethod
    def load_latex(output_dir: str = "output/latex") -> Path:
        """
        Load existing LaTeX project from output directory.
        
        Args:
            output_dir: Path to the LaTeX output directory
            
        Returns:
            Path to the LaTeX directory
            
        Raises:
            FileNotFoundError: If the LaTeX directory or paper.tex doesn't exist
        """
        latex_dir = Path(output_dir)
        
        if not latex_dir.exists():
            raise FileNotFoundError(
                f"LaTeX directory not found at {latex_dir}. "
                f"Set LOAD_LATEX = False to generate it."
            )
        
        paper_tex = latex_dir / "paper.tex"
        if not paper_tex.exists():
            raise FileNotFoundError(
                f"paper.tex not found at {paper_tex}. "
                f"Set LOAD_LATEX = False to generate it."
            )
        
        print(f"[PaperConverter] Loaded existing LaTeX project from {latex_dir}")
        return latex_dir

    def compile_latex(self, latex_dir: Path) -> bool:
        """Compile LaTeX project to PDF using Makefile."""
        try:
            print(f"[PaperConverter] Compiling LaTeX project...")
            result = subprocess.run(
                ["make"],
                cwd=latex_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # 1 minute timeout
            )
            pdf_path = latex_dir / "result" / "paper.pdf"
            if pdf_path.exists():
                print(f"[PaperConverter] PDF generated at {pdf_path}")
                return True
            else:
                print(f"[PaperConverter] Compilation succeeded but PDF not found at {pdf_path}")
                return False
        except subprocess.TimeoutExpired:
            print(f"[PaperConverter] LaTeX compilation timed out after 60 seconds")
            return False
        except subprocess.CalledProcessError as e:
            print(f"[PaperConverter] LaTeX compilation failed with exit code {e.returncode}")
            if e.stdout:
                print(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                print(f"STDERR:\n{e.stderr}")
            # Also check for log files that might have more info
            log_file = latex_dir / "temp" / "paper.log"
            if log_file.exists():
                print(f"\nLast 50 lines of LaTeX log file:")
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines[-50:]:
                        print(line.rstrip())
            return False
        except Exception as e:
            print(f"[PaperConverter] Error during compilation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _setup_latex_directory(self, output_dir: Path) -> Path:
        """Copy LaTeX template to output directory."""

        template_dir = Path(f"latex_templates/{Settings.LATEX_TEMPLATE}")
        
        if not template_dir.exists():
            raise FileNotFoundError(f"LaTeX template not found at {template_dir}")
        
        # Copy entire template directory
        latex_dir = output_dir
        if latex_dir.exists():
            # Remove existing directory to ensure clean copy
            shutil.rmtree(latex_dir)
        
        shutil.copytree(template_dir, latex_dir)
        
        print(f"[PaperConverter] Copied LaTeX template to {latex_dir}")
        return latex_dir

    def _populate_metadata(self, latex_dir: Path, metadata: LaTeXMetadata) -> None:
        """Update paper.tex with metadata using template-defined author format."""
        
        paper_path = latex_dir / "paper.tex"
        
        if not paper_path.exists():
            print(f"[PaperConverter] paper.tex not found at {paper_path}")
            return
        
        # Read current paper.tex content
        content = paper_path.read_text(encoding="utf-8")
        
        # Replace title placeholder (graceful if missing)
        if "%%TITLE%%" in content:
            content = content.replace("%%TITLE%%", metadata.title)
            print(f"[PaperConverter] Set title: {metadata.title[:50]}...")
        else:
            print("[PaperConverter] No %%TITLE%% placeholder found in template, skipping title")
        
        # Extract author block template from paper.tex
        # Template is between %%BEGIN_AUTHOR%% and %%END_AUTHOR%%
        author_pattern = re.compile(
            r'%%BEGIN_AUTHOR%%\s*(.*?)\s*%%END_AUTHOR%%',
            re.DOTALL
        )
        match = author_pattern.search(content)
        
        if match:
            author_template = match.group(1)
            
            # Generate author blocks using template
            author_blocks: list[str] = []
            for author in metadata.authors:
                block = author_template
                # Replace template placeholders with author data
                # Support both IEEE-style and JAIR-style fields
                block = block.replace("{{name}}", author.get("name", "") or "Author")
                block = block.replace("{{affiliation}}", author.get("affiliation", "") or "Institution")
                block = block.replace("{{department}}", author.get("department", ""))
                block = block.replace("{{address}}", author.get("address", ""))
                block = block.replace("{{email}}", author.get("email", "") or "author@institution.com")
                # JAIR-specific fields - country is REQUIRED by acmart
                block = block.replace("{{city}}", author.get("city", "") or "City")
                block = block.replace("{{country}}", author.get("country", "") or "Country")
                block = block.replace("{{state}}", author.get("state", ""))
                author_blocks.append(block.strip())
            
            # Join all authors (JAIR uses separate \author blocks, IEEE uses \and)
            # Check if template uses \and separator or separate author blocks
            if "\\and" in author_template or "\\And" in author_template:
                full_author_section = "\n\\and\n".join(author_blocks)
            else:
                # JAIR-style: each author is a separate block
                full_author_section = "\n\n".join(author_blocks)
            
            # Replace the entire author block (including markers) with formatted authors
            # Use lambda to avoid backslash interpretation in replacement string
            content = author_pattern.sub(lambda m: full_author_section, content)
            print(f"[PaperConverter] Applied template-based author formatting for {len(metadata.authors)} author(s)")
        else:
            print("[PaperConverter] No %%BEGIN_AUTHOR%%...%%END_AUTHOR%% block found in template")
        
        # Handle %%SHORTAUTHORS%% placeholder (used by JAIR template)
        if "%%SHORTAUTHORS%%" in content:
            if metadata.authors:
                # Extract last names
                last_names = []
                for author in metadata.authors:
                    name = author.get("name", "")
                    if name:
                        # Assume last word is last name
                        parts = name.strip().split()
                        if parts:
                            last_names.append(parts[-1])
                
                if len(last_names) == 1:
                    short_authors = last_names[0]
                elif len(last_names) == 2:
                    short_authors = f"{last_names[0]} \\& {last_names[1]}"
                elif len(last_names) > 2:
                    short_authors = f"{last_names[0]} et al."
                else:
                    short_authors = "Author"
                
                content = content.replace("%%SHORTAUTHORS%%", short_authors)
                print(f"[PaperConverter] Set short authors: {short_authors}")
            else:
                content = content.replace("%%SHORTAUTHORS%%", "Author")
        
        # Write updated content
        paper_path.write_text(content, encoding="utf-8")
        print(f"[PaperConverter] Updated paper.tex with metadata")

    def _inject_sections_into_tex(self, latex_dir: Path, paper_draft: PaperDraft, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Convert PaperDraft sections to LaTeX and inject at %%CONTENT%% placeholder."""
        
        paper_path = latex_dir / "paper.tex"
        if not paper_path.exists():
             print(f"[PaperConverter] paper.tex not found at {paper_path}")
             return
             
        # Read the template with %%CONTENT%% placeholder
        paper_content = paper_path.read_text(encoding="utf-8")
        
        # Define section order and attribute mapping
        sections_order = [
            (Section.INTRODUCTION, "introduction"),
            (Section.RELATED_WORK, "related_work"),
            (Section.METHODS, "methods"),
            (Section.RESULTS, "results"),
            (Section.DISCUSSION, "discussion"),
            (Section.CONCLUSION, "conclusion"),
            (Section.ACKNOWLEDGEMENTS, "acknowledgements"),
        ]
        
        # Handle abstract separately (has its own %%ABSTRACT%% placeholder)
        abstract_content = paper_draft.abstract
        if "%%ABSTRACT%%" in paper_content:
            if abstract_content:
                if progress_callback:
                    progress_callback("Converting Abstract to LaTeX")
                latex_abstract = MarkdownToLaTeX.convert_section_to_latex(abstract_content, Section.ABSTRACT, self.model)
                paper_content = paper_content.replace("%%ABSTRACT%%", latex_abstract)
            else:
                paper_content = paper_content.replace("%%ABSTRACT%%", "% No abstract provided")
        else:
            print("[PaperConverter] No %%ABSTRACT%% placeholder found in template, skipping abstract")
        
        # Process each section with its own placeholder
        for section_type, attr_name in sections_order:
            placeholder = f"%%{section_type.name}%%"  # e.g., %%INTRODUCTION%%, %%METHODS%%
            section_content = getattr(paper_draft, attr_name, None)
            

            # Check if placeholder exists in template
            if placeholder not in paper_content:
                if section_type != Section.ACKNOWLEDGEMENTS:
                    print(f"[PaperConverter] No {placeholder} placeholder in template, skipping {section_type.value}")
                else:
                    print(f"[PaperConverter] No {placeholder} placeholder in template for ACKNOWLEDGEMENTS")
                continue
            
            print(f"[PaperConverter] Found placeholder {placeholder}")

            # Handle empty content
            if not section_content:
                if section_type == Section.ACKNOWLEDGEMENTS:
                    # Remove placeholder entirely for missing acknowledgements
                    paper_content = paper_content.replace(placeholder, "")
                    print(f"[PaperConverter] Skipping {section_type.value} (not provided)")
                else:
                    paper_content = paper_content.replace(placeholder, f"% Empty section: {section_type.value}")
                    print(f"[PaperConverter] Empty section: {section_type.value}")
                continue
            
            # Convert to LaTeX
            print(f"[PaperConverter] Converting {section_type.value} to LaTeX (length: {len(section_content)})")
            if progress_callback:
                progress_callback(f"Converting {section_type.value} to LaTeX")
            
            latex_content = MarkdownToLaTeX.convert_section_to_latex(section_content, section_type, self.model)
            
            if not latex_content:
                print(f"[PaperConverter] Conversion returned empty string for {section_type.value}!")
                # Keep placeholder but verify why content was lost
            
            # Special handling for Acknowledgements: Wrap with environment/header if needed
            if section_type == Section.ACKNOWLEDGEMENTS and latex_content.strip():
                if Settings.LATEX_TEMPLATE == "jair":
                    latex_content = f"\\begin{{acks}}\n{latex_content}\n\\end{{acks}}"
                elif Settings.LATEX_TEMPLATE == "ieee_transaction":
                    latex_content = f"\\section{{Acknowledgements}}\n{latex_content}"
                else:
                    # Default fallback
                    latex_content = f"\\section{{Acknowledgements}}\n{latex_content}"
            
            # Inject into placeholder
            paper_content = paper_content.replace(placeholder, latex_content)
            print(f"[PaperConverter] Injected {len(latex_content)} chars into {placeholder}")
        
        # Write back the fully populated LaTeX file
        paper_path.write_text(paper_content, encoding="utf-8")
        print(f"[PaperConverter] Injected all sections into {paper_path}")

    def _generate_bibliography(self, latex_dir: Path, paper_draft: PaperDraft, indexed_papers: list[Paper]) -> None:
        """Generate bibliography.bib from citations in PaperDraft."""

        bib_content = generate_literature_bib(paper_draft, indexed_papers)
        
        bib_path = latex_dir / "bibliography.bib"
        bib_path.write_text(bib_content, encoding="utf-8")
        
        print(f"[PaperConverter] Generated bibliography.bib with {len(bib_content.split('@')) - 1} entries")



    def _copy_plot_images(self, latex_dir: Path, experiment_result: ExperimentResult) -> None:
        """Copy plot images from experiments/plots to LaTeX images directory."""
        
        images_dir = latex_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        plots_dir = Path("output/experiments/plots")
        if not plots_dir.exists():
            print("[PaperConverter] No plots directory found")
            return
        
        # Copy all plot files from plots directory
        plot_extensions = {'.png', '.pdf', '.jpg', '.jpeg'}
        copied_count = 0
        for plot_file in plots_dir.iterdir():
            if plot_file.is_file() and plot_file.suffix.lower() in plot_extensions:
                dest_path = images_dir / plot_file.name
                shutil.copy2(plot_file, dest_path)
                copied_count += 1
                print(f"[PaperConverter] Copied plot: {plot_file.name}")
        
        print(f"[PaperConverter] Copied {copied_count} plot image(s) to images/")

    @staticmethod
    def generate_new_pdf(status_callback: callable = None) -> bool:
        """
        Generate LaTeX project and compile to PDF.
        
        This handles:
        1. Loading paper draft
        2. Loading indexed papers
        3. Loading experiment result (optional)
        4. Loading hypothesis for metadata
        5. Converting to LaTeX
        6. Compiling to PDF
        
        Args:
            status_callback: Optional callback function(str) for progress updates.
            
        Returns:
            True if PDF was successfully generated, False otherwise.
        """
        
        if status_callback:
            status_callback("Loading paper draft")
        paper_draft = PaperWritingPipeline.load_paper_draft("output/paper_draft.md")
        
        if status_callback:
            status_callback("Loading indexed papers")
        indexed_papers = LiteratureSearch.load_papers("output/papers.json")
        
        if status_callback:
            status_callback("Loading experiment results")
        experiment_result = None
        experiment_result_file = "output/experiments/experiment_result.json"
        if Path(experiment_result_file).exists():
            experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
        
        if status_callback:
            status_callback("Loading hypothesis")
        selected_hypothesis = HypothesisBuilder.load_hypothesis("output/hypothesis.md")
        if selected_hypothesis is None:
            raise ValueError("No hypothesis found")
        
        metadata = LaTeXMetadata.from_settings(generated_title=paper_draft.title)
        
        if status_callback:
            status_callback("Generating LaTeX project")
        converter = PaperConverter()
        latex_dir = converter.convert_to_latex(
            paper_draft=paper_draft,
            metadata=metadata,
            indexed_papers=indexed_papers,
            experiment_result=experiment_result,
            progress_callback=status_callback
        )
        
        if status_callback:
            status_callback("Compiling LaTeX to PDF")
        return converter.compile_latex(latex_dir)

