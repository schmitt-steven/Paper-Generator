"""Convert PaperDraft to LaTeX project."""

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Any, List, Set
from pathlib import Path
from settings import Settings
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft, Section
from utils.lazy_model_loader import LazyModelMixin
from phases.latex_generation.bibliography import generate_literature_bib
from phases.latex_generation.markdown_to_latex import MarkdownToLaTeX
from phases.experimentation.experiment_state import ExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class LaTeXMetadata:
    """Metadata for LaTeX document generation (IEEEtran format)."""

    title: str
    authors: list[dict[str, str]]  # List of author dictionaries

    @classmethod
    def from_settings(cls, generated_title: str) -> "LaTeXMetadata":
        """Create LaTeXMetadata from settings.
        
        Args:
            generated_title: Title from PaperDraft (respects Settings.LATEX_TITLE if set)
        
        Returns:
            LaTeXMetadata with all fields for IEEEtran template
        """
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
    ) -> Path:
        """Convert PaperDraft to LaTeX project."""

        output_dir = Path("output/latex")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[PaperConverter] Converting PaperDraft to LaTeX")
        
        latex_dir = self._setup_latex_directory(output_dir)
        
        self._populate_metadata(latex_dir, metadata)
        
        self._inject_sections_into_tex(latex_dir, paper_draft)
        
        self._generate_bibliography(latex_dir, paper_draft, indexed_papers)
        

        
        if experiment_result:
            self._copy_plot_images(latex_dir, experiment_result)
        
        logger.info(f"[PaperConverter] LaTeX project generated at {latex_dir}")
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
        
        logger.info(f"[PaperConverter] Loaded existing LaTeX project from {latex_dir}")
        return latex_dir

    def compile_latex(self, latex_dir: Path) -> bool:
        """Compile LaTeX project to PDF using Makefile."""
        try:
            logger.info(f"[PaperConverter] Compiling LaTeX project...")
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
                logger.info(f"[PaperConverter] PDF generated at {pdf_path}")
                return True
            else:
                logger.error(f"[PaperConverter] Compilation succeeded but PDF not found at {pdf_path}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"[PaperConverter] LaTeX compilation timed out after 60 seconds")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"[PaperConverter] LaTeX compilation failed with exit code {e.returncode}")
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
            logger.error(f"[PaperConverter] Error during compilation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _setup_latex_directory(self, output_dir: Path) -> Path:
        """Copy LaTeX template to output directory."""

        template_dir = Path("latex_template/tex")
        
        if not template_dir.exists():
            raise FileNotFoundError(f"LaTeX template not found at {template_dir}")
        
        # Copy entire template directory
        latex_dir = output_dir
        if latex_dir.exists():
            # Remove existing directory to ensure clean copy
            shutil.rmtree(latex_dir)
        
        shutil.copytree(template_dir, latex_dir)
        
        logger.info(f"[PaperConverter] Copied LaTeX template to {latex_dir}")
        return latex_dir

    def _populate_metadata(self, latex_dir: Path, metadata: LaTeXMetadata) -> None:
        """Update paper.tex with metadata for IEEEtran template."""
        
        paper_path = latex_dir / "paper.tex"
        
        if not paper_path.exists():
            logger.error(f"[PaperConverter] paper.tex not found at {paper_path}")
            return
        
        # Read current paper.tex content
        content = paper_path.read_text(encoding="utf-8")
        
        # Replace title
        content = content.replace(
            r"\newcommand{\dokumententitel}[0]{Paper Title}",
            f"\\newcommand{{\\dokumententitel}}[0]{{{metadata.title}}}"
        )
        
        # Generate author blocks for IEEEtran format
        author_blocks: list[str] = []
        for i, author in enumerate(metadata.authors):
            # Build author block
            author_name = author.get("name", "Author Name")
            author_lines = [f"  \\IEEEauthorblockN{{{author_name}}}"]
            
            # Build affiliation block
            affiliation_parts = []
            if author.get("affiliation"):
                affiliation_parts.append(author["affiliation"])
            if author.get("department"):
                affiliation_parts.append(author["department"])
            if author.get("address"):
                affiliation_parts.append(author["address"])
            if author.get("email"):
                affiliation_parts.append(f"Email: {author['email']}")
            
            if affiliation_parts:
                affiliation = "\\\\\n    ".join(affiliation_parts)
                author_lines.append(f"  \\IEEEauthorblockA{{\n    {affiliation}\n    }}")
            
            # Join this author's blocks
            author_block = "\n".join(author_lines)
            author_blocks.append(author_block)
        
        # Join all authors with \and separator (except last one)
        full_author_section = "\n  \\and\n".join(author_blocks)
        
        # Replace the placeholder
        content = content.replace("%%AUTHOR_BLOCKS%%", full_author_section)
        
        # Write updated content
        paper_path.write_text(content, encoding="utf-8")
        logger.info(f"[PaperConverter] Updated paper.tex with {len(metadata.authors)} author(s)")

    def _inject_sections_into_tex(self, latex_dir: Path, paper_draft: PaperDraft) -> None:
        """Convert PaperDraft sections to LaTeX and inject directly into paper.tex."""
        
        paper_path = latex_dir / "paper.tex"
        if not paper_path.exists():
             logger.error(f"[PaperConverter] paper.tex not found at {paper_path}")
             return
             
        # Read the template with placeholders
        paper_content = paper_path.read_text(encoding="utf-8")
        
        # Map sections to placeholders (e.g. Section.INTRODUCTION -> %%INTRODUCTION%%)
        section_mapping = {
            Section.ABSTRACT: "%%ABSTRACT%%",
            Section.INTRODUCTION: "%%INTRODUCTION%%",
            Section.RELATED_WORK: "%%RELATED_WORK%%",
            Section.METHODS: "%%METHODS%%",
            Section.RESULTS: "%%RESULTS%%",
            Section.DISCUSSION: "%%DISCUSSION%%",
            Section.CONCLUSION: "%%CONCLUSION%%",
            Section.ACKNOWLEDGEMENTS: "%%ACKNOWLEDGEMENTS%%",
        }
        
        # Map Section enum to PaperDraft attribute names
        attr_map = {
            Section.ABSTRACT: "abstract",
            Section.INTRODUCTION: "introduction",
            Section.RELATED_WORK: "related_work",
            Section.METHODS: "methods",
            Section.RESULTS: "results",
            Section.DISCUSSION: "discussion",
            Section.CONCLUSION: "conclusion",
            Section.ACKNOWLEDGEMENTS: "acknowledgements",
        }

        for section_type, placeholder in section_mapping.items():
            # Get markdown content
            attr_name = attr_map[section_type]
            section_content = getattr(paper_draft, attr_name, None)
            
            # Handle empty content
            if not section_content:
                if section_type == Section.ACKNOWLEDGEMENTS:
                    logger.info(f"[PaperConverter] Skipping {section_type.value} (not provided)")
                    # Remove placeholder
                    paper_content = paper_content.replace(placeholder, "")
                else:
                    logger.warning(f"[PaperConverter] Empty section: {section_type.value}")
                    # Replace with empty string or comment
                    paper_content = paper_content.replace(placeholder, f"% Empty section: {section_type.value}")
                continue
            
            # Convert to LaTeX
            logger.info(f"[PaperConverter] Converting {section_type.value} to LaTeX...")
            latex_content = MarkdownToLaTeX.convert_section_to_latex(section_content, section_type, self.model)
            
            # Post-processing
            if section_type == Section.ABSTRACT:
                # Abstract is just the content, environment is already in paper.tex
                pass
            else:
                # Append section header if missing
                if "\\section{" not in latex_content and section_type != Section.ACKNOWLEDGEMENTS:
                    section_title = section_type.value
                    latex_content = f"\\section{{{section_title}}}\n\\label{{sec:{section_title.lower().replace(' ', '_')}}}\n\n{latex_content}"
            
            # Inject into paper content
            paper_content = paper_content.replace(placeholder, latex_content)
            
        # Write back the fully fully populated LaTeX file
        paper_path.write_text(paper_content, encoding="utf-8")
        logger.info(f"[PaperConverter] Injected all sections into {paper_path}")

    def _generate_bibliography(self, latex_dir: Path, paper_draft: PaperDraft, indexed_papers: list[Paper]) -> None:
        """Generate literature.bib from citations in PaperDraft."""

        bib_content = generate_literature_bib(paper_draft, indexed_papers)
        
        bib_path = latex_dir / "literature.bib"
        bib_path.write_text(bib_content, encoding="utf-8")
        
        logger.info(f"[PaperConverter] Generated literature.bib with {len(bib_content.split('@')) - 1} entries")



    def _copy_plot_images(self, latex_dir: Path, experiment_result: ExperimentResult) -> None:
        """Copy plot images from experiments/plots to LaTeX images directory."""
        
        images_dir = latex_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        plots_dir = Path("output/experiments/plots")
        if not plots_dir.exists():
            logger.info("[PaperConverter] No plots directory found")
            return
        
        # Copy all plot files from plots directory
        plot_extensions = {'.png', '.pdf', '.jpg', '.jpeg'}
        copied_count = 0
        for plot_file in plots_dir.iterdir():
            if plot_file.is_file() and plot_file.suffix.lower() in plot_extensions:
                dest_path = images_dir / plot_file.name
                shutil.copy2(plot_file, dest_path)
                copied_count += 1
                logger.debug(f"[PaperConverter] Copied plot: {plot_file.name}")
        
        logger.info(f"[PaperConverter] Copied {copied_count} plot image(s) to images/")

