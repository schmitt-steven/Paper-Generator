"""Convert PaperDraft to LaTeX project."""

import logging
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List, Optional, Set

from phases.paper_search.arxiv_api import Paper
from phases.paper_writing.data_models import PaperDraft, Section
from phases.latex_generation.metadata import LaTeXMetadata
from phases.latex_generation.markdown_to_latex import MarkdownToLaTeX
from phases.latex_generation.bibliography import generate_literature_bib
from phases.experimentation.experiment_state import ExperimentResult
from utils.lazy_model_loader import LazyModelMixin
from settings import Settings

logger = logging.getLogger(__name__)


class PaperConverter(LazyModelMixin):
    """Converts PaperDraft to compilable LaTeX project."""

    def __init__(self, model_name: str = None):
        """Initialize PaperConverter."""
       
        self.model_name = model_name or Settings.LATEX_GENERATION_MODEL
        self._model = None  # Lazy-loaded via LazyModelMixin

    def convert_to_latex(
        self,
        paper_draft: PaperDraft,
        metadata: LaTeXMetadata,
        indexed_papers: List[Paper],
        experiment_result: Optional[ExperimentResult] = None,
    ) -> Path:
        """Convert PaperDraft to LaTeX project."""

        output_dir = Path("output/latex")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[PaperConverter] Converting PaperDraft to LaTeX")
        
        latex_dir = self._setup_latex_directory(output_dir)
        
        self._populate_metadata(latex_dir, metadata)
        
        self._convert_sections_to_chapters(latex_dir, paper_draft)
        
        self._generate_bibliography(latex_dir, paper_draft, indexed_papers)
        
        self._generate_abbreviations(latex_dir)
        
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
                check=True
            )
            pdf_path = Path("output/result/paper.pdf")
            if pdf_path.exists():
                logger.info(f"[PaperConverter] PDF generated at {pdf_path}")
                return True
            else:
                logger.error(f"[PaperConverter] Compilation succeeded but PDF not found at {pdf_path}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"[PaperConverter] LaTeX compilation failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"[PaperConverter] Error during compilation: {e}")
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
        author_blocks = []
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

    def _convert_sections_to_chapters(self, latex_dir: Path, paper_draft: PaperDraft) -> None:
        """Convert PaperDraft sections to LaTeX chapter files."""
        
        # Map sections to chapter files
        section_mapping = {
            Section.ABSTRACT: "abstract.tex",
            Section.INTRODUCTION: "chapters/introduction.tex",
            Section.RELATED_WORK: "chapters/related_work.tex",
            Section.METHODS: "chapters/methods.tex",
            Section.RESULTS: "chapters/results.tex",
            Section.DISCUSSION: "chapters/discussion.tex",
            Section.CONCLUSION: "chapters/conclusion.tex",
        }
        
        for section_type, filename in section_mapping.items():
            # Get markdown content
            # Map Section enum to PaperDraft attribute names
            attr_map = {
                Section.ABSTRACT: "abstract",
                Section.INTRODUCTION: "introduction",
                Section.RELATED_WORK: "related_work",
                Section.METHODS: "methods",
                Section.RESULTS: "results",
                Section.DISCUSSION: "discussion",
                Section.CONCLUSION: "conclusion",
            }
            attr_name = attr_map[section_type]
            section_content = getattr(paper_draft, attr_name)
            
            if not section_content:
                logger.warning(f"[PaperConverter] Empty section: {section_type.value}")
                continue
            
            # Convert to LaTeX
            logger.info(f"[PaperConverter] Converting {section_type.value} to LaTeX...")
            latex_content = MarkdownToLaTeX.convert_section_to_latex(section_content, section_type, self.model)
            
            # Handle abstract specially - just plain content (paper.tex has \begin{abstract}...\end{abstract})
            if section_type == Section.ABSTRACT:
                # Abstract is already wrapped in \begin{abstract} environment in paper.tex
                # Just write the plain content
                pass
            else:
                # For IEEEtran, use \section instead of \chapter
                if "\\section{" not in latex_content:
                    section_title = section_type.value
                    latex_content = f"\\section{{{section_title}}}\n\\label{{sec:{section_title.lower().replace(' ', '_')}}}\n\n{latex_content}"
            
            # Write to file
            file_path = latex_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(latex_content, encoding="utf-8")
            logger.info(f"[PaperConverter] Wrote {filename}")

    def _generate_bibliography(self, latex_dir: Path, paper_draft: PaperDraft, indexed_papers: List[Paper]) -> None:
        """Generate literature.bib from citations in PaperDraft."""

        # Extract from markdown format (before conversion)
        bib_content = generate_literature_bib(paper_draft, indexed_papers, is_latex=False)
        
        bib_path = latex_dir / "literature.bib"
        bib_path.write_text(bib_content, encoding="utf-8")
        
        logger.info(f"[PaperConverter] Generated literature.bib with {len(bib_content.split('@')) - 1} entries")

    def _generate_abbreviations(self, latex_dir: Path) -> None:
        """Extract abbreviation definitions from LaTeX files and generate abbreviations.tex."""
        definitions: dict[str, tuple[str, str]] = {}  # key -> (abbr, full)
        keys: Set[str] = set()
        
        def scan_file(file_path: Path) -> None:
            if not file_path.exists():
                return
            content = file_path.read_text(encoding="utf-8")
            
            # Extract "Full Form (ABBR)"
            # Pattern: text like "Artificial Intelligence (AI)" or "Machine Learning (ML)"
            pattern = r'([A-Z][^()]*(?:\s+[A-Z][^()]*)*)\s*\(([A-Z]+)\)'
            for match in re.finditer(pattern, content):
                full_form = match.group(1).strip()
                abbr = match.group(2).strip()
                key = abbr.lower()
                
                # Store definition
                if key not in definitions:
                    definitions[key] = (abbr, full_form)
                    keys.add(abbr.upper())
                
                # Replace with \ac{ABBR} (only first occurrence per file)
                # Note: This is a simple approach - might replace multiple times
                new_content = content.replace(match.group(0), f"\\ac{{{abbr}}}", 1)
                if new_content != content:
                    content = new_content
                    file_path.write_text(content, encoding="utf-8")
            
            # Also extract existing \ac{key} patterns
            keys.update(re.findall(r'\\ac(?:s|l|f)?\{([^}]+)\}', content))
        
        # Scan abstract.tex
        scan_file(latex_dir / "abstract.tex")
        
        # Scan all .tex files in chapters directory
        chapters_dir = latex_dir / "chapters"
        if chapters_dir.exists():
            for file_path in chapters_dir.glob("*.tex"):
                scan_file(file_path)
        
        if not keys:
            return
        
        # Generate abbreviations.tex for IEEEtran template (uses acronym package, not glossaries)
        lines = ["% Abbreviations file", "% Automatically generated", ""]
        for key in sorted(keys):
            if key in definitions:
                abbr, full = definitions[key]
            else:
                # Fallback: infer from key
                abbr = key.upper()
                full = key.replace('_', ' ').title()
            full = full.replace("&", "\\&").replace("%", "\\%")
            # Use \acro format for acronym package instead of \newacronym
            lines.append(f"\\acro{{{abbr}}}{{{full}}}")
        
        (latex_dir / "chapters" / "abbreviations.tex").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[PaperConverter] Generated abbreviations.tex with {len(keys)} entries")

    def _copy_plot_images(self, latex_dir: Path, experiment_result: ExperimentResult) -> None:
        """Copy plot images from experiments/plots to LaTeX images directory."""
        
        images_dir = latex_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        plots_dir = Path("output/experiments/plots")
        if not plots_dir.exists():
            logger.info("[PaperConverter] No plots directory found")
            return
        
        # Copy all plot files from plots directory
        plot_extensions = {'.png', '.svg', '.pdf', '.jpg', '.jpeg'}
        copied_count = 0
        for plot_file in plots_dir.iterdir():
            if plot_file.is_file() and plot_file.suffix.lower() in plot_extensions:
                dest_path = images_dir / plot_file.name
                shutil.copy2(plot_file, dest_path)
                copied_count += 1
                logger.debug(f"[PaperConverter] Copied plot: {plot_file.name}")
        
        logger.info(f"[PaperConverter] Copied {copied_count} plot image(s) to images/")

