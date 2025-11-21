"""Convert PaperDraft to LaTeX project."""

import logging
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Set

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

    def __init__(self, model_name: Optional[str] = None):
        """Initialize PaperConverter."""
       
        self.model_name = model_name or Settings.LATEX_GENERATION_MODEL
        self._model: Optional[Any] = None  # Lazy-loaded via LazyModelMixin

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
        
        self._generate_abbreviations(latex_dir, paper_draft)
        
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
            pdf_path = latex_dir / "result" / "paper.pdf"
            if pdf_path.exists():
                logger.info(f"[PaperConverter] PDF generated at {pdf_path}")
                return True
            else:
                logger.error(f"[PaperConverter] Compilation succeeded but PDF not found at {pdf_path}")
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
        author_blocks: List[str] = []
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
            print(f"[PaperConverter] Converting {section_type.value} to LaTeX...")
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

    def _generate_abbreviations(self, latex_dir: Path, paper_draft: PaperDraft) -> None:
        """Extract abbreviation definitions from paper draft and generate abbreviations.tex."""
        definitions: dict[str, tuple[str, str]] = {}  # key -> (abbr, full)
        keys: Set[str] = set()
        
        def extract_abbrevs_from_text(text: str) -> None:
            """Extract "Full Form (ABBR)" patterns from text."""
            if not text:
                return
            
            # Clean up malformed nested patterns first
            # "Full Form (Full Form (ABBR))" -> "Full Form (ABBR)"
            text = re.sub(
                r'([A-Z][A-Za-z\s\-]+?)\s*\([^\)]*\1[^\)]*\(([A-Z]{2,})\)\)',
                r'\1 (\2)',
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
            
            # Match: capitalized phrase followed by (2+ capital letters)
            pattern = r'([A-Z][A-Za-z\s\-]+?)\s*\(([A-Z]{2,})\)'
            
            for match in re.finditer(pattern, text):
                full_form = match.group(1).strip()
                abbr = match.group(2).strip().upper()
                key = abbr.lower()
                
                # Remove leading unwanted words
                words = full_form.split()
                unwanted_starts = {'unlike', 'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                                'such', 'some', 'any', 'all', 'each', 'every', 'both'}
                
                while words and words[0].lower() in unwanted_starts:
                    words.pop(0)
                
                # Skip leading lowercase words
                while words and words[0][0].islower():
                    words.pop(0)
                
                if not words:
                    continue
                
                full_form = ' '.join(words)
                
                # Store first occurrence only
                if key not in definitions:
                    definitions[key] = (abbr, full_form)
                    keys.add(abbr)
        
        # Extract from markdown sections
        for section_content in [
            paper_draft.abstract,
            paper_draft.introduction,
            paper_draft.related_work,
            paper_draft.methods,
            paper_draft.results,
            paper_draft.discussion,
            paper_draft.conclusion,
        ]:
            extract_abbrevs_from_text(section_content)
        
        # Scan LaTeX files for additional abbreviations
        def scan_latex_file(file_path: Path) -> None:
            if not file_path.exists():
                return
            
            content = file_path.read_text(encoding="utf-8")
            extract_abbrevs_from_text(content)
            
            # Extract any existing \ac{KEY} usage
            found_keys = re.findall(r'\\ac(?:s|l|f|p)?\{([^}]+)\}', content)
            for key in found_keys:
                keys.add(key.upper())
        
        # Scan all LaTeX files
        scan_latex_file(latex_dir / "abstract.tex")
        chapters_dir = latex_dir / "chapters"
        if chapters_dir.exists():
            for file_path in chapters_dir.glob("*.tex"):
                if file_path.name != "abbreviations.tex":
                    scan_latex_file(file_path)
        
        if not keys:
            logger.warning("[PaperConverter] No abbreviations found")
            return
        
        # Generate abbreviations.tex
        lines = [
            "% Abbreviations file",
            "% Automatically generated",
            ""
        ]
        
        for key in sorted(keys):
            key_lower = key.lower()
            
            if key_lower in definitions:
                abbr, full = definitions[key_lower]
            else:
                abbr = key.upper()
                full = key.replace('_', ' ').title()
            
            # Escape LaTeX special characters
            full = full.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            lines.append(f"\\acro{{{abbr}}}{{{full}}}")
        
        abbrev_file = latex_dir / "chapters" / "abbreviations.tex"
        abbrev_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[PaperConverter] Generated abbreviations.tex with {len(keys)} entries")
        
        # Convert abbreviations in LaTeX files to \ac{} format
        self._convert_abbreviations_to_ac(latex_dir, keys, definitions)


    def _convert_abbreviations_to_ac(self, latex_dir: Path, keys: Set[str], definitions: dict[str, tuple[str, str]]) -> None:
        """Convert all abbreviation mentions to \ac{} format - LaTeX handles first occurrence expansion."""
        
        def process_file(file_path: Path) -> None:
            if not file_path.exists():
                return
                
            content = file_path.read_text(encoding="utf-8")
            original_content = content
            
            # Clean up any malformed nested patterns from source
            for abbr in sorted(keys, key=len, reverse=True):
                key_lower = abbr.lower()
                if key_lower not in definitions:
                    continue
                
                known_full_form = definitions[key_lower][1]
                escaped_full = re.escape(known_full_form)
                escaped_abbr = re.escape(abbr)
                
                # Remove nested duplicates: "Full Form (Full Form (ABBR))" -> "Full Form (ABBR)"
                nested_pattern = rf'{escaped_full}\s*\([^\)]*{escaped_full}[^\)]*\({escaped_abbr}\)\)'
                content = re.sub(
                    nested_pattern, 
                    rf'{known_full_form} ({abbr})',
                    content,
                    flags=re.IGNORECASE | re.DOTALL
                )
            
            # Convert "Full Form (ABBR)" -> \ac{ABBR}
            # The \ac{} command will automatically expand to "Full Form (ABBR)" on first use
            for abbr in sorted(keys, key=len, reverse=True):
                key_lower = abbr.lower()
                if key_lower not in definitions:
                    continue
                
                known_full_form = definitions[key_lower][1]
                escaped_full = re.escape(known_full_form)
                escaped_abbr = re.escape(abbr)
                
                # Match "Full Form (ABBR)" but NOT if already \ac{ABBR}
                pattern = rf'{escaped_full}\s*\((?!\\ac\{{){escaped_abbr}\)'
                content = re.sub(pattern, rf'\\ac{{{abbr}}}', content, flags=re.IGNORECASE)
            
            # Convert standalone "ABBR" -> \ac{ABBR}
            for abbr in sorted(keys, key=len, reverse=True):
                escaped_abbr = re.escape(abbr)
                
                # Match standalone abbreviation, but not if already in \ac{}
                pattern = rf'(?<!\\ac\{{)(?<!\\)\b{escaped_abbr}\b(?!\}})'
                
                def safe_replace(match):
                    pos = match.start()
                    
                    # Check if already inside \ac{...}
                    before = content[max(0, pos-10):pos]
                    if '\\ac{' in before and not ')' in before[before.rfind('\\ac{'):]:
                        return match.group(0)
                    
                    after = content[pos+len(abbr):min(len(content), pos+len(abbr)+10)]
                    if after.startswith('}'):
                        return match.group(0)
                        
                    return rf'\ac{{{abbr}}}'
                
                content = re.sub(pattern, safe_replace, content)
            
            if content != original_content:
                file_path.write_text(content, encoding="utf-8")
                logger.info(f"[PaperConverter] Converted abbreviations to \\ac{{}} in {file_path.name}")
        
        # Process all chapter files
        chapters_dir = latex_dir / "chapters"
        if chapters_dir.exists():
            for file_path in chapters_dir.glob("*.tex"):
                if file_path.name != "abbreviations.tex":
                    process_file(file_path)
        
        process_file(latex_dir / "abstract.tex")

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

