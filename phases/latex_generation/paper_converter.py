"""Convert PaperDraft to LaTeX project."""

import logging
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List, Optional, Set

import lmstudio as lms
from phases.paper_search.arxiv_api import Paper
from phases.paper_writing.data_models import PaperDraft, Section
from phases.latex_generation.metadata import LaTeXMetadata
from phases.latex_generation.markdown_to_latex import MarkdownToLaTeX
from phases.latex_generation.bibliography import generate_literature_bib
from phases.experimentation.experiment_state import ExperimentResult
from settings import Settings

logger = logging.getLogger(__name__)


class PaperConverter:
    """Converts PaperDraft to compilable LaTeX project."""

    def __init__(self, model_name: str = None):
        """Initialize PaperConverter."""
       
        self.model_name = model_name or Settings.LATEX_CONVERSION_MODEL
        self.llm = lms.llm(self.model_name)

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
        """Generate docinfo.tex with metadata."""
        
        docinfo_path = latex_dir / "docinfo.tex"
        
        # Convert comma-separated authors to LaTeX \and format
        # Accepts: "John Doe, Jane Smith" or "John Doe, Jane Smith, and Bob Johnson"
        # Converts to: "John Doe \\and Jane Smith" or "John Doe \\and Jane Smith \\and Bob Johnson"
        if "," in metadata.author:
            # Remove "and" before last author if present, then split by comma
            author_str = metadata.author.replace(", and ", ", ").replace(" and ", ", ")
            authors_list = [a.strip() for a in author_str.split(",")]
            authors_tex = " \\and ".join(authors_list)
        elif "\\and" in metadata.author:
            # Already in LaTeX format, just normalize spacing
            authors_tex = metadata.author.replace("\\and", " \\and ")
        else:
            authors_tex = metadata.author
        
        # Convert comma-separated supervisors to LaTeX \\ format
        # Accepts: "Prof. John Doe, Dr. Jane Smith"
        # Converts to: "Prof. John Doe\\\\Dr. Jane Smith"
        if "," in metadata.supervisor:
            supervisors_list = [s.strip() for s in metadata.supervisor.split(",")]
            supervisors_tex = " \\\\ ".join(supervisors_list)
        elif "\\\\" in metadata.supervisor:
            # Already in LaTeX format, just normalize spacing
            supervisors_tex = metadata.supervisor.replace("\\\\", " \\\\ ")
        else:
            supervisors_tex = metadata.supervisor
        
        submission_type = "digital" if metadata.digital_submission else "paper"
        
        docinfo_content = textwrap.dedent(f"""\
            % -------------------------------------------------------
            % Document Information and Settings
            %

            % Language setting (English only)
            \\newcommand{{\\hsmasprache}}{{en}}

            % Submission format
            \\newcommand{{\\hsmaabgabe}}{{{submission_type}}}

            % Publication and restriction flags
            \\newcommand{{\\hsmapublizieren}}{{opensource}}

            % Feature flags
            \\newcommand{{\\hsmaquellcode}}{{nosourcecode}}
            \\newcommand{{\\hsmasymbole}}{{nosymbole}}
            \\newcommand{{\\hsmaglossar}}{{noglossar}}
            \\newcommand{{\\hsmatc}}{{notc}}

            % Title (English only)
            \\newcommand{{\\hsmatitelen}}{{{metadata.title}}}

            % Authors (supports multiple authors - use \\and to separate)
            \\newcommand{{\\hsmaauthors}}{{{authors_tex}}}

            % Authors in bibliography format (Lastname, Firstname)
            % Note: This is a simplified version - you may want to format this properly
            \\newcommand{{\\hsmaauthorsbib}}{{{metadata.author}}}

            % Location and date
            \\newcommand{{\\hsmaort}}{{Mannheim}}
            \\newcommand{{\\hsmaabgabedatum}}{{{metadata.submission_date}}}

            % Supervisors (supports multiple supervisors - use \\\\ to separate lines)
            \\newcommand{{\\hsmasupervisors}}{{{supervisors_tex}}}

            % Optional fields (leave empty if not needed)
            \\newcommand{{\\hsmafakultaet}}{{{metadata.faculty}}}
            \\newcommand{{\\hsmastudiengang}}{{{metadata.study_program}}}
            \\newcommand{{\\hsmafirma}}{{{metadata.company}}}
        """)
        
        docinfo_path.write_text(docinfo_content, encoding="utf-8")
        logger.info(f"[PaperConverter] Generated docinfo.tex")

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
            latex_content = MarkdownToLaTeX.convert_section_to_latex(section_content, section_type, self.llm)
            
            # Handle abstract specially - wrap in \newcommand
            if section_type == Section.ABSTRACT:
                # Wrap abstract content in \newcommand{\hsmaabstracten}{...}
                abstract_header = textwrap.dedent("""\
                    % -------------------------------------------------------
                    % Abstract
                    % This file defines the abstract content for the paper.
                    % The \\hsmaabstracten command is automatically populated with content
                    % from the paper generation pipeline.
                    %
                    % Note: If you want to use quotation marks in the abstract, do not use
                    %       "` and "', but rather \\enquote{}. "` and "' are not properly
                    %       recognized.
                    \\newcommand{\\hsmaabstracten}{""")
                abstract_footer = "}"
                latex_content = f"{abstract_header}{latex_content}{abstract_footer}"
            else:
                # Wrap in chapter if needed
                if "\\chapter{" not in latex_content:
                    section_title = section_type.value
                    latex_content = f"\\chapter{{{section_title}}}\n\\label{{{section_title.replace(' ', '')}}}\n\n{latex_content}"
            
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
            
            # Extract "Full Form (ABBR)" patterns and convert to \gls{key}
            # Pattern: text like "Artificial Intelligence (AI)" or "Machine Learning (ML)"
            pattern = r'([A-Z][^()]*(?:\s+[A-Z][^()]*)*)\s*\(([A-Z]+)\)'
            for match in re.finditer(pattern, content):
                full_form = match.group(1).strip()
                abbr = match.group(2).strip()
                key = abbr.lower()
                
                # Store definition
                if key not in definitions:
                    definitions[key] = (abbr, full_form)
                    keys.add(key)
                
                # Replace with \gls{key} (only first occurrence per file)
                # Note: This is a simple approach - might replace multiple times
                new_content = content.replace(match.group(0), f"\\gls{{{key}}}", 1)
                if new_content != content:
                    content = new_content
                    file_path.write_text(content, encoding="utf-8")
            
            # Also extract existing \gls{key} patterns
            keys.update(re.findall(r'\\gls(?:pl)?\{([^}]+)\}', content))
        
        # Scan abstract.tex
        scan_file(latex_dir / "abstract.tex")
        
        # Scan all .tex files in chapters directory
        chapters_dir = latex_dir / "chapters"
        if chapters_dir.exists():
            for file_path in chapters_dir.glob("*.tex"):
                scan_file(file_path)
        
        if not keys:
            return
        
        # Generate abbreviations.tex
        lines = ["% Abbreviations file", "% Automatically generated", ""]
        for key in sorted(keys):
            if key in definitions:
                abbr, full = definitions[key]
            else:
                # Fallback: infer from key
                abbr = key.upper()
                full = key.replace('_', ' ').title()
            full = full.replace("&", "\\&").replace("%", "\\%")
            lines.append(f"\\newacronym{{{key}}}{{{abbr}}}{{{full}}}")
        
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

