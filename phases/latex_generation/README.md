# LaTeX Generation Phase

Converts a `PaperDraft` object to a compilable LaTeX project and generates a PDF.

## Components

### `LaTeXMetadata` ([paper_converter.py](paper_converter.py))

Document metadata (title, authors) loaded from settings.

### `MarkdownToLaTeX` ([markdown_to_latex.py](markdown_to_latex.py))

Uses LLM to convert markdown sections to LaTeX format.

### Bibliography Functions ([bibliography.py](bibliography.py))

- Extracts citation keys from markdown text
- Creates paper mapping from citation keys to Paper objects
- Generates `bibliography.bib` from cited papers

### `PaperConverter` ([paper_converter.py](paper_converter.py))

Handles the complete LaTeX conversion workflow:
- Sets up LaTeX directory from template
- Converts each section to LaTeX using LLM
- Populates metadata (title, authors)
- Copies plot images to LaTeX images directory
- Generates bibliography and abbreviations
- Compiles to PDF using Makefile

## Output

`output/latex/` containing:
- `paper.tex` - Main LaTeX document
- `chapters/` - LaTeX files for each section
- `bibliography.bib` - Generated bibliography
- `abbreviations.tex` - Extracted abbreviations
- `images/` - Copied plot images
- `result/paper.pdf` - Final compiled PDF
