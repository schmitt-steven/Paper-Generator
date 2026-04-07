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
- `paper.tex` - Main LaTeX document (with injected content)
- `bibliography.bib` - Generated bibliography
- `images/` - Copied plot images
- `result/paper.pdf` - Final compiled PDF


## Adding Custom LaTeX Templates

Templates are stored in the `latex_templates/` directory.

### Required Structure

```text
latex_templates/
└── your_template_name/
    ├── paper.tex (main file, MUST be named "paper.tex")
    ├── Makefile (for compilation)
    └── ... (any .cls, .sty or other template dependencies)
```

### Placeholders

The template system uses placeholders in the `paper.tex` file:

- **Metadata**: `%%TITLE%%`, `%%ABSTRACT%%`
- **Sections**: `%%INTRODUCTION%%`, `%%RELATED_WORK%%`, `%%METHODS%%`, `%%RESULTS%%`, `%%DISCUSSION%%`, `%%CONCLUSION%%`, `%%ACKNOWLEDGEMENTS%%`
- **Authors**:
  - `%%BEGIN_AUTHOR%% ... %%END_AUTHOR%%`
  - Available fields: `{{name}}`, `{{affiliation}}`, `{{department}}`, `{{city}}`, `{{country}}`, `{{address}}`, `{{email}}`
  - `%%SHORTAUTHORS%%`: Auto-generated from author last names.
  - *Note: Only one author placeholder block is required; the generator handles multiple authors automatically.*

### Bibliography

No placeholder is needed. The generator creates a `bibliography.bib` file in the output directory. Include it in your template via `\addbibresource{bibliography.bib}` or similar.

After adding a template folder, restart the app to select it on the settings page.
