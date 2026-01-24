"""Info popup texts"""

START_PAGE_INFO = """Start Page"""

WRITING_GUIDELINES_INFO = """Writing Guidelines"""

SETTINGS_INFO = """Settings screen"""


CODE_FILES_INFO = """code files"""

USER_REQUIREMENTS_INFO = """user requirements"""

PAPER_CONCEPT_INFO = """paper concept"""

PAPER_SELECTION_INFO = """paper selection"""

HYPOTHESIS_INFO = """hypothesis"""

EXPERIMENT_PLAN_INFO = """experiment plan"""

EXPERIMENT_RESULTS_INFO = """experiment results"""

EVIDENCE_INFO = """evidence"""

WRITING_PROMPTS_INFO = """writing prompts"""

PAPER_DRAFT_INFO = """paper draft"""

RESULT_INFO = """result"""

SECTION_GUIDELINES_INFO = """section guidelines"""

LATEX_TEMPLATE_INFO = """\
## Adding Custom LaTeX Templates

All LaTeX templates are stored in the latex_templates/ directory.

### Required structure
  latex_templates/
  └── your_template_name/
  &nbsp;&nbsp;├── paper.tex (your main file, MUST be called "paper.tex")
  &nbsp;&nbsp;├── Makefile (for compilation)
  &nbsp;&nbsp;└── ... (any .cls, .sty or whatever your template uses)

### Available Placeholders

The template system uses placeholders that can be placed in the paper.tex file.

#### Metadata
  `%%TITLE%%`
  `%%ABSTRACT%%`
#### Sections
  `%%INTRODUCTION%%`
  `%%RELATED_WORK%%`
  `%%METHODS%%`
  `%%RESULTS%%`
  `%%DISCUSSION%%`
  `%%CONCLUSION%%`
  `%%ACKNOWLEDGEMENTS%%`
#### Authors
  `%%BEGIN_AUTHOR%% ... %%END_AUTHOR%%`
  Available fields: `{{name}}`, `{{affiliation}}`, `{{department}}`, `{{city}}`, `{{country}}`, `{{address}}`, `{{email}}`
  `%%SHORTAUTHORS%%` - Auto-generated from author last names (for page headers)

  Only ONE author placeholder is required.
  The generator handles multiple authors automatically.

### Bibliography
  No placeholder needed.
  The generator creates a bibliography.bib file and saves it to the output directory.
  Include the bibliography via \\addbibresource{bibliography.bib} or similar.

After adding a template folder, restart the app to select it on the settings page.
"""
