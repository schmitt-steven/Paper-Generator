"""Info popup texts"""

START_PAGE_INFO = """\
## Welcome to the Paper Generator!

This app helps your write a scientific paper from start to finish.

### Getting Started

1.  **Configure the app** in the Settings.
2.  **Define your topic and requirements** in `paper_specification.md`.
3.  **Check the style guidelines** in `style_guidelines.md` and adjust them if needed.
4.  **Upload related code files** (optional, but recommended).

Click **Generate** to begin the process!

---

This project is open source and available on GitHub:
[https://github.com/schmitt-steven/Paper-Generator](https://github.com/schmitt-steven/Paper-Generator)
"""

STYLE_GUIDELINES_INFO = """\
Specify **how** each section of the paper should be written.

For example, you can specify:

- How long the section should be
- What writing style should be used
- How the section should be structured

# File Structure
  
The section style guidelines file **must** have the following structure:

```markdown
# Style Guidelines

## Abstract
...

## Introduction
...

## Related Work
...

## Methods
...

## Results
...

## Discussion
...

## Conclusion
...

## Acknowledgements
...
```
"""

SETTINGS_INFO = """\
## General

## LaTeX Template

## Authors

## LLM Models

## Appearance
"""


CODE_FILES_INFO = """\
Upload your code files relevant to the topic of the paper here.

It's not required to upload code files, but it can help the LLM to better understand your algorithm and generate more accurate results.

The code files will also be used as context for the experimentation phase. The LLM even has the option to import methods or classes from the provided code.
This can be helpful if you're trying to compare many different algorithms with each other, so the LLM doesn't have to "reinvent the wheel". 

# Code Structure Best Practices

To ensure the best results during the experimentation phase, follow these guidelines:

- **Avoid Global State**: Do not rely on global variables for critical logic (e.g., model instances, configuration parameters).
- **Use Classes/Functions**: Encapsulate logic in classes or functions that accept dependencies as arguments.
- **Importability**: Ensure your code can be imported without executing immediate side effects (use `if __name__ == "__main__":` for scripts).
"""

PAPER_SPECIFICATION_INFO = """\
Specify the topic of your paper and requirements for the sections of the paper here.

Some questions to ask yourself:

- What exactly do you want to write about?
- Why is it novel or relevant?
- What are the fundamentals the LLM must know about?
- What do you want to prove/falsify?
- What content should each section of the paper contain?
- ...

The content of this file is the basis of all following steps!<br>
Make sure it corresponds to the paper you have in your mind!

Note: If you want to specify the style/form the sections should be written in, adjust the `style_guidelines.md` file.

# File Structure
  
The requirements file **must** have the following structure:

```markdown
## General Information

### Topic
...

### Hypothesis
...

## Section Requirements

### Abstract
...

### Introduction
...

### Related Work
...

### Methods
...

### Results
...

### Discussion
...

### Conclusion
...

### Acknowledgements
...
```
"""

PAPER_CONCEPT_INFO = """paper concept"""

PAPER_SELECTION_INFO = """paper selection"""

HYPOTHESIS_INFO = """hypothesis"""

EXPERIMENT_PLAN_INFO = """experiment plan"""

EXPERIMENT_RESULTS_INFO = """experiment results"""

EVIDENCE_INFO = """evidence"""

WRITING_PROMPTS_INFO = """writing prompts"""

PAPER_DRAFT_INFO = """paper draft"""

RESULT_INFO = """result"""

STYLE_GUIDELINES_EDITOR_INFO = """style guidelines"""

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

Available fields:

- `{{name}}`
- `{{affiliation}}`
- `{{department}}`
- `{{city}}`
- `{{country}}`
- `{{address}}`
- `{{email}}`

`%%SHORTAUTHORS%%` - Auto-generated from author last names (for page headers)

Only ONE author placeholder is required.

The generator handles multiple authors automatically.

### Bibliography
No placeholder needed.

The generator creates a bibliography.bib file and saves it to the output directory.

Include the bibliography via \\addbibresource{bibliography.bib} or similar.

After adding a template folder, restart the app to select it on the settings page.
"""
