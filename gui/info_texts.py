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

GENERAL_SETTINGS_INFO = """\
**Paper Title** — Title printed on the paper. If left empty, the LLM generates one automatically.

**Include Acknowledgements** — Toggle whether an acknowledgements section is generated.

**Semantic Scholar API Key** — Optional API key for higher rate limits when searching for literature.

**Unpaywall Email** — Optional email address used to query the Unpaywall API to find free PDFs of papers.

**Evidence Search Queries** — This setting influences the "Critique" step of the Draft-Critique-Retrive-Improve" writing pipeline. \
It sets the maximum number of search queries the critic can suggest per section. \
These search queries are then used in the "Retrieve" step to find relevant passages in the indexed papers. \
Each query retrieves chunks from the indexed papers, which are then scored by an LLM and filtered down to the most relevant passages. \
This is by far the most time-consuming step of the entire writing process. \
Five queries per section can easily take anything upwards of 30 minutes depending on \
your hardware. Lowering this value significantly speeds up the paper generation. Set to 0 to skip \
evidence retrieval entirely (the critic still suggests improvements, but there won't be a search for supporting evidence).
"""


CODE_FILES_INFO = """\
Upload your code files relevant to the topic of the paper here.

It's not required to upload code files, but it can help the LLM to better understand your algorithm and generate more accurate results.

The code files will also be used as context for the experimentation phase. The LLM even has the option to import methods or classes from the provided code.
This can be helpful if you're trying to compare many different algorithms with each other, so the LLM doesn't have to "reinvent the wheel". 

# Code Structure Best Practices

For the best results during the experimentation phase, follow these guidelines:

- **Avoid Global State**: Do not rely on global variables for critical logic (e.g., model instances, configuration parameters).
- **Use Classes/Functions**: Encapsulate logic in classes or functions that accept dependencies as arguments.
- **Importability**: Ensure your code can be imported without executing immediate side effects (use `if __name__ == "__main__":` for scripts).
"""

USER_EXPERIMENT_INFO = """\
## Bring Your Own Experiment

You can use one of your uploaded Python files as the experiment, \
skipping both experiment plan generation and experiment code generation.

Your script will be executed directly via Python subprocess.

### Requirements

- **Python only**: The file must be a `.py` file.
- **Headless execution**: Avoid GUI windows, `plt.show()` or similar. Use `matplotlib.use('Agg')` before importing pyplot.
- **Save plots** to a `plots/` subdirectory (relative path). Use **PDF** as the file format.
- **Save results** to `results.json` in the working directory.
- **Use relative paths**: The working directory is `output/experiments/`. All file I/O should use relative paths.
- **No interactive input**: No `input()`, `sys.stdin`, or similar.
- **Finish within 10 minutes**: The execution has a default 600-second timeout. You can change this in the code.
- **Importable files**: Your other uploaded code files will be copied to the working directory, so the script can import from them.

### What Gets Skipped

When a user experiment is active:
1. Experiment plan generation is skipped.
2. Experiment code generation is skipped.
3. Your code files are copied to `output/experiments/` and the experiment script executed directly.

The validation, plot captioning and verdict generation steps still run normally.
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

RESEARCH_CONTEXT_INFO = """\
The system extracts the research topic from your paper specification and provided code files.
It compiles a paper description, isolates code snippets and lists open questions.
The literature search phase uses these parts to find related papers.

Review the generated context to catch misunderstandings early.
You can open the file in your preferred editor to make manual changes.

Use the regenerate button if you changed the paper specification or code files to reflect the changes.
"""

LITERATURE_SEARCH_INFO = """\
Upload any PDFs of relevant papers you already have here.
If the file is named after arXiv or Semantic Scholar ID's, the system can extract their real metadata for properly generating their citations.
If not, the system will try to extract the paper's title and search for it on Semantic Scholar.

The system can also query the Semantic Scholar API to find literature matching your research context.
It ranks papers by relevance (semantic similarity to research context description), citation count, and recency.
The system checks if a free PDF is available for each paper or if its closed access.
If its closed access, the respective entry in the list will provide a upload button to add the PDF manually.

Delete any papers that you dislike or do not fit your topic suing the "X"-button.

Once you continue to the next screen, the system will automatically detect any non-processed papers to download and convert them to Markdown."""

HYPOTHESIS_INFO = """\
The system translates your research context into a testable statement.
It provides a logical justification and defines testable success criteria.

Review the hypothesis to confirm it matches your research idea.
As always, you can edit the text directly or regenerate it.
"""

EXPERIMENT_PLAN_INFO = """\
The system outlines the objective, setup, metrics, and expected outputs for the following experiment.
Review the plan to catch any design errors before the experiment code is generated.
"""

EXPERIMENT_RESULTS_INFO = """\
The system writes the Python code, executes it and validates the results.
It compares the output against the success criteria of the hypothesis to determine a final verdict.

A vision-language model writes captions for any generated plots.

If the execution fails or disproved your hypothesis, you can edit the code and re-run it manually.
Alternatively, you can go back and tweak your hypothesis and/or experiment plan, and then come back to either re-run the code or entirely regnerate the experiment.
"""

WRITING_PROMPTS_INFO = """\
The system writes the paper section by section.
This screen shows exactly what data the system sends to the language model.

You can use this screen to trace the origin of every generated section.
"""

PAPER_DRAFT_INFO = """\
The system takes all generated sections and assembles them into a single Markdown document.
The draft includes the title, abstract, all main sections, and optionally acknowledgements.
"""

RESULT_INFO = """\
The system converts the Markdown draft into LaTeX code.
It inserts the text into the selected LaTeX template and generates the bibliography entries.
The system then runs a compiler to build the final PDF.

Which LaTeX template is used, depends on what template you set on the Settings screen.
Check the LaTeX templates section in the settings for instructions on how to add custom LaTeX templates.
"""

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
