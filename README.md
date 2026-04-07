# Local Research Paper Generator

This app builds complete research papers locally and privately. Zero API costs. A GUI app uses local language and embedding models via LM Studio to search literature, formulate hypotheses, execute code for live experiments, draft text sections and compile the final LaTeX document. It uses a Human-in-the-Loop apprach to always keep you in control of the research process.

## Generation Process

The entire process consists of 6 phases:
1. **[Context Analysis](phases/context_analysis/)** - Analyzes user's specification and code to define the context of the research
2. **[Literature Search](phases/literature_search/)** - Automated literature search: searches (via [Semantic Scholar](https://www.semanticscholar.org/), [arXiv](https://arxiv.org/), [Unpaywall](https://unpaywall.org/)), ranks, filters, and downloads relevant papers
3. **[Hypothesis Generation](phases/hypothesis_generation/)** - Generates testable research hypotheses
4. **[Experimentation](phases/experimentation/)** - Automated experimentation: generates, executes, debugs, and validates experiments
5. **[Section Writing](phases/paper_writing/)** - Drafts, critiques, searches for evidence, and improves each section
6. **[Document Compilation](phases/document_generation/)** - Converts the draft to LaTeX and compiles it to a PDF

## Requirements

- **Python 3.11+**
- **LaTeX** (MacTeX or TeX Live, MikTeX, etc.)
- **LM Studio** running in background with at least 3 downloaded models:
  - One LLM capable of tool use
  - One VLM or multimodal model
  - One embedding model

## Installation

Install [LM Studio](https://lmstudio.ai/) first.

### macOS

```bash
# Xcode tools
xcode-select --install

# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# LaTeX
brew install --cask mactex

# Python dependencies
pip install -r requirements.txt
```

### Windows

1. Install [MikTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/)
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After the dependencies are installed, run:

```bash
python main.py  # or ./main.py
```

## LM Studio Settings

**Recommended:**

- Developer → Server Settings → **Enable** "Only keep last JIT loaded models"
- App Settings → Developer → Local LLM Service → **Enable** "Enable Local LLM Service"