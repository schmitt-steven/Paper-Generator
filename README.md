# Paper Generator

Automated research paper generator using local language and embedding models via LM Studio.

Work in progress, everything subject to change...

## Generation Process

The entire process consists of 6 phases:
1. **[Context Analysis](phases/context_analysis/)** - Analyzes user's specification and code to define the context of the research
2. **[Literature Search](phases/paper_search/)** - Automated literature search: searches (via [Semantic Scholar](https://www.semanticscholar.org/), [arXiv](https://arxiv.org/), [Unpaywall](https://unpaywall.org/)), ranks, filters, and downloads relevant papers
3. **[Hypothesis Generation](phases/hypothesis_generation/)** - Generates testable research hypotheses
4. **[Experimentation](phases/experimentation/)** - Automated experimentation: generates, executes, debugs, and validates experiments
5. **[Section Writing](phases/paper_writing/)** - Drafts, critiques, searches for evidence, and improves each section
6. **[Document Compilation](phases/latex_generation/)** - Converts the draft to LaTeX and compiles it to a PDF

## Requirements

- **Python 3.11+**
- **LaTeX** (MacTeX or TeX Live, MikTeX, etc.)
- **LM Studio** running in background with at least 3 downloaded models:
  - One LLM capable of tool use
  - One VLM or multimodal model
  - One embedding model

## Installation

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

### Linux (Debian-based)

```bash
# Install LaTeX
sudo apt install texlive-full

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