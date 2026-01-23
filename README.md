# Paper Generator

Automated academic paper generator using local language and embedding models via LM Studio.

Work in progress, everything subject to change...

## Generation Process

The entire process consists of 6 phases:
1. **[Context Analysis](phases/context_analysis/)** - Analyzes user code and requirements to generate a novel research concept
2. **[Paper Search](phases/paper_search/)** - Automated literature review: searches (via [Semantic Scholar](https://www.semanticscholar.org/), [arXiv](https://arxiv.org/), [Unpaywall](https://unpaywall.org/)), ranks, filters, and downloads relevant papers
3. **[Hypothesis Generation](phases/hypothesis_generation/)** - Generates valid, testable research hypotheses
4. **[Experimentation](phases/experimentation/)** - Automated experimentation: generates, executes, debugs, and validates scientific experiments
5. **[Paper Writing](phases/paper_writing/)** - Drafts, critiques, searches for evidence, and improves each section
6. **[LaTeX Generation](phases/latex_generation/)** - Converts the draft to a compiled LaTeX PDF

## Requirements

- **Python 3.10+**
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
./main.py
# or
python main.py
```
## Configuration

Simply use the in-app Settings screen to change:

- **LLM Models:** Select specific models to use for each phase (Analysis, Search, Writing, etc.).
  - *Note: MLX embedding models are currently NOT supported by LM Studio. Please use GGUF embedding models (e.g., `text-embedding-qwen3-embedding-4b`).*
- **API Keys:**
  - `SEMANTIC_SCHOLAR_API_KEY`: (Optional) for higher rate limits and faster paper search.
  - `UNPAYWALL_EMAIL`: (Optional) to identify open-access PDF versions of papers.
- **LaTeX:**
  - `LATEX_TEMPLATE`: Choose the template for the final PDF (e.g., `ieee_conference`, `jair`).
  - `LATEX_AUTHORS`: Configure the author details.



All generated files (PDFs, markdown drafts, experiment results) are saved to the `output/` folder.

## LM Studio Settings

**Recommended:**

- Developer → Server Settings → **Enable** "Only keep last JIT loaded models"
- App Settings → Developer → Local LLM Service → **Enable** "Enable Local LLM Service"