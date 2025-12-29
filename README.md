# Paper Generator

Automatic academic paper generator using local language and embedding models via LM Studio.

Work in progress, everything subject to change...

## Generation Process

1. **[Context Analysis](phases/context_analysis/)** — Analyze user code and requirements to generate a paper concept
2. **[Paper Search](phases/paper_search/)** — Search, rank, filter, and download relevant academic papers
3. **[Hypothesis Generation](phases/hypothesis_generation/)** — Create a structured research hypothesis
4. **[Experimentation](phases/experimentation/)** — Run experiments to test the hypothesis
5. **[Paper Writing](phases/paper_writing/)** — Gather evidence and write each section using RAG
6. **[LaTeX Generation](phases/latex_generation/)** — Convert to LaTeX and compile to PDF

## Requirements

- **Python 3.10+**
- **LaTeX** (MacTeX, TeX Live, MikTeX, etc.)
- **LM Studio** running in background with:
  - One LLM capable of tool use
  - One VLM or multimodal model
  - One embedding model

### Python Packages

- `lmstudio` (LM Studio SDK)
- `pydantic` (data validation)
- `pymupdf4llm` (PDF parsing)
- `pymupdf` (PDF manipulation)
- `requests` (Semantic Scholar API)
- `sv_ttk` (Tkinter theme)
- `Pillow` (image processing)
- Could be used by LLM for experiments:
  - `numpy`, `matplotlib`, `seaborn`, `pygame`, `scipy`

## Installation (macOS)

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Python packages
python3 -m pip install lmstudio pydantic pymupdf4llm pymupdf requests sv_ttk Pillow scipy numpy matplotlib seaborn pygame

# Install LaTeX (full distribution, ~4GB)
brew install --cask mactex
```

## Usage

```bash
python3 main.py
```

All generated files are saved to the `output/` folder.

## LM Studio Settings

**Recommended:**

- Developer → Server Settings → **Enable** "Only keep last JIT loaded models"
- App Settings → Developer → Local LLM Service → **Enable** "Enable Local LLM Service"

**Note:** MLX embedding models are NOT supported by LM Studio yet.
See https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/808
