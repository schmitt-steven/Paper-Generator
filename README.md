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
- **Inkscape** (for SVG plot support in LaTeX)
- **LM Studio** running in background with at least 3 downloaded models:
  - One LLM capable of tool use
  - One VLM or multimodal model
  - One embedding model

## Installation

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install LaTeX (full distribution, ~4GB)
brew install --cask mactex

# Install Inkscape (for SVG support)
brew install --cask inkscape

# Install Python dependencies
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Install LaTeX
sudo apt-get install texlive-full

# Install Inkscape
sudo apt-get install inkscape

# Install Python dependencies
pip install -r requirements.txt
```

### Windows

1. Install [MikTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/)
2. Install [Inkscape](https://inkscape.org/release/)
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
./main.py
# or
python3 main.py
```

All generated files are saved to the `output/` folder.

## LM Studio Settings

**Recommended:**

- Developer → Server Settings → **Enable** "Only keep last JIT loaded models"
- App Settings → Developer → Local LLM Service → **Enable** "Enable Local LLM Service"

**Note:** MLX embedding models are NOT supported by LM Studio yet.
See https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/808
