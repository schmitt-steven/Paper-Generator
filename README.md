# Paper Generator

Automatic academic paper generator using local language and embedding models via LM Studio.

The first generated papers can be found in the `examples` folder.

Work in progresss, everything subject to change...

## Requirements

- #### Python 3.10+

- #### Python Packages
    - `lmstudio`
    - `pydantic`
    - `pymupdf4llm` (for PDF parsing)
    - `requests` (for Semantic Scholar API calls)
    - `sv_ttk` (Tkinter theme)
    - Could be used by LLM for experiments:
        - `numpy`
        - `matplotlib`
        - `seaborn`
        - `pygame`

- #### LaTeX (MacTeX or BasicTeX, TeX Live, MikTeX...)

### LM Studio

The LM Studio App must be installed and running in the background.

You need at least:
- one LLM capable of tool use
- one VLM or multimodal model
- one embedding model

You could also run the code with just two local models:
- one multimodal language model that supports tool use
- one embedding model

Note: MLX embedding models are NOT supported by LM Studio yet :/\
See https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/808

**Recommendations:**

Under Developer → Server Settings:\
**Enable** "Only keep last JIT loaded models"

Under App Settings → Developer → Local LLM Service (headless):\
**Enable** "Enable Local LLM Service"

## Installation Commands for macOS

```bash
# Install Xcode Command Line Tools (for tools like make)
xcode-select --install

# Install Python packages
python3 -m pip install lmstudio numpy matplotlib seaborn pymupdf4llm sv_ttk pydantic

# Install LaTeX
# MacTeX (full distribution, ~4GB)
brew install --cask mactex
```

## Usage

Run `python3 -m gui.app` from the root directory

All generated files are saved to the `output` folder

## Algorithm

The simplified algorithm as an UML-like diagram.
A more detailed explanation of the paper writing process can be found in `phases/paper_writing/`

![Paper Generator Algorithm](algorithm_diagram.png)
