# Paper Generator

Automatic academic paper generator using local language and embedding models.

## Requirements

- #### Python 3.10+

- #### Python Packages
    - `lmstudio`
    - `feedparser`
    - `requests`
    - `pymupdf4llm`
    - Could be used by LLM for experiments:
        - `numpy`
        - `matplotlib`
        - `seaborn`
        - `pygame`


- #### LaTeX (MacTeX or BasicTeX, TeX Live, MikTeX...)

### LM Studio

The LM Studio App must be installed.

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
python3 -m pip install lmstudio numpy matplotlib seaborn pymupdf4llm feedparser requests

# Install LaTeX
# MacTeX (full distribution, ~4GB)
brew install --cask mactex
```

## Usage

First, put your files (notes, code, ...) related to the paper/topic into the `user_files/` folder.

Adjust `settings.py`, then run:\
`python3 paper_generator.py`

All generated files are saved to output/


## TODOs

- Switch from Arxiv API to SemanticScholar API (more papers, more metadata, embeddings of abstracts, ...)
- Add human-in-the-loop feature
- Add review/improvement loop to the paper writing process
- Try coding agent like SWE-agent or OpenHands
- Add logic if hypothesis was disproven
- Add logic to test multiple hypotheses
- Improve logic for executing steps (e.g. run_from(X), run_only(X), run_steps([X, Y]), run_until(X))
- Standardize saving/loading
- Finetune prompts; update formatting (XML tags)
- Use tokenizers instead of approximations
...

## Algorithm

The simplified algorithm as an UML-like diagram.
A more detailed explanation of the paper writing process can be found in `phases/paper_writing/`

![Paper Generator Algorithm](algorithm_diagram.png)
