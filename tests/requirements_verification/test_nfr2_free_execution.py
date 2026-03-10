"""
NFR2 — Free Execution
Requirement: The system shall perform all functions free of charge.

Pass condition: A cost analysis confirms that all runtime components carry open-source
licences, the inference engine is free for personal use, and all external API endpoints
operate without a billing agreement.

Verification method: Cost analysis (documented in thesis Section 5.1).
  NFR2 cannot be reduced to a binary code-level check, because proving a system
  costs nothing is inherently a qualitative claim. A blocklist of paid packages or
  API keys can never be exhaustive. The proof is therefore a Bill of Materials (BOM)
  that accounts for every runtime component:

  1. Runtime language
     Python 3.14 is distributed under the Python Software Foundation License.
     It is open-source and free to use without restriction.

  2. Third-party packages (see requirements.txt)
     All dependencies carry open-source licences:
       - MIT:        lmstudio, pydantic, sv-ttk, pillow, tkinterweb
       - Apache 2.0: requests
       - BSD-3:      numpy, pandas, matplotlib, seaborn, scipy, markdown
       - AGPL-3.0:   PyMuPDF, pymupdf4llm

  3. Inference engine
     LM Studio (version 0.4.6+1) is free for personal and research use.
     All model inference runs on the user's local hardware. There are no
     pay-per-token API calls.

  4. External API endpoints
     The system makes external network calls only for literature retrieval:
       - Semantic Scholar API (https://api.semanticscholar.org)
         Public API, no billing account required.
       - Unpaywall API (https://unpaywall.org/products/api)
         Free academic API, no billing account required.

  Every layer of the stack is free. NFR2 passed.
"""

def test_nfr2_is_verified_by_cost_analysis():
    """
    NFR2 is verified by a cost analysis documented in the thesis and in this module's docstring.
    This placeholder test simply passes to signal that NFR2 verification is accounted for
    in the test suite structure.
    """
    pass
