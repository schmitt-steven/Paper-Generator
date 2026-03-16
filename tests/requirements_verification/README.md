# Requirements Verification Test Suite

This directory contains the automated tests used to verify the Paper Generator's system requirements for the evaluation chapter.

All tests are described in more detail in thesis section 5.1.

## Overview

The evaluation uses three methods to verify the requirements:

1. **Static Analysis**  
   Tests that parse the project's source code and configuration files. These require no active LLM server and execute instantly.
   - `C1`: Technology Stack
   - `NFR1`: Privacy

2. **Output Inspection & Integration**  
   Tests that inspect the `output/` directory after a complete pipeline execution and run integration checks on the prompt and model loaded states. These verify that the system produced the artifacts required by the functional requirements (FR1–FR8).

3. **Cost Analysis (Documentary)**  
   `NFR2` (Free Execution) is verified via a documented Bill of Materials (BOM) rather than an automated script. See the NFR2 section below.

## Running the Tests

The automated tests are written using `pytest`.
To run all at once from the project root:

```bash
# Run all automated verification tests
python -m pytest tests/requirements_verification/ -v
```

---

## Verification Methods by Requirement

### FR1–FR6: Output Generation
**Tests:** `test_fr1_context.py` through `test_fr6_compilation.py`  
These tests inspect the `output/` directory after a complete pipeline execution to verify that the system successfully generated the expected structural files, code, and the final compiled PDF.

### FR7: Human-in-the-Loop
**Test:** `test_fr7_human_in_the_loop.py`  
Verifies that the system can load manually edited artifact files back into the prompt. It uses an input interception method that injects a UUID into a file, simulates a phase transition, and proves the UUID is present in the outgoing prompt payload sent to the LM Studio SDK.

### FR8: Model Selection
**Test:** `test_fr8_model_selection.py`  
Verifies dynamic model switching at runtime. It simulates phase transitions by sending load requests to the LM Studio API and polls the `/v1/models` endpoint to confirm the newly assigned model successfully loads into memory.

### C1: Technology Stack
**Test:** `test_c1_tech_stack.py`  
Scans the production codebase to confirm it contains `.py` files, imports the `tkinter` GUI library and imports the `lmstudio` Python SDK.

### NFR1: Privacy
**Test:** `test_nfr1_privacy.py`  
Verifies local-only inference by tracing all model loads. It proves that:
1. Every model load in the codebase uses `lms.llm()` or `lms.embedding_model()`.
2. The `lms` alias is always bound to the `lmstudio` SDK.
Because the LM Studio SDK connects exclusively to the local `localhost` server by design, this proves no inference data leaves the machine.

### NFR2: Free Execution
Pass condition: A cost analysis confirms that all runtime components carry open-source
licences, the inference engine is free for personal use, and all external API endpoints
operate without a billing agreement.

Verification method: Cost analysis (documented in thesis Section 5.1).
  NFR2 cannot be reduced to a binary code-level check, because proving a system
  costs nothing is inherently a qualitative claim. A blocklist of paid packages or
  API keys can never be exhaustive. The proof is therefore a Bill of Materials (BOM)
  that accounts for every runtime component.
