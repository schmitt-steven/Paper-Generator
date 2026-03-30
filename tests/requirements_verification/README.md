# Requirements Verification Test Suite

This directory contains the automated tests used to verify the Paper Generator's system requirements for the evaluation chapter.

All tests are described in more detail in section 5.1 of the thesis.

## Overview

The evaluation uses three methods to verify the requirements:

1. **Static Analysis**  
   Tests that parse the project's source code and configuration files. These require no active LLM server and execute instantly.
   - `C1`: Technology Stack
   - `NFR1`: Privacy

2. **Output Inspection & Integration**  
   Tests that inspect the `output/` directory after a complete pipeline execution and run integration checks on the prompt and model loaded states. These verify that the system produced the artifacts required by the functional requirements (FR1–FR8).

3. **Cost Analysis (Documentary)**  
   `NFR2` (Free Execution) is verified via a Bill of Materials (BOM). See the NFR2 section below.

## Running the Tests

The automated tests are written using `pytest`.

To run a single test, specify the file path:

```bash
# Run a specific verification test
python -m pytest tests/requirements_verification/test_fr8_model_selection.py -v
```

To run all at once from root:

```bash
# Run all automated verification tests
python -m pytest tests/requirements_verification/ -v
```
