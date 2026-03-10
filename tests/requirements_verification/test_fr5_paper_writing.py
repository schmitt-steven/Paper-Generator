"""
FR5 — Paper Writing
Requirement: The system shall generate text sections that include citations from the
retrieved literature.

Pass condition: The generated Markdown files contain citation keys.

Method: E2E Execution & Output artifact inspection.
"""

from pathlib import Path
import os
import re
import re
import sys
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKSPACE = PROJECT_ROOT / "tests" / "requirements_verification" / "test_workspace"
OUTPUT_DIR = TEST_WORKSPACE / "output"
PAPER_DRAFT_MD = OUTPUT_DIR / "paper_draft.md"
PAPERS_JSON = OUTPUT_DIR / "papers.json"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module", autouse=True)
def run_paper_writing_phase():
    """Run the Paper Writing phase inside the persistent test workspace."""
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))
    
    # FR5 is the deep end of the pipeline and requires all previous outputs.
    # If run as part of the full test suite, FR1-4 will have already generated these.
    required_files = [
        OUTPUT_DIR / "research_context.md",
        OUTPUT_DIR / "hypothesis.md",
        OUTPUT_DIR / "papers.json",
        OUTPUT_DIR / "experiments" / "experiment_plan.md"
    ]
    
    missing = [f.name for f in required_files if not f.exists()]
    if missing:
        os.chdir(original_cwd)
        pytest.skip(f"Cannot run standalone FR5. Missing prerequisites: {missing}. Please run the full test suite.")
        return
        
    if not PAPER_DRAFT_MD.exists():
        from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
        pipeline = PaperWritingPipeline()
        pipeline.run_pipeline()
        
    os.chdir(original_cwd)


def test_fr5_paper_draft_exists():
    """FR5: The output directory contains a generated paper draft Markdown file."""
    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert PAPER_DRAFT_MD.exists(), f"{PAPER_DRAFT_MD.name} not found in output directory."
    assert PAPER_DRAFT_MD.stat().st_size > 0, f"{PAPER_DRAFT_MD.name} is empty."


def test_fr5_markdown_files_contain_citations():
    """
    FR5: The generated Markdown files contain citation keys.
    Citations format depends on prompt, but standard involves brackets, e.g. [Smith2020...]
    or some form of LaTeX cite commands if generated directly.
    """
    assert PAPER_DRAFT_MD.exists(), "Draft file must exist to check for citations."
    content = PAPER_DRAFT_MD.read_text(encoding="utf-8")
    
    # Regular expression for finding [AuthorYear...] or \cite{AuthorYear...}
    citation_patterns = [
        re.compile(r'\[([A-Za-z]+[0-9]{4}.*?)\]'), # e.g. [Diekhoff2024RecursiveBQ]
        re.compile(r'\\cite\{([A-Za-z]+[0-9]{4}.*?)\}') # e.g. \cite{Diekhoff2024RecursiveBQ}
    ]
    
    found_keys = []
    for p in citation_patterns:
        for match in p.finditer(content):
            # The regex group 1 is the actual key without brackets
            found_keys.append(match.group(1).strip())
            
    assert len(found_keys) > 0, f"No citation keys found in {PAPER_DRAFT_MD.name}."


def test_fr5_at_least_one_real_citation():
    """
    FR5: Verifies that at least one of the generated citation keys actually exists 
    in the retrieved 'papers.json' database.
    
    The LLM may hallucinate some keys, but for the integration to be successful, 
    at least one valid entry from the backend must have made it into the document.
    """
    assert PAPER_DRAFT_MD.exists(), "Draft file missing."
    assert PAPERS_JSON.exists(), "papers.json missing."
    
    content = PAPER_DRAFT_MD.read_text(encoding="utf-8")
    
    # Same extraction logic as above
    citation_patterns = [
        re.compile(r'\[([A-Za-z]+[0-9]{4}.*?)\]'),
        re.compile(r'\\cite\{([A-Za-z]+[0-9]{4}.*?)\}')
    ]
    
    found_keys = []
    for p in citation_patterns:
        for match in p.finditer(content):
            found_keys.append(match.group(1).strip())
            
    # Load backend database keys
    try:
        papers_data = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
        real_keys = [p.get("citationTitle", "") for p in papers_data]
    except Exception as e:
        pytest.fail(f"Failed to parse papers.json: {e}")
        
    # Check if at least one generated key exists in real keys
    match_found = any(key in real_keys for key in found_keys)
    
    assert match_found, (
        f"None of the {len(found_keys)} generated citations ({found_keys[:3]}...) "
        f"match any keys in the database ({len(real_keys)} available).\n"
        f"All generated keys were hallucinated."
    )

