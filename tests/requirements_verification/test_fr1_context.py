"""
FR1 — Context Analysis
Requirement: The system shall process user data and produce a structured research topic definition.

Pass condition: The output directory contains a research_context.md file with the expected
sections: Taxonomic Classification, Problem Definition, and Open Questions.

Method: E2E Execution & Output artifact inspection.
"""

from pathlib import Path
import os
import re
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKSPACE = PROJECT_ROOT / "tests" / "requirements_verification" / "test_workspace"
OUTPUT_DIR = TEST_WORKSPACE / "output"
CONTEXT_FILE = OUTPUT_DIR / "research_context.md"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module", autouse=True)
def run_context_phase():
    """Run the Context Analysis phase inside the persistent test workspace."""
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))
    
    # Only run if not already generated, allowing sequential test runs to be fast
    if not CONTEXT_FILE.exists():
        from phases.context_analysis.research_context_generator import ResearchContextGenerator
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ResearchContextGenerator.generate_new_context()
        
    os.chdir(original_cwd)


def test_fr1_output_file_exists():
    """FR1: The output/research_context.md file exists and is not empty."""
    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert CONTEXT_FILE.exists(), f"{CONTEXT_FILE.name} not found in output directory."
    assert CONTEXT_FILE.stat().st_size > 0, f"{CONTEXT_FILE.name} is empty."


def test_fr1_contains_required_sections():
    """
    FR1: The context file contains the required structural sections.
    """
    assert CONTEXT_FILE.exists(), "File must exist to check sections"
    content = CONTEXT_FILE.read_text(encoding="utf-8").lower()
    
    required_patterns = [
        re.compile(r'^#+\s*.*taxonomic classification.*$', re.MULTILINE),
        re.compile(r'^#+\s*.*problem definition.*$', re.MULTILINE),
        re.compile(r'^#+\s*.*open questions.*$', re.MULTILINE)
    ]
    
    missing = []
    for pattern in required_patterns:
        if not pattern.search(content):
            missing.append(pattern.pattern)
            
    assert not missing, f"Missing expected sections in {CONTEXT_FILE.name}: {missing}"
