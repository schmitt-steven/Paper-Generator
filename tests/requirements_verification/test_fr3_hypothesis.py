"""
FR3 — Hypothesis Generation
Requirement: The system shall derive a formal hypothesis from the provided context.

Pass condition: The hypothesis output file contains all three fields: description,
rationale, and success criteria.

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
HYPOTHESIS_FILE = OUTPUT_DIR / "hypothesis.md"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module", autouse=True)
def run_hypothesis_phase():
    """Run the Hypothesis Generation phase inside the persistent test workspace."""
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))
    
    # 1. Ensure prerequisite context exists
    if not (OUTPUT_DIR / "research_context.md").exists():
        from phases.context_analysis.research_context_generator import ResearchContextGenerator
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ResearchContextGenerator.generate_new_context()
        
    # 2. Run hypothesis generation if not already generated
    if not HYPOTHESIS_FILE.exists():
        from phases.context_analysis.research_context_generator import ResearchContextGenerator
        from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
        from settings import Settings
        
        context = ResearchContextGenerator.load_research_context(str(OUTPUT_DIR / "research_context.md"))
        builder = HypothesisBuilder(
            model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
            research_context=context,
            top_limitations=[],
            num_papers_analyzed=0
        )
        builder.build_hypothesis()
        
    os.chdir(original_cwd)


def test_fr3_output_file_exists():
    """FR3: The output/hypothesis.md file exists and is not empty."""
    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert HYPOTHESIS_FILE.exists(), f"{HYPOTHESIS_FILE.name} not found in output directory."
    assert HYPOTHESIS_FILE.stat().st_size > 0, f"{HYPOTHESIS_FILE.name} is empty."


def test_fr3_contains_required_fields():
    """
    FR3: The Markdown structure contains the three required structural fields (headings).
    """
    assert HYPOTHESIS_FILE.exists(), "File must exist to check fields"
    
    content = HYPOTHESIS_FILE.read_text(encoding="utf-8").lower()
    
    # Check for Markdown headings matching the required sections
    required_patterns = [
        re.compile(r'^#+\s*.*description.*$', re.MULTILINE),
        re.compile(r'^#+\s*.*rationale.*$', re.MULTILINE),
        re.compile(r'^#+\s*.*success criteria.*$', re.MULTILINE)
    ]
    
    missing = []
    for pattern in required_patterns:
        if not pattern.search(content):
            missing.append(pattern.pattern)
            
    assert not missing, f"Missing expected sections in {HYPOTHESIS_FILE.name}: {missing}"
