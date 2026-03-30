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

    # 2. Run hypothesis generation
    from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
    HypothesisBuilder.generate_new_hypothesis()

    os.chdir(original_cwd)


def test_fr3_output_file_exists(log_evidence):
    """FR3: The output/hypothesis.md file exists and is not empty."""
    exists = HYPOTHESIS_FILE.exists()
    size = HYPOTHESIS_FILE.stat().st_size if exists else 0
    log_evidence("file_path", str(HYPOTHESIS_FILE.relative_to(PROJECT_ROOT)))
    log_evidence("file_exists", exists)
    log_evidence("file_size_bytes", size)

    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert exists, f"{HYPOTHESIS_FILE.name} not found in output directory."
    assert size > 0, f"{HYPOTHESIS_FILE.name} is empty."


def test_fr3_contains_required_fields(log_evidence):
    """
    FR3: The Markdown structure contains the three required structural fields (headings).
    """
    assert HYPOTHESIS_FILE.exists(), "File must exist to check fields"

    content = HYPOTHESIS_FILE.read_text(encoding="utf-8")
    content_lower = content.lower()

    required_patterns = [
        re.compile(r'^#+\s*.*description.*$', re.MULTILINE),
        re.compile(r'^#+\s*.*rationale.*$', re.MULTILINE),
        re.compile(r'^#+\s*.*success criteria.*$', re.MULTILINE)
    ]

    found = []
    missing = []
    for pattern in required_patterns:
        if pattern.search(content_lower):
            found.append(pattern.pattern)
        else:
            missing.append(pattern.pattern)

    log_evidence("sections_found", found)
    log_evidence("sections_missing", missing)
    log_evidence("full_content", content)

    assert not missing, f"Missing expected sections in {HYPOTHESIS_FILE.name}: {missing}"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
