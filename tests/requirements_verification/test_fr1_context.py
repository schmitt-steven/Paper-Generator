"""
FR1 — Context Analysis
Requirement: The system shall process user data and produce a structured research topic definition.

Pass condition: The output directory contains a research_context.md file with the expected
top-level sections (Research Context, Open Questions for Literature Search, Dataset
Descriptions, Important Code Snippets) and each section has non-empty content.

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

    from phases.context_analysis.research_context_generator import ResearchContextGenerator
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ResearchContextGenerator.generate_new_context()

    os.chdir(original_cwd)


def test_fr1_output_file_exists(log_evidence):
    """FR1: The output/research_context.md file exists and is not empty."""
    exists = CONTEXT_FILE.exists()
    size = CONTEXT_FILE.stat().st_size if exists else 0
    log_evidence("file_path", str(CONTEXT_FILE.relative_to(PROJECT_ROOT)))
    log_evidence("file_exists", exists)
    log_evidence("file_size_bytes", size)

    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert exists, f"{CONTEXT_FILE.name} not found in output directory."
    assert size > 0, f"{CONTEXT_FILE.name} is empty."


def _parse_top_level_sections(content: str) -> dict[str, str]:
    """Split a markdown file by top-level (single #) headers, return {header: body}."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        if re.match(r'^# ', line):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line.lstrip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def test_fr1_contains_required_sections(log_evidence):
    """
    FR1: The context file contains all four required top-level sections, each with non-empty content.
    """
    assert CONTEXT_FILE.exists(), "File must exist to check sections"
    content = CONTEXT_FILE.read_text(encoding="utf-8")

    sections = _parse_top_level_sections(content)
    section_names = list(sections.keys())

    required_sections = [
        "Research Context",
        "Open Questions for Literature Search",
        "Dataset Descriptions",
        "Important Code Snippets",
    ]

    missing = []
    empty = []
    for name in required_sections:
        if name not in sections:
            missing.append(name)
        elif not sections[name]:
            empty.append(name)

    log_evidence("sections_found", section_names)
    log_evidence("sections_missing", missing)
    log_evidence("sections_empty", empty)
    log_evidence("content_preview", content[:500])

    assert not missing, f"Missing required sections in {CONTEXT_FILE.name}: {missing}"
    assert not empty, f"Sections present but empty in {CONTEXT_FILE.name}: {empty}"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
