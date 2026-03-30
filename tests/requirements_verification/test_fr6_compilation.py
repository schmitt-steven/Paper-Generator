"""
FR6 — Document Compilation
Requirement: The system shall compile the generated content into a PDF.

Pass condition: The LaTeX compiler returns Exit Code 0 and the output .pdf file has
a size > 0 KB.

Method: E2E Execution & Output artifact inspection.
"""

from pathlib import Path
import os
import subprocess
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKSPACE = PROJECT_ROOT / "tests" / "requirements_verification" / "test_workspace"
OUTPUT_DIR = TEST_WORKSPACE / "output"
PAPER_DRAFT_MD = OUTPUT_DIR / "paper_draft.md"
LATEX_DIR = OUTPUT_DIR / "latex"
PDF_FILE = LATEX_DIR / "result" / "paper.pdf"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module", autouse=True)
def run_compilation_phase():
    """Run the Compilation phase inside the persistent test workspace."""
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))

    # FR6 requires the generated paper draft to exist.
    if not PAPER_DRAFT_MD.exists():
        os.chdir(original_cwd)
        pytest.skip("Cannot run standalone FR6. Missing prerequisite: paper_draft.md. Please run the full test suite.")
        return

    from phases.latex_generation.paper_converter import PaperConverter
    PaperConverter.generate_new_pdf()

    os.chdir(original_cwd)


def test_fr6_compilation_succeeds(log_evidence):
    """
    FR6: The LaTeX compiler returns Exit Code 0 and the output .pdf file has a size > 0 KB.
    We verify this by running `make` in the output/latex directory.
    """
    assert LATEX_DIR.exists(), f"LaTeX directory not found at {LATEX_DIR}. Phase execution failed."

    result = subprocess.run(
        ["make"],
        cwd=str(LATEX_DIR),
        capture_output=True,
        text=True
    )

    log_evidence("compiler_exit_code", result.returncode)
    log_evidence("compiler_stdout_tail", result.stdout[-1000:] if result.stdout else "")
    log_evidence("compiler_stderr_tail", result.stderr[-500:] if result.stderr else "")

    assert result.returncode == 0, (
        f"LaTeX compiler (make) failed with exit code {result.returncode}.\n"
        f"Stderr: {result.stderr}\nStdout: {result.stdout}"
    )


def test_fr6_pdf_file_exists_and_valid(log_evidence):
    """
    FR6: The output .pdf file has a size > 0 KB.
    """
    exists = PDF_FILE.exists()
    size_kb = round(PDF_FILE.stat().st_size / 1024, 1) if exists else 0
    log_evidence("pdf_path", str(PDF_FILE.relative_to(PROJECT_ROOT)))
    log_evidence("pdf_exists", exists)
    log_evidence("pdf_size_kb", size_kb)

    assert exists, f"Output PDF file not found at {PDF_FILE}."
    assert PDF_FILE.stat().st_size > 0, f"Output PDF file {PDF_FILE.name} is empty (0 KB)."


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
