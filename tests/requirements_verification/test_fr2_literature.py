"""
FR2 — Literature Search
Requirement: The system shall query external databases and store both metadata and full-text documents.

Pass condition: The output directory contains a non-empty papers.json and at least one
downloaded .pdf file.

Method: E2E Execution & Output artifact inspection.
"""

from pathlib import Path
import os
import json
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKSPACE = PROJECT_ROOT / "tests" / "requirements_verification" / "test_workspace"
OUTPUT_DIR = TEST_WORKSPACE / "output"
PAPERS_JSON_FILE = OUTPUT_DIR / "papers.json"
LITERATURE_DIR = OUTPUT_DIR / "literature"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module", autouse=True)
def run_literature_phase():
    """Run the Literature Search phase inside the persistent test workspace."""
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))

    # 1. Ensure prerequisite context exists
    if not (OUTPUT_DIR / "research_context.md").exists():
        from phases.context_analysis.research_context_generator import ResearchContextGenerator
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ResearchContextGenerator.generate_new_context()

    # 2. Run literature search
    from phases.context_analysis.research_context_generator import ResearchContextGenerator
    from phases.paper_search.literature_search import LiteratureSearch
    from settings import Settings

    context = ResearchContextGenerator.load_research_context(str(OUTPUT_DIR / "research_context.md"))
    searcher = LiteratureSearch(model_name=Settings.LITERATURE_SEARCH_MODEL)

    # Execute automated search
    ranked_papers = searcher.run_automated_search(
        research_context=context,
        user_papers=[],
        progress_callback=None
    )

    # Save results and download PDFs
    searcher.save_papers(ranked_papers, filename="papers.json", output_dir=str(OUTPUT_DIR))
    OUTPUT_DIR.joinpath("literature").mkdir(exist_ok=True)
    searcher.download_papers_as_pdfs(ranked_papers, base_folder=str(OUTPUT_DIR / "literature"))

    os.chdir(original_cwd)


def test_fr2_papers_json_exists_and_valid(log_evidence):
    """FR2: The output/papers.json file exists and contains a non-empty list of papers."""
    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert PAPERS_JSON_FILE.exists(), f"{PAPERS_JSON_FILE.name} not found in output directory."
    assert PAPERS_JSON_FILE.stat().st_size > 0, f"{PAPERS_JSON_FILE.name} is empty."

    try:
        data = json.loads(PAPERS_JSON_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        assert False, f"{PAPERS_JSON_FILE.name} contains invalid JSON: {e}"

    titles = [p.get("title", p.get("citationTitle", "")) for p in data[:5]]
    log_evidence("paper_count", len(data))
    log_evidence("first_5_titles", titles)

    assert isinstance(data, list), f"{PAPERS_JSON_FILE.name} should contain a list of papers."
    assert len(data) > 0, f"{PAPERS_JSON_FILE.name} contains an empty list of papers."


def test_fr2_at_least_one_pdf_downloaded(log_evidence):
    """
    FR2: The output directory (specifically output/literature) contains at least one downloaded .pdf file.
    """
    assert LITERATURE_DIR.exists(), f"Literature directory not found at {LITERATURE_DIR}."

    pdf_files = list(LITERATURE_DIR.rglob("*.pdf"))
    pdf_names = [str(p.relative_to(OUTPUT_DIR)) for p in pdf_files]
    log_evidence("pdf_count", len(pdf_files))
    log_evidence("pdf_files", pdf_names)

    assert len(pdf_files) > 0, f"No .pdf files found in {LITERATURE_DIR}."


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
