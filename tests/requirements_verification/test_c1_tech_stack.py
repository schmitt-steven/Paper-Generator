"""
C1 — Technology Stack
Requirement: The system shall be implemented using Python (language),
             Tkinter (GUI), and the LM Studio Python SDK (inference engine).

Pass condition: The codebase contains .py source files, imports tkinter,
                imports from the lmstudio SDK package, and the requirements.txt
                file explicitly lists the LM Studio SDK dependency.

Method: Static source code analysis.
  - Verifies .py files exist in the production source directories.
  - Scans source files for `import tkinter` or `from tkinter`.
  - Scans source files for `import lmstudio` or `import lmstudio as lms`.
  - Verifies `lmstudio` is listed as a dependency in requirements.txt.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCLUDE_DIRS = {".git", "__pycache__", "old_code", "examples", "tests", "thesis"}


def collect_source_files() -> list[Path]:
    return [
        p for p in PROJECT_ROOT.rglob("*.py")
        if not any(ex in p.parts for ex in EXCLUDE_DIRS)
    ]


SOURCE_FILES = collect_source_files()


def test_c1_python_files_exist():
    """C1 (Python): The project contains .py source files."""
    assert SOURCE_FILES, "No .py source files found in the project."


def test_c1_tkinter_imported():
    """C1 (Tkinter): At least one source file imports tkinter."""
    hits = []
    for path in SOURCE_FILES:
        source = path.read_text(encoding="utf-8", errors="ignore")
        for line in source.splitlines():
            s = line.strip()
            if s.startswith("import tkinter") or s.startswith("from tkinter"):
                hits.append(str(path.relative_to(PROJECT_ROOT)))
                break
    assert hits, "No `import tkinter` found in any source file."


def test_c1_lmstudio_sdk_imported():
    """C1 (LM Studio SDK): At least one source file imports the lmstudio package."""
    hits = []
    for path in SOURCE_FILES:
        source = path.read_text(encoding="utf-8", errors="ignore")
        for line in source.splitlines():
            s = line.strip()
            if s.startswith("import lmstudio") or s.startswith("from lmstudio") or "import lmstudio" in s:
                hits.append(str(path.relative_to(PROJECT_ROOT)))
                break
    assert hits, "No `import lmstudio` found in any source file."


def test_c1_requirements_file_lists_lmstudio():
    """C1 (LM Studio SDK): The requirements.txt file lists lmstudio as a dependency."""
    req_file = PROJECT_ROOT / "requirements.txt"
    assert req_file.exists(), "requirements.txt not found."
    
    found = False
    for line in req_file.read_text().splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        # Split off version specifiers like "lmstudio>=1.5.0"
        pkg_name = line.split(">=")[0].split("<=")[0].split("==")[0].split("!=")[0].strip()
        if pkg_name == "lmstudio":
            found = True
            break
            
    assert found, "lmstudio is not listed as a dependency in requirements.txt"
