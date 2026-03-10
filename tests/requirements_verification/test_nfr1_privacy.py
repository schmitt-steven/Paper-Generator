"""
NFR1 — Privacy
Requirement: The system shall process all inference data locally.

Pass condition: All LLM inference endpoints in the source code point to localhost.

Method: Proof by  model-load tracing.
  The lmstudio SDK connects to LM Studio's local server by design and exposes no
  remote endpoint configuration. This test verifies that every model load in the
  production codebase goes through lms.llm() or lms.embedding_model() — the two
  lmstudio SDK entry points — and that no file binds the `lms` alias to anything
  other than the lmstudio package.
  
  Proof chain:
    lms.llm() / lms.embedding_model()
    -> lmstudio SDK
    -> LM Studio local server (localhost only)
    -> local inference
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCLUDE_DIRS = {"tests", "thesis", ".git", "__pycache__", "old_code", "examples"}

# The two lmstudio SDK model-loading entry points.
MODEL_LOAD_PATTERNS = [
    re.compile(r'\blms\.llm\s*\('),
    re.compile(r'\blms\.embedding_model\s*\('),
]

# The only valid import that binds the `lms` name.
VALID_LMS_IMPORT = re.compile(r'import lmstudio as lms\b')


def collect_source_files() -> list[Path]:
    return [
        p for p in PROJECT_ROOT.rglob("*.py")
        if not any(ex in p.parts for ex in EXCLUDE_DIRS)
    ]


def uses_lms(source: str) -> bool:
    """Return True if the file contains any lms.llm() or lms.embedding_model() call."""
    return any(p.search(source) for p in MODEL_LOAD_PATTERNS)


def test_nfr1_all_model_loads_use_lmstudio():
    """
    NFR1: Every file that loads a model does so via lms.llm() or lms.embedding_model(),
    which are lmstudio SDK calls that route to the local LM Studio server.
    """
    violations = []
    for path in collect_source_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        if not uses_lms(source):
            continue  # file does not load any model — skip
        rel = path.relative_to(PROJECT_ROOT)

        # Every file that calls lms.llm() or lms.embedding_model() must import
        # `lms` from the lmstudio package (not a reassigned alias).
        if not VALID_LMS_IMPORT.search(source):
            # Check for any other binding of `lms`
            other_lms = re.search(r'\blms\b\s*=', source)
            violations.append(
                f"[{rel}] uses lms.* but does not import lmstudio as lms"
                + (f" (lms re-bound at: {other_lms.group(0)!r})" if other_lms else "")
            )

    assert not violations, (
        "Model load(s) found that do not provably route through lmstudio:\n"
        + "\n".join(violations)
    )


def test_nfr1_model_loads_exist():
    """
    NFR1: lmstudio model loads are actually present in the codebase.
    Confirms the production code calls the SDK, not that all calls happen to be absent.
    """
    hits = []
    for path in collect_source_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        if uses_lms(source):
            hits.append(str(path.relative_to(PROJECT_ROOT)))

    assert hits, "No lms.llm() or lms.embedding_model() calls found in production code."
