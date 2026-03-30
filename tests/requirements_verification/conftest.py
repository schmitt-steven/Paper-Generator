import json
import platform
import subprocess
import pytest
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from settings import Settings

RUNS_DIR = Path(__file__).resolve().parent / "test_runs"


# ---------------------------------------------------------------------------
# Evidence Plugin — records every test run to a timestamped folder
# ---------------------------------------------------------------------------

class _EvidencePlugin:
    """Pytest plugin that saves stdout, outcomes, and structured proof data
    for every run into test_runs/<timestamp>_<label>/."""

    def __init__(self):
        self._ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir: Path | None = None  # set in pytest_sessionstart
        self._test_results: list[dict] = []
        self.evidence: dict[str, dict] = {}

    def pytest_sessionstart(self, session):
        # Derive a short label from the collected test node IDs.
        # At sessionstart the items aren't collected yet, so we read
        # the raw CLI args instead.
        args = session.config.args  # list of paths/nodeids passed on the CLI
        # Single file → use its stem. Anything else (directory, no args) → "all".
        if len(args) == 1:
            stem = Path(args[0].split("::")[0]).stem
            label = stem if stem else "all"
        else:
            label = "all"

        folder_name = f"{label}_{self._ts}"
        self.run_dir = RUNS_DIR / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "evidence").mkdir(exist_ok=True)

    # -- hooks ---------------------------------------------------------------

    def pytest_runtest_logreport(self, report):
        # Capture setup failures and all call outcomes
        if report.when == "call" or (report.when == "setup" and report.failed):
            self._test_results.append({
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "when": report.when,
                "duration_s": round(report.duration, 3),
                "stdout": getattr(report, "capstdout", "") or "",
                "stderr": getattr(report, "capstderr", "") or "",
                "failure": str(report.longrepr) if report.longrepr else None,
            })

    def pytest_sessionfinish(self, session, exitstatus):
        if self.run_dir is None:
            return  # sessionstart never fired, nothing to write
        passed  = sum(1 for r in self._test_results if r["outcome"] == "passed")
        failed  = sum(1 for r in self._test_results if r["outcome"] == "failed")
        errors  = sum(1 for r in self._test_results if r["outcome"] == "error")
        skipped = sum(1 for r in self._test_results if r["outcome"] == "skipped")

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(PROJECT_ROOT), text=True
            ).strip()
        except Exception:
            git_commit = "unknown"

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "exit_status": int(exitstatus),
            "summary": {
                "total": len(self._test_results),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
            },
            "tests": [
                {
                    "nodeid": r["nodeid"],
                    "outcome": r["outcome"],
                    "duration_s": r["duration_s"],
                    "failure_summary": r["failure"].splitlines()[0] if r["failure"] else None,
                }
                for r in self._test_results
            ],
        }

        (self.run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Human-readable results.txt
        lines = [
            f"Test Run : {manifest['timestamp']}",
            f"Commit   : {git_commit}",
            f"Platform : {manifest['platform']}",
            f"Python   : {manifest['python']}",
            "",
            f"Results  : {passed} passed, {failed} failed, {errors} errors, {skipped} skipped",
            "=" * 70,
            "",
        ]
        for r in self._test_results:
            status = r["outcome"].upper().ljust(6)
            lines.append(f"[{status}] {r['nodeid']}  ({r['duration_s']}s)")
            if r["stdout"].strip():
                lines.append("  --- stdout ---")
                for ln in r["stdout"].rstrip().splitlines():
                    lines.append(f"  {ln}")
            if r["failure"]:
                lines.append("  --- FAILURE ---")
                for ln in r["failure"].splitlines()[:30]:
                    lines.append(f"  {ln}")
            lines.append("")

        (self.run_dir / "results.txt").write_text("\n".join(lines), encoding="utf-8")

        # Per-test evidence files
        evidence_dir = self.run_dir / "evidence"
        for nodeid, data in self.evidence.items():
            safe = nodeid.replace("/", "_").replace("::", "__").replace(".", "_")
            (evidence_dir / f"{safe}.json").write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )

        print(f"\n[Evidence] Run saved → {self.run_dir}")


def pytest_configure(config):
    plugin = _EvidencePlugin()
    config._evidence_plugin = plugin
    config.pluginmanager.register(plugin, "_evidence_plugin")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def run_dir(pytestconfig):
    """The timestamped directory for this test run."""
    return pytestconfig._evidence_plugin.run_dir


@pytest.fixture
def log_evidence(pytestconfig, request):
    """
    Callable fixture: log_evidence(key, value) saves structured proof data
    for the current test into the run's evidence folder.
    """
    store = pytestconfig._evidence_plugin.evidence
    nodeid = request.node.nodeid

    def _log(key, value):
        if nodeid not in store:
            store[nodeid] = {}
        store[nodeid][key] = value

    return _log


@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """
    Globally isolate and mock the Settings class for all verification tests.

    This ensures that tests:
    1. NEVER overwrite the user's real settings.py via save_to_file().
    2. Always use a consistent set of models for E2E integration, rather
       than whatever the user happened to select in the UI last.
    """

    # 1. Override the save function to do absolutely nothing during tests
    def _mock_save_to_file():
        pass

    Settings.save_to_file = classmethod(lambda cls: _mock_save_to_file())

    # 2. Hardcode the baseline models used for E2E verification
    # Using the standard 80B model to ensure consistent, testable capability
    test_llm = "qwen/qwen3.5-35b-a3b"
    test_embedding = "text-embedding-qwen3-embedding-4b@q5_0"

    # Apply to all phases
    Settings.CODE_ANALYSIS_MODEL = test_llm
    Settings.CONTEXT_GENERATOR_MODEL = test_llm
    Settings.LITERATURE_SEARCH_MODEL = test_llm
    Settings.HYPOTHESIS_BUILDER_MODEL = test_llm
    Settings.EXPERIMENT_PLAN_MODEL = test_llm
    Settings.EXPERIMENT_CODE_WRITE_MODEL = test_llm
    Settings.EXPERIMENT_VALIDATION_MODEL = test_llm
    Settings.EXPERIMENT_VERDICT_MODEL = test_llm
    Settings.PAPER_WRITING_MODEL = test_llm
    Settings.LATEX_GENERATION_MODEL = test_llm

    # Must explicitly be embedding models
    Settings.PAPER_RANKING_EMBEDDING_MODEL = test_embedding
    Settings.PAPER_INDEXING_EMBEDDING_MODEL = test_embedding

    # Vision models
    Settings.EXPERIMENT_PLOT_CAPTION_MODEL = "qwen3-vl-32b-instruct-mlx"

    # LATEX_TEMPLATE stays "minimalistic" — the test_workspace contains its own
    # copy of that template at test_workspace/latex_templates/minimalistic/,
    # so the chdir() calls ensure the test-local copy is used, not the real one.

    Settings.UNPAYWALL_EMAIL = "test@example.com"
    Settings.SEMANTIC_SCHOLAR_API_KEY = ""

    yield  # Run tests
    # Mocking is torn down after the session (though Python exits anyway)
