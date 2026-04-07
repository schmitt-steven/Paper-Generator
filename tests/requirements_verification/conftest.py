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
    """Pytest plugin that saves test results to test_runs/<label>_<timestamp>/."""

    def __init__(self):
        self._ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir: Path | None = None
        self._test_results: list[dict] = []

    def pytest_sessionstart(self, session):
        args = session.config.args
        if len(args) == 1:
            stem = Path(args[0].split("::")[0]).stem
            label = stem if stem else "all"
        else:
            label = "all"

        folder_name = f"{label}_{self._ts}"
        self.run_dir = RUNS_DIR / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # -- hooks ---------------------------------------------------------------

    def pytest_runtest_logreport(self, report):
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
            return
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

        lines = [
            f"Test Run : {datetime.now().isoformat()}",
            f"Commit   : {git_commit}",
            f"Platform : {platform.platform()}",
            f"Python   : {platform.python_version()}",
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
def log_evidence():
    """No-op fixture kept for backward compatibility with existing tests."""
    def _log(key, value):
        pass
    return _log


@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """Globally isolate and mock the Settings class for all verification tests."""

    # Override the save function
    def _mock_save_to_file():
        pass

    Settings.save_to_file = classmethod(lambda cls: _mock_save_to_file())

    # 2. Models used for E2E verification
    test_llm = "qwen/qwen3.5-35b-a3b"
    test_embedding = "text-embedding-qwen3-embedding-4b@q5_0"

    # Apply to all phases
    Settings.CODE_ANALYSIS_MODEL = test_llm
    Settings.CONTEXT_GENERATOR_MODEL = test_llm
    Settings.LITERATURE_SEARCH_MODEL = test_llm
    Settings.EXPERIMENT_PLAN_MODEL = test_llm
    Settings.EXPERIMENT_CODE_WRITE_MODEL = test_llm
    Settings.EXPERIMENT_VALIDATION_MODEL = test_llm
    Settings.EXPERIMENT_VERDICT_MODEL = test_llm
    Settings.PAPER_WRITING_MODEL = test_llm
    Settings.LATEX_GENERATION_MODEL = test_llm

    # Use other model for model selection verification
    Settings.HYPOTHESIS_BUILDER_MODEL = "qwen3.5-27b"

    # Must explicitly be embedding models
    Settings.PAPER_RANKING_EMBEDDING_MODEL = test_embedding
    Settings.PAPER_INDEXING_EMBEDDING_MODEL = test_embedding

    # Vision models
    Settings.EXPERIMENT_PLOT_CAPTION_MODEL = test_llm

    Settings.LATEX_TEMPLATE = "minimalistic"
    Settings.UNPAYWALL_EMAIL = "test@example.com"
    Settings.SEMANTIC_SCHOLAR_API_KEY = ""
    Settings.CRITIC_MAX_SEARCH_QUERIES = 1

    yield
