"""
FR4 — Experimentation
Requirement: The system shall generate code, execute it, and save output artifacts.

Pass condition: The output directory contains a generated experiment script, the runner
executes it without an unhandled exception, and at least one artifact is saved.

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
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
SCRIPT_FILE = EXPERIMENTS_DIR / "experiment.py"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module", autouse=True)
def run_experimentation_phase():
    """Run the Experimentation phase inside the persistent test workspace."""
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))
    
    # 1. Ensure Context and Hypothesis exist
    if not (OUTPUT_DIR / "hypothesis.md").exists():
        from phases.context_analysis.research_context_generator import ResearchContextGenerator
        from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
        from settings import Settings
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not (OUTPUT_DIR / "research_context.md").exists():
            ResearchContextGenerator.generate_new_context()
            
        context = ResearchContextGenerator.load_research_context(str(OUTPUT_DIR / "research_context.md"))
        builder = HypothesisBuilder(
            model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
            research_context=context,
            top_limitations=[],
            num_papers_analyzed=0
        )
        builder.build_hypothesis()
    
    # 2. Run Experimentation if the script doesn't exist yet
    if not SCRIPT_FILE.exists():
        from phases.experimentation.experiment_runner import ExperimentRunner
        from phases.context_analysis.research_context_generator import ResearchContextGenerator
        from phases.hypothesis_generation.hypothesis_builder import Hypothesis
        
        context = ResearchContextGenerator.load_research_context(str(OUTPUT_DIR / "research_context.md"))
        hypothesis = Hypothesis.load_hypothesis(str(OUTPUT_DIR / "hypothesis.md"))
        
        runner = ExperimentRunner(base_output_dir=str(EXPERIMENTS_DIR))
        
        # We run the automated generation, which saves out the experiment.py
        runner.run_experimentation_phase(
            hypothesis=hypothesis,
            research_context=context,
            user_code=[]
        )
        
    os.chdir(original_cwd)


def test_fr4_script_exists():
    """FR4: The output directory contains a generated experiment script."""
    assert OUTPUT_DIR.exists(), f"Output directory not found at {OUTPUT_DIR}. Phase execution failed."
    assert EXPERIMENTS_DIR.exists(), f"Experiments directory not found at {EXPERIMENTS_DIR}."
    assert SCRIPT_FILE.exists(), f"{SCRIPT_FILE.name} not found in {EXPERIMENTS_DIR}."
    assert SCRIPT_FILE.stat().st_size > 0, f"{SCRIPT_FILE.name} is empty."


def test_fr4_runner_executes_without_exception():
    """
    FR4: The runner executes the script without an unhandled exception.
    We test this by running the script as a subprocess and verifying exit code 0.
    """
    assert SCRIPT_FILE.exists(), f"Script {SCRIPT_FILE.name} must exist to execute it."
    
    # Run the script using the current Python interpreter
    result = subprocess.run(
        [sys.executable, str(SCRIPT_FILE)],
        cwd=str(TEST_WORKSPACE),
        capture_output=True,
        text=True
    )
    
    # We verify that it exited successfully.
    assert result.returncode == 0, f"Experiment script failed with exit code {result.returncode}.\nStderr: {result.stderr}\nStdout: {result.stdout}"


def test_fr4_artifacts_saved():
    """
    FR4: At least one artifact is saved after execution.
    The script typically saves out .json results or .pdf/.png plots in the experiments directory.
    """
    assert EXPERIMENTS_DIR.exists(), "Experiments directory must exist."
    
    all_files = list(EXPERIMENTS_DIR.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    
    # Filter out known generated code/plan definitions
    excluded_names = {"experiment.py", "experiment_plan.md", "hypothesis.md", ".DS_Store"}
    artifacts = [f for f in files_only if f.name not in excluded_names and not f.name.endswith(".pyc")]
    
    assert len(artifacts) > 0, f"No output artifacts found in {EXPERIMENTS_DIR}."

