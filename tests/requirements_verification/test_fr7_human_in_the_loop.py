"""
FR7 — Human-in-the-Loop
Requirement: The system shall persist each phase's output so the user can review and edit it
             between phases.

Pass condition: After a phase completes, its output is saved to disk. When the next phase
                executes, the exact contents of that file (including any manual edits) are
                successfully loaded into the prompt payload sent to the inference engine.

Method: Input Interception
  To avoid relying on non-deterministic LLM behavior, this test does not check the generated
  output of Phase 2. Instead, it:
    1. Writes a mock hypothesis.md file containing a unique, un-hallucinatable UUID token.
    2. Runs the next phase's prompt construction logic (Experimentation Phase).
    3. Intercepts the generated prompt payload right before it would be sent to the LLM.
    4. Asserts the UUID is present in the prompt string.
"""

from pathlib import Path
import os
import json
import uuid
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKSPACE = PROJECT_ROOT / "tests" / "requirements_verification" / "test_workspace"
OUTPUT_DIR = TEST_WORKSPACE / "output"
HYPOTHESIS_FILE = OUTPUT_DIR / "hypothesis.md"


def test_fr7_human_edit_interception():
    """
    FR7: Simulates a human editing a file between phases, then proves the system
    loads that exact edited content into the next phase's prompt payload.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from phases.experimentation.experiment_runner import ExperimentRunner
    from phases.hypothesis_generation.hypothesis_builder import Hypothesis
    from phases.context_analysis.research_context_generator import ResearchContext
    import unittest.mock as mock
    
    # Use the test workspace
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))
    
    # A unique token that an LLM could never generate by chance.
    mock_token = f"HUMAN-EDIT-TOKEN-{uuid.uuid4()}"
    
    # Recreate the exact text structure the system uses
    edited_hypothesis_text = f"""# Research Hypothesis
    
## Description
This is a mock description containing {mock_token}.

## Rationale
Mock rationale.

## Success Criteria
Mock success criteria.
"""
    # Simulate human explicitly saving this file
    HYPOTHESIS_FILE.write_text(edited_hypothesis_text, encoding="utf-8")
    
    # We parse it just like the Orchestrator does before passing to the Runner
    from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
    mock_hypothesis = HypothesisBuilder.load_hypothesis(str(HYPOTHESIS_FILE))
    
    mock_context = ResearchContext(
        description="Mock Description",
        code_snippets="",
        open_questions=""
    )
    
    # 2. Setup the Interceptor
    intercepted_payload = None
    
    class MockModel:
        def respond(self, prompt, **kwargs):
            nonlocal intercepted_payload
            intercepted_payload = prompt
            class DummyResult:
                content = "dummy"
            return DummyResult()
            
    with mock.patch("lmstudio.llm", return_value=MockModel()):
        # 3. Trigger the next phase's prompt construction
        runner = ExperimentRunner(base_output_dir=str(OUTPUT_DIR))
        
        # We call the internal plan generation method directly, which builds the
        # large prompt string containing the hypothesis, and sends it to the LLM.
        runner._generate_experiment_plan(
            hypothesis=mock_hypothesis,
            research_context=mock_context
        )
        
    os.chdir(original_cwd)
    
    # 4. Assert
    assert intercepted_payload is not None, "Failed to intercept the prompt payload."
    
    payload_str = str(intercepted_payload)
    
    assert mock_token in payload_str, (
        "FR7 Failed: The human-in-the-loop edit was lost. The unique token "
        "was not found in the payload constructed for the inference engine."
    )

