"""
FR7 — Human-in-the-Loop
Requirement: The system shall persist each phase's output so the user can review and edit it
             between phases.

Pass condition: After a phase completes, its output is saved to disk. When the next phase
                executes, the exact contents of that file (including any manual edits) are
                successfully loaded into the prompt payload sent to the inference engine.

Method: Input Interception
  1. Runs the Hypothesis Generation phase with a mocked LLM. Verifies its output is
     written to disk as hypothesis.md.
  2. Edits that file on disk, injecting a unique UUID token (simulating a human edit).
  3. Loads the edited file and runs the Experimentation phase's prompt construction.
  4. Intercepts the prompt payload right before it would be sent to the LLM and asserts
     the UUID token from the human edit is present.
"""

from pathlib import Path
import os
import json
import uuid
import pytest
import sys
import unittest.mock as mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKSPACE = PROJECT_ROOT / "tests" / "requirements_verification" / "test_workspace"
OUTPUT_DIR = TEST_WORKSPACE / "output"
HYPOTHESIS_FILE = OUTPUT_DIR / "hypothesis.md"


def test_fr7_human_edit_interception(log_evidence):
    """
    FR7: Phase 1 saves output to disk → human edits the file → Phase 2's prompt
    contains the edited content.
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder, Hypothesis
    from phases.experimentation.experiment_runner import ExperimentRunner
    from phases.context_analysis.research_context_generator import ResearchContext
    from phases.context_analysis.paper_specification import PaperSpecification

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(str(TEST_WORKSPACE))

    try:
        # ── Step 1: Run Phase 1 (Hypothesis Generation) with mocked LLM ──
        # The mock LLM returns a valid structured hypothesis JSON.
        phase1_response = json.dumps({
            "id": "user_hypothesis",
            "description": "Original LLM-generated description.",
            "rationale": "Original rationale.",
            "success_criteria": "Original success criteria.",
        })

        class Phase1MockModel:
            def respond(self, prompt, **kwargs):
                class Result:
                    content = phase1_response
                return Result()

        mock_context = ResearchContext(
            description="Mock research context",
            code_snippets="",
            open_questions=""
        )
        mock_spec = PaperSpecification(
            topic="Mock topic",
            hypothesis="Some raw hypothesis text",
            abstract="", introduction="", related_work="",
            methods="Mock methods", results="Mock results",
            discussion="", conclusion=""
        )

        builder = HypothesisBuilder(
            model_name="mock-model",
            research_context=mock_context,
            top_limitations=[],
            num_papers_analyzed=0
        )

        with mock.patch.object(builder, "_model", Phase1MockModel()):
            hypothesis = builder.create_hypothesis_from_user_input(mock_spec)

        # Verify Phase 1 wrote the file to disk
        assert HYPOTHESIS_FILE.exists(), (
            "FR7 Failed (Step 1): Hypothesis phase did not persist its output to disk."
        )
        saved_content = HYPOTHESIS_FILE.read_text(encoding="utf-8")
        assert "Original LLM-generated description" in saved_content

        log_evidence("phase1_file_saved", True)
        log_evidence("phase1_file_path", str(HYPOTHESIS_FILE))

        # ── Step 2: Simulate human editing the file on disk ──
        human_token = f"HUMAN-EDIT-TOKEN-{uuid.uuid4()}"
        edited_content = saved_content.replace(
            "Original LLM-generated description.",
            f"Edited description containing {human_token}."
        )
        HYPOTHESIS_FILE.write_text(edited_content, encoding="utf-8")

        log_evidence("injected_token", human_token)

        # ── Step 3: Load the edited file and run Phase 2's prompt construction ──
        edited_hypothesis = HypothesisBuilder.load_hypothesis(str(HYPOTHESIS_FILE))
        assert edited_hypothesis is not None, "Failed to load the edited hypothesis file."
        assert human_token in edited_hypothesis.description, (
            "The loaded hypothesis object does not contain the human edit token."
        )

        intercepted_payload = None

        class Phase2MockModel:
            def respond(self, prompt, **kwargs):
                nonlocal intercepted_payload
                intercepted_payload = prompt
                class Result:
                    content = "dummy plan"
                return Result()

        with mock.patch("lmstudio.llm", return_value=Phase2MockModel()):
            runner = ExperimentRunner(base_output_dir=str(OUTPUT_DIR))
            runner._generate_experiment_plan(
                hypothesis=edited_hypothesis,
                research_context=mock_context
            )

        # ── Step 4: Assert the human edit appears in Phase 2's prompt ──
        assert intercepted_payload is not None, "Failed to intercept the Phase 2 prompt payload."

        payload_str = str(intercepted_payload)
        token_found = human_token in payload_str

        token_index = payload_str.find(human_token)
        surrounding = (
            payload_str[max(0, token_index - 100):token_index + len(human_token) + 100]
            if token_found else ""
        )

        log_evidence("token_found_in_phase2_payload", token_found)
        log_evidence("payload_length_chars", len(payload_str))
        log_evidence("token_context_in_payload", surrounding)

        assert token_found, (
            "FR7 Failed: The human-in-the-loop edit was lost. The unique token written "
            "to disk between phases was not found in Phase 2's prompt payload."
        )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
