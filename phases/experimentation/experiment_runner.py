import os
import fitz # PyMuPDF
import textwrap
import traceback
import re
import json
import lmstudio as lms
from settings import Settings
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple, List, Any
from pathlib import Path
from pydantic import BaseModel
from utils.file_utils import save_json, load_json, save_markdown, load_markdown
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from phases.context_analysis.research_context_generator import ResearchContext
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.context_analysis.paper_specification import PaperSpecification
from phases.context_analysis.user_code_analysis import CodeAnalyzer, UserCode
from phases.hypothesis_generation.hypothesis_builder import Hypothesis
from phases.experimentation.experiment_state import (
    HypothesisEvaluation, ExecutionResult, CodeGenerationResult,
    ExperimentFiles, ValidationResult, VerdictResult, ExperimentResult, Plot
)
from phases.experimentation.code_executor import CodeExecutor
from utils.llm_utils import remove_thinking_blocks


EXPERIMENT_PLAN_FILE = "experiment_plan.md"


class ExperimentRunner:
    """Runs experiment to test a hypothesis."""
    
    def __init__(self, base_output_dir: str = "output/experiments"):
        self.settings = Settings
        self.executor = CodeExecutor()

        self.base_output_dir = base_output_dir        
        os.makedirs(base_output_dir, exist_ok=True)
    
    def _remove_markdown_formatting(self, code_content: str) -> str:
        """Remove markdown code block markers from code content."""
        # Remove ```python or ``` at the start
        code_content = re.sub(r'^```python\s*\n', '', code_content, flags=re.MULTILINE)
        code_content = re.sub(r'^```\s*\n', '', code_content, flags=re.MULTILINE)
        # Remove ``` at the end
        code_content = re.sub(r'\n```\s*$', '', code_content, flags=re.MULTILINE)
        code_content = re.sub(r'```\s*$', '', code_content)
        # Remove any remaining ``` markers
        code_content = re.sub(r'```python', '', code_content)
        code_content = re.sub(r'```', '', code_content)
        return code_content.strip()
    
    def _format_user_code_files(self, user_code: list[UserCode], use_signatures_only: bool = False) -> str:
        """Format user code files for inclusion in prompts.
        
        Args:
            user_code: List of UserCode objects
            use_signatures_only: If True, outputs an API reference using signatures.
                               If False, outputs full file content (for planning).
        """
        if not user_code:
            return "No user code files provided"
        
        if use_signatures_only:
            sections = ["[AVAILABLE LOCAL MODULES]", "(The following files are in your current directory. Import them if they match your needs.)"]
            for code_file in user_code:
                sections.append(f"\nModule: {code_file.file_name}")
                if code_file.signatures:
                    for sig in code_file.signatures:
                         sections.append(f"  - {sig}")
                else:
                    sections.append("  (No signatures extracted)")
                    
            return "\n".join(sections)
        else:
            sections = ["[USER CODE FILES]"]
            for code_file in user_code:
                sections.append(f"\nFile: {code_file.file_name}")
                if code_file.summary:
                    sections.append(f"Summary: {code_file.summary}")
                sections.append(code_file.file_content)
                sections.append("\n---")
            
            return "\n".join(sections)
    
    def _write_experiment_code(
        self,
        experiment_plan: str,
        hypothesis: Hypothesis,
        research_context: ResearchContext,
        output_dir: str,
        paper_specification: Optional[PaperSpecification] = None,
        user_code: Optional[list[UserCode]] = None,
        status_callback: callable = None
    ) -> CodeGenerationResult:
        """Generate experiment code in chunks, save to file, execute, and return results."""
        
        try:
            # Format paper specification section
            paper_spec_section = ""
            requirements_parts = []
            if paper_specification:
                if paper_specification.methods and paper_specification.methods.strip():
                    requirements_parts.append(f"Methods: {paper_specification.methods}")
                if paper_specification.results and paper_specification.results.strip():
                    requirements_parts.append(f"Results: {paper_specification.results}")
            
            if requirements_parts:
                paper_spec_section = "\n[PAPER SPECIFICATION]\n" + "\n\n".join(requirements_parts)
            else:
                paper_spec_section = "\n[PAPER SPECIFICATION]\nNo specific experimentation requirements provided"

            # Format user code files section
            user_code_section = ""
            code_instructions = ""
            if user_code:
                # Use signatures only for coding phase to encourage importing
                user_code_section = f"\n{self._format_user_code_files(user_code, use_signatures_only=True)}"
                code_instructions = "\n[CRITICAL: EXISTING CODE PROVIDED]\nYou have access to the local modules listed above. Review them to see if they fit your experiment plan. If they do, IMPORT them (e.g., `from my_algo import run_algo`). If they do not fit or if you need different logic, you may implement your own solution, but prefer reusing existing robust code where possible.\n\n[WARNING: GLOBAL STATE]\nCheck if the user code relies on global variables (e.g., `model = PersistentModel()` defined at module level). If so, you MUST import and use that global variable directly. Do NOT create a new local instance if the functions rely on the global one (e.g., if `propagate_reward` uses the global `model`).\n\n[WARNING: CONSTANTS]\nDo NOT hardcode state/action dimensions (e.g. `num_of_states = 5760`). Instead, READ them from the imported user code/agent if available (e.g. `agent.num_of_states`). Mismatched dimensions cause IndexErrors."
            else:
                user_code_section = "\n[USER CODE FILES]\nNo user code files provided - generate new code from scratch."
            
            # System prompt for all chunks
            system_prompt = textwrap.dedent(f"""\
                [ROLE]
                You are an expert at writing scientific experiment code in Python.

                [TASK]
                Generate experiment code in logical chunks to test a given hypothesis.

                [REQUIREMENTS]
                - Write clean, concise Python code
                - Save plots to plots/ directory as .pdf files (create with os.makedirs if needed)
                - Save results to JSON in current directory
                - Print concise, meaningful output (~100-200 lines max)
                - Output ONLY Python code, NO markdown formatting
                - Code MUST complete in under 5 minutes. Reduce iterations, computations, or parameter combinations if needed. Optimize loops and maintain scientific validity.
                - CRITICAL: Run HEADLESS - no UI windows, no matplotlib interactive mode.
                - DRY Principle: Reuse existing functions for simulation/execution logic. Do not duplicate complex loops (like game loops or training steps) for visualization - call the original functions to ensure consistency and prevent infinite loops/logic errors.
                
                [PYGAME HEADLESS MODE - MANDATORY IF USING PYGAME]
                If using pygame for game logic, you MUST initialize it in headless mode:
                ```
                import os
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                os.environ['SDL_AUDIODRIVER'] = 'dummy'
                import pygame
                pygame.init()
                # Do NOT call pygame.display.set_mode() with real dimensions
                # Instead: screen = pygame.display.set_mode((1, 1))  # minimal dummy surface
                ```
                NEVER use: pygame.display.set_mode((800, 600)) or similar real window sizes.
                NEVER use: pygame.display.flip(), pygame.event.get() in a render loop.
                Just compute game states and actions programmatically without rendering.

                [AVAILABLE PACKAGES]
                The following Python packages are available (optional - use if helpful):
                - numpy: Numerical computing, arrays, mathematical operations
                - matplotlib: Plotting and visualization (use 'Agg' backend, save to file only, NO plt.show())
                - seaborn: Statistical data visualization (built on matplotlib)
                - pygame: Game LOGIC only with headless mode as shown above
                These packages are available but not required - use them only if they help test the hypothesis.

                [HYPOTHESIS]
                Description: {hypothesis.description}
                Rationale: {hypothesis.rationale}
                Success Criteria: {hypothesis.success_criteria}

                {paper_spec_section}
                {user_code_section}
                {code_instructions}

                [EXPERIMENT_PLAN]
                {experiment_plan}"""
            )

            chat = lms.Chat(system_prompt)
            current_code = ""
            
            # Step 1/4: Imports and structures
            try:
                print(f"Generating imports and data structures...")
                chunk_message = """Generate ONLY imports and data structure definitions.

                Include:
                - All necessary imports (if using matplotlib, use matplotlib.use('Agg') BEFORE importing pyplot)
                - Any classes or data structures needed
                - Global constants

                Do NOT include algorithm implementations, experiment logic, or visualization yet.
                
                CRITICAL: Before writing code, mentally verify there are no syntax errors or obvious bugs."""
                if user_code:
                    chunk_message += "\n\nIf user code files were provided above, add their imports (e.g. `from my_file import my_func`)."
                chat.add_user_message(chunk_message)

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.0})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating imports chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Step 2/4: Algorithms
            try:
                print(f"Generating algorithm implementations...")
                chunk_message = """Implement the code for the algorithm(s) being tested and merge it with the previous response.

                Include everything from the previous response, then add:
                - The proposed method/algorithm being tested (as described in the experiment plan)
                - The baseline/comparison method (as described in the experiment plan)
                - Any helper functions needed for the algorithms
                The most important part is to implement the algorithms as described in the experiment plan.
                
                CRITICAL: Before responding, verify:
                - No infinite loops (all loops have proper termination conditions)
                - No blocking calls (no input(), no UI windows, no pygame.display)
                - All variable names and function signatures are consistent

                Output the COMPLETE code so far (imports and data structures + algorithms)."""
                if user_code:
                    chunk_message += "\n\nIf relevant, use the imported functions from the user code files to implement the logic."
                chat.add_user_message(chunk_message)

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.0})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating algorithms chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Step 3/4: Experiment setup and execution
            try:
                print(f"Generating experiment execution logic...")
                chunk_message = textwrap.dedent("""\
                    Implement the code for the experiment setup and execution and merge it with the previous response.

                    Include everything from the previous response, then add:
                    - Experiment setup and execution (as described in the experiment plan)
                    - Running the proposed method and baseline/comparison method
                    - Metric collection and measurement
                    - Save results to JSON file in current directory
                    - Concise stdout output with key metrics
                    
                    CRITICAL: Before responding, verify:
                    - Experiment will complete in under 5 minutes (reduce iterations if needed)
                    - No infinite loops, no UI/display calls
                    - All loops have reasonable bounds and termination conditions

                    Output the COMPLETE code so far (imports and data structures + algorithms + experiment).
                    Do NOT include visualization yet.
                """)
                if user_code:
                    chunk_message += "\n\nIntegrate the experiment logic with the imported user code where applicable."
                chat.add_user_message(chunk_message)

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.0})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating experiment chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Step 4/4: Visualization/Plotting
            try:
                print(f"Generating visualization code...")
                chunk_message = textwrap.dedent("""\
                    Generate the COMPLETE final code including everything from before PLUS visualization and summary.

                    Include everything from the previous response, then add:
                    - Create plots/ directory
                    - Generate comparison plots (use matplotlib Agg backend - NO plt.show())
                    - Save plots to plots/ as .pdf files
                    - CRITICAL: For EVERY plot you generate, print a text summary to stdout that describes the exact data shown. 
                      Format: "[Plot Summary: <filename>] <summary text with numbers/stats>"
                      Example: "[Plot Summary: learning_curve.pdf] RBQL reached 0.9 reward at ep 450, Standard Q at ep 720."
                    - Print concise summary of the results (NEVER guess the results, only print the actual results)
                    
                    FINAL CHECK before responding:
                    - No plt.show() calls (headless execution)
                    - No pygame.display or UI windows
                    - All loops terminate properly
                    - Code will complete in under 10 minutes

                    Output the COMPLETE, FINAL code (imports & data structures + algorithms + experiment + visualization).
                """)
                if user_code:
                    chunk_message += "\n\nIf user code files were provided above, ensure the final code integrates all existing functionality with the new experiment and visualization code."
                chat.add_user_message(chunk_message)

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.0})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating visualization chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Save the complete code (from final chunk)
            code_file_path = os.path.join(output_dir, "experiment.py")
            code_file_path = os.path.abspath(code_file_path)
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(current_code)
            print(f"Code saved to {code_file_path}")
            
            if status_callback:
                status_callback("Executing experiment code")
            print(f"Executing generated code: {code_file_path}")
            execution_result = self.executor.execute_file(code_file_path, output_dir=output_dir)
            
            return CodeGenerationResult(
                code_file_path=code_file_path,
                execution_result=execution_result
            )
        except Exception as e:
            error_msg = f"ERROR in _write_experiment_code: {e}"
            print(error_msg)
            traceback.print_exc()
            return CodeGenerationResult(
                code_file_path=None,
                execution_result=ExecutionResult(
                    stdout="",
                    stderr=str(e),
                    return_code=-1,
                    plot_files=[],
                    result_files=[]
                )
            )
    
    def _fix_experiment_code(
        self,
        code_file_path: str,
        error_message: str,
        stdout: str,
        stderr: str,
        hypothesis: Hypothesis,
        output_dir: str,
        user_code: Optional[list[UserCode]] = None,
        chat: Optional[lms.Chat] = None,
        fix_attempt: int = 1,
        max_attempts: int = 5
    ) -> tuple[ExecutionResult, lms.Chat]:
        """Fix errors in experiment code file based on execution output.
        
        Returns:
            Tuple of (ExecutionResult, Chat) - Chat is returned to maintain conversation context
        """
        
        if not os.path.exists(code_file_path):
            return (
                ExecutionResult(
                    stdout="",
                    stderr=f"Error: Code file not found at {code_file_path}",
                    return_code=-1,
                    plot_files=[],
                    result_files=[]
                ),
                chat or lms.Chat("")
            )
        
        try:
            # Read the existing code file
            with open(code_file_path, 'r', encoding='utf-8') as f:
                broken_code = f.read()
            
            # Create or use existing chat
            if chat is None:
                # First attempt - create new chat
                system_prompt = textwrap.dedent("""\
                    [ROLE]
                    You are an expert at fixing errors in Python code.

                    [TASK]
                    Fix errors in the given Python code.

                    [REQUIREMENTS]
                    1. Analyze the error message carefully to understand the root cause
                    2. Fix the underlying data structure or logic issue, not just the symptom
                    3. Preserve the original code structure and functionality - only change what's necessary
                    4. Maintain existing plot/results saving functionality
                    5. Do NOT add new functionality unrelated to fixing the errors

                    [ANALYSIS_STEPS]
                    - Read the entire code file before making changes
                    - Identify where the error occurs and trace back to find the root cause
                    - Check for inconsistencies: if similar classes/methods exist, ensure they handle data types and operations consistently
                    - Verify data type matches: ensure variables are used in contexts that match their types (e.g., tuples vs integers, correct array dimensions)
                    - Check bounds and indices: verify array/container sizes match the values being accessed
                    - Look for patterns: if one method handles something correctly, similar methods should follow the same pattern

                    [OUTPUT_FORMAT]
                    Always output the COMPLETE fixed Python code file, NO further markdown or explanations.
                """)
                
                chat = lms.Chat(system_prompt)
            
            # Build user message with context
            attempt_context = "" 
            if fix_attempt > 1:
                attempt_context = textwrap.dedent(f"""\
                    Note: This is fix attempt {fix_attempt}/{max_attempts}.
                    Previous attempts failed. Please analyze the root cause carefully.
                    
                    Critical: Read the ENTIRE code file. Compare how similar classes/methods handle the same operations.
                    Look for inconsistencies in data type handling, index calculations, or state conversions.
                    The error message tells you WHERE it fails - trace back to find WHY it fails.
                """)
            
            # Detect timeout errors and add specific guidance
            timeout_context = ""
            is_timeout = "timeout" in error_message.lower() or "timed out" in error_message.lower() or "time limit" in stderr.lower()
            if is_timeout:
                timeout_context = textwrap.dedent("""\
                    [TIMEOUT ERROR DETECTED]
                    The code took too long to execute. You MUST fix this by:
                    1. REDUCE iterations/episodes/trials (e.g., 1000 -> 100, 100 -> 20)
                    2. REMOVE any pygame.display, UI windows, or interactive elements - run HEADLESS
                    3. REMOVE any plt.show() calls - save plots directly without display
                    4. CHECK for infinite loops - ensure all while loops have proper termination
                    5. SIMPLIFY computations - use vectorized numpy operations where possible
                    6. REDUCE parameter combinations if doing grid search
                    The experiment MUST complete in under 5 minutes total.
                """)
            
            # Truncate long outputs to avoid context truncation
            stdout_preview = stdout[:1000] if len(stdout) > 2000 else stdout
            stderr_preview = stderr[:1000] if len(stderr) > 2000 else stderr

            # Add user code context if valid
            user_code_section = ""
            if user_code:
                user_code_section = f"\n[AVAILABLE USER CODE]\n{self._format_user_code_files(user_code, use_signatures_only=True)}\n"
            
            user_message = textwrap.dedent(f"""\
                [TASK]
                Fix the errors in this Python code.

                {attempt_context}
                {timeout_context}
                {user_code_section}

                [CODE_TO_FIX]
                ```python
                {broken_code}
                ```

                [ERROR_INFORMATION]
                Error Message: {error_message}
                STDOUT: {stdout_preview}
                STDERR: {stderr_preview}

                [INSTRUCTIONS]
                Analyze the error carefully and fix all faulty parts of the code.
                [WARNING: GLOBAL STATE] If user code has global variables (like `model = ...`), IMPORT and USE them. Do not shadow them with local instances.
                [WARNING: CONSTANTS] Do NOT hardcode state dimensions. Use `agent.num_of_states` or similar from user code.
                
                [OUTPUT_REQUIREMENT]
                IMPORTANT: Output the COMPLETE fixed Python code file from start to finish. Do not truncate or omit any parts.
            """)

            print(f"Fixing experiment code (attempt {fix_attempt}/{max_attempts}): {code_file_path}")
            chat.add_user_message(user_message)
            model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
            result = model.respond(chat, config={"temperature": 0.0})
            cleaned_code = remove_thinking_blocks(result.content)
            
            # Remove markdown code block markers
            cleaned_code = self._remove_markdown_formatting(cleaned_code)
            
            # Save the fixed code back to the file
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            print(f"Executing code after fix: {code_file_path}")
            execution_result = self.executor.execute_file(code_file_path, output_dir=output_dir)
            
            return execution_result, chat
        except Exception as e:
            error_msg = f"ERROR in _fix_experiment_code: {e}"
            print(error_msg)
            traceback.print_exc()
            return (
                ExecutionResult(
                    stdout="",
                    stderr=str(e),
                    return_code=-1,
                    plot_files=[],
                    result_files=[]
                ),
                chat or lms.Chat("")
            )
    
    def _validate_experiment_results(
        self,
        execution_result: ExecutionResult,
        experiment_plan: str,
        hypothesis: Hypothesis,
        code_file_path: str,
        user_code: Optional[list[UserCode]] = None
    ) -> ValidationResult:
        """Validate that experiment results are sound/meaningful."""

        # Read generated code
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            code_content = f"Could not read code: {e}"
        
        # Prepare result summary
        stdout_summary = execution_result.stdout
        plot_count = len(execution_result.plot_files)
        result_file_count = len(execution_result.result_files)
        
        system_prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert at debugging scientific experiment code and validating results.

            [TASK]
            Analyze experiment code and outputs to determine if the experiment ran correctly and produced valid results.

            [VALIDATION_APPROACH]
            1. Check OUTPUTS first: Did the experiment produce the expected files and metrics?
            2. Check RESULTS: Are the values plausible? (e.g., NaN, all zeros, identical values across conditions = red flag)
            3. Check CODE only if results look wrong: Trace the specific bug causing the invalid output.
            
            [COMMON BUGS TO LOOK FOR]
            - Global vs local variable confusion (function modifies wrong variable)
            - Off-by-one errors in convergence checks
            - Metrics that can never reach threshold (e.g., cumulative avg that's dragged down by early failures)
            - Missing resets between runs (state pollution across trials)
            - Major logic errors (e.g. infinite loops, zero updates). 
            
            [IMPORTANT]
            - Do NOT nitpick algorithm implementation details if the results look plausible!
            - Focus on CRITICAL failures (crashes, NaNs, zero results).
            - If it runs and produces clean plots, lean towards VALID.
            - Only report bugs you can trace to specific line numbers
            - Don't guess or hallucinate issues that aren't in the code"""
        )
                    
        validation_prompt = textwrap.dedent(f"""\
            [HYPOTHESIS]
            {hypothesis.description}

            {f"\n[AVAILABLE USER CODE]\n{self._format_user_code_files(user_code, use_signatures_only=True)}\n" if user_code else ""}
            
            Success Criteria: {hypothesis.success_criteria}

            [EXPERIMENT_PLAN]
            {experiment_plan}

            [EXECUTION SUMMARY]
            - Return Code: {execution_result.return_code}
            - Plots Generated: {plot_count} ({', '.join([os.path.basename(p) for p in execution_result.plot_files]) if execution_result.plot_files else 'none'})
            - JSON Results: {result_file_count} file(s)

            [STDOUT]
            {stdout_summary}

            [CODE]
            ```python
            {code_content}
            ```

            [VALIDATION TASK]
            1. Are the results valid? (Yes/No)
            2. If No: What specific bug causes the invalid results? Cite line numbers.
            3. What fix is needed? (Be specific, not generic)
            
            Keep response concise. No fluff."""
        )
        
        try:
            chat = lms.Chat(system_prompt)
            chat.add_user_message(validation_prompt)
            model = lms.llm(self.settings.EXPERIMENT_VALIDATION_MODEL)
            result = model.respond(chat, response_format=ValidationResult, config={"timeout": 120})
            parsed_dict = result.parsed
            
            validation_result = ValidationResult(**parsed_dict)
            
            return validation_result
        except Exception as e:
            print(f"ERROR in validation: {e}")
            traceback.print_exc()
            # Default to invalid if validation fails
            return ValidationResult(
                is_valid=False,
                reasoning=f"Validation check failed: {str(e)}",
                issues="Validation system error"
            )
    
    def _improve_experiment_code(
        self,
        code_file_path: str,
        validation_result: ValidationResult,
        hypothesis: Hypothesis,
        output_dir: str,
        user_code: Optional[list[UserCode]] = None
    ) -> ExecutionResult:
        """
        Improve experiment code based on validation feedback.
        
        Returns:
            ExecutionResult from executing the improved code
        """
        print(f"Improving experiment code based on validation feedback...")
        
        if not os.path.exists(code_file_path):
            return ExecutionResult(
                stdout="",
                stderr=f"Error: Code file not found at {code_file_path}",
                return_code=-1,
                plot_files=[],
                result_files=[]
            )
        
        try:
            # Read the existing code file
            with open(code_file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()
            
            # Format validation feedback
            feedback_text = f"Reasoning: {validation_result.reasoning}"
            if validation_result.issues:
                feedback_text += f"\n\nIssues identified:\n{validation_result.issues}"
            
            prompt = textwrap.dedent(f"""\
                [ROLE]
                You are an expert at improving scientific experiment code.

                [TASK]
                Improve the given experiment code based on validation feedback.

                [REQUIREMENTS]
                1. Address all issues identified in the validation feedback as well as possible.
                2. Ensure the code actually tests the hypothesis as described in the experiment plan
                3. Ensure plots are saved to "plots/" directory as .pdf files (relative to execution directory) - create this directory if needed using os.makedirs("plots", exist_ok=True)
                4. Save detailed results/metrics to JSON file in the current directory (do NOT create an "output" directory - the code already runs from the output directory)
                5. Ensure stdout output is concise and meaningful - key metrics, conclusions and results only, avoid loop spam
                6. Make sure the experiment is complete and meaningful (e.g., not too short, collects proper metrics, etc.)
                7. Preserve any working, valid parts of the code
                8. [WARNING: GLOBAL STATE] If user code has global variables (like `model = ...`), IMPORT and USE them. Do not shadow them with local instances.
                9. [WARNING: CONSTANTS] Do NOT hardcode state dimensions. Use `agent.num_of_states` or similar from user code.
                10. DRY Principle: Reuse simulation functions (e.g. `run_episode`) for plotting/metrics. Do not re-write loops inline.

                [AVAILABLE_PACKAGES]
                The following Python packages are available (optional - use if helpful):
                - numpy: Numerical computing, arrays, mathematical operations
                - matplotlib: Plotting and visualization (use 'Agg' backend, save to file only - NO interactive display)
                - seaborn: Statistical data visualization (built on matplotlib)
                - pygame: Game logic ONLY - do NOT create display windows, run headless
                CRITICAL: Run HEADLESS - no UI windows, no pygame display, no matplotlib.show(). Use 'Agg' backend.
                These packages are available but not required - use them only if they help test the hypothesis.

                {f"\n[AVAILABLE USER CODE]\n{self._format_user_code_files(user_code, use_signatures_only=True)}\n" if user_code else ""}

                [HYPOTHESIS]
                Description: {hypothesis.description}
                Rationale: {hypothesis.rationale}
                Success Criteria: {hypothesis.success_criteria}

                [CURRENT_CODE]
                ```python
                {current_code}
                ```

                [VALIDATION_FEEDBACK]
                {feedback_text}

                [OUTPUT_FORMAT]
                Output ONLY the improved Python code, NO further markdown or explanations. Your answer will be saved to a code file.
            """)

            model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
            result = model.respond(prompt, config={"temperature": 0.0})
            improved_code = remove_thinking_blocks(result.content)
            
            # Remove markdown code block markers
            improved_code = self._remove_markdown_formatting(improved_code)
            
            # Save the improved code back to the file
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(improved_code)
            
            # Automatically execute the improved code
            print(f"Executing improved code: {code_file_path}")
            execution_result = self.executor.execute_file(code_file_path, output_dir=output_dir)
            
            return execution_result
        except Exception as e:
            error_msg = f"ERROR in _improve_experiment_code: {e}"
            print(error_msg)
            traceback.print_exc()
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                plot_files=[],
                result_files=[]
            )
    
    def _generate_plot_captions(
        self,
        plot_files: list[str],
        hypothesis: Hypothesis,
        experiment_plan: str,
        stdout: str
    ) -> list[Plot]:
        """Generate captions for plot files using LM Studio VLM API."""
        
        if not plot_files:
            return []
        
        plots = []
        
        # System prompt for plot caption generation
        plot_caption_prompt = """<role>
You write figure captions for scientific research papers.
</role>

<task>
Generate a publication-quality figure caption for the provided plot.
</task>

<requirements>
- State WHAT the figure shows, not WHY or HOW
- Use past tense for methods, present tense for results
- No interpretation, speculation, or methodology explanation
- No phrases like "This figure shows", "As we can see", "The plot demonstrates"
- Start directly with the subject (e.g., "Comparison of...", "Performance of...", "Distribution of...")
- 1-2 sentences maximum
- Include specific metrics/values visible in the figure when relevant
- Use precise scientific terminology
</requirements>

<good_examples>
- "Comparison of convergence rates between RBQL and standard Q-learning across 1000 episodes. RBQL achieves 95% optimal performance at episode 450 vs. 720 for Q-learning."
- "Distribution of prediction errors for the three model variants. Median error for Model A: 0.023, Model B: 0.041, Model C: 0.089."
- "Training loss curves for baseline (blue) and proposed method (orange) over 50 epochs."
</good_examples>

<bad_examples>
Do NOT do this:
- "This figure demonstrates how our method outperforms the baseline..." (interpretation)
- "As we can see from the plot, the results clearly show that..." (filler phrases)
- "The experiment was designed to test whether..." (methodology)
</bad_examples>"""
        
        for plot_file in plot_files:
            # Extract just the filename from the full path
            filename = os.path.basename(plot_file)
            
            # Prepare the image for VLM
            try:
                # Handle PDF files by converting first page to image
                image_path = plot_file
                temp_image = None
                
                if plot_file.lower().endswith('.pdf'):
                    try:
                        doc = fitz.open(plot_file)
                        if len(doc) > 0:
                            page = doc[0]
                            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) # 1.5x zoom (balance quality/speed)
                            temp_image = plot_file.replace('.pdf', '_preview.png')
                            pix.save(temp_image)
                            image_path = temp_image
                        doc.close()
                    except Exception as pdf_err:
                        print(f"Warning: Failed to convert PDF {filename} to image: {pdf_err}")
                        # Continue with original file, likely to fail but worth a try if VLM supports PDF
                
                image_handle = lms.prepare_image(image_path)
                
                # Clean up temp file
                if temp_image and os.path.exists(temp_image):
                    # We can't delete immediately if prepare_image is lazy, but usually it reads bytes
                    # Let's verify lms behavior or just keep it for now and relying on OS cleanup or overwrite
                    pass 
            except Exception as e:
                print(f"ERROR: Failed to prepare image {filename}: {e}")
                traceback.print_exc()
                # Fallback caption without image
                plots.append(Plot(filename=plot_file, caption=f"Experimental results for: {hypothesis.description}"))
                continue
            
            user_message = textwrap.dedent(f"""\
                <context>
                Hypothesis: {hypothesis.description}
                Success Criteria: {hypothesis.success_criteria}
                </context>

                <experiment_output>
                {stdout[-2000:] if len(stdout) > 2000 else stdout}
                </experiment_output>

                <output_format>
                Write ONLY the caption (1-2 sentences). No preamble, no explanation.
                </output_format>""")
                        
            try:
                chat = lms.Chat(plot_caption_prompt)
                chat.add_user_message(user_message, images=[image_handle])
                model = lms.llm(self.settings.EXPERIMENT_PLOT_CAPTION_MODEL)
                result = model.respond(chat, config={"temperature": 0.0, "timeout": 120})
                caption = remove_thinking_blocks(result.content).strip()
                if caption.startswith('"') and caption.endswith('"'):
                    caption = caption[1:-1]
                plots.append(Plot(filename=plot_file, caption=caption))
            except Exception as e:
                print(f"ERROR: Failed to generate caption for {filename}: {e}")
                traceback.print_exc()
                # Fallback caption
                plots.append(Plot(filename=plot_file, caption=f"Experimental results for: {hypothesis.description}"))
                print(f"ERROR: Failed to generate caption for {filename}: {e}")
                traceback.print_exc()
                # Fallback caption
                plots.append(Plot(filename=plot_file, caption=f"Experimental results for: {hypothesis.description}"))
        
        return plots
    
    def _generate_experiment_plan(
        self,
        hypothesis: Hypothesis,
        research_context: ResearchContext,
        paper_specification: Optional[PaperSpecification] = None,
        user_code: Optional[list[UserCode]] = None
    ) -> str:
        """Generate a detailed experiment plan for testing a hypothesis."""

        # Format paper specification section
        paper_spec_section = ""
        requirements_parts = []
        if paper_specification:
            if paper_specification.methods and paper_specification.methods.strip():
                requirements_parts.append(f"Methods: {paper_specification.methods}")
            if paper_specification.results and paper_specification.results.strip():
                requirements_parts.append(f"Results: {paper_specification.results}")
        
        if requirements_parts:
            paper_spec_section = "[PAPER SPECIFICATION]\n" + "\n\n".join(requirements_parts)
        else:
            paper_spec_section = "[PAPER SPECIFICATION]\nNo specific experimentation requirements provided"

        # Format user code files section
        user_code_section = ""
        if user_code:
            user_code_section = self._format_user_code_files(user_code)
        else:
            user_code_section = "[USER CODE FILES]\nNo user code files provided"

        # Build instructions based on whether user code exists
        code_instructions = ""
        if user_code:
            code_instructions = "\n[IMPORTANT: USER CODE PROVIDED]\nBuild upon and adapt the existing user code files provided below. Do not start from scratch - extend and modify the existing code to test the hypothesis. Identify what needs to be added, modified, or adapted in the existing code."

        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"

        prompt = textwrap.dedent(f"""\
            [TASK]
            Create a detailed, concise experiment plan for testing a given hypothesis.
            The plan will be used to generate the experiment code in Python.
            You must output only the plan/description. Do NOT generate the full code yet.
            Focus on describing the experiment design, setup, metrics, and approach in natural language.

            [PLAN_REQUIREMENTS]
            Include:
            - Objective and success criteria
            - Required mathematical formulas/technical details
            - Experiment setup
            - Metrics to measure
            - Implementation approach
            - Output requirements: 
              * Detailed results/metrics stored in JSON file
              * Concise, meaningful output to stdout (key metrics, conclusions)
              * Plot(s) for visualization (saved as .pdf)
            - Experiment MUST complete in under 5 minutes. Use reasonable parameter ranges and reduce iterations/computations/parameter combinations if needed.
            
            [CRITICAL: HEADLESS EXECUTION MANDATORY]
            - NO pygame display windows - if using pygame for game logic, initialize with:
              os.environ['SDL_VIDEODRIVER'] = 'dummy'
              os.environ['SDL_AUDIODRIVER'] = 'dummy'
            - NO plt.show() - save plots directly to file with savefig()
            - NO interactive visualizations or UI of any kind
            - If testing game/RL algorithms: simulate game logic WITHOUT rendering frames, just compute states/actions/rewards programmatically

            {title_section}[RESEARCH_CONTEXT]
            {research_context.description}

            [HYPOTHESIS]
            Description: {hypothesis.description}
            Rationale: {hypothesis.rationale}
            Success Criteria: {hypothesis.success_criteria}

            {paper_spec_section}

            {user_code_section}
            {code_instructions}"""
        )

        try:
            model = lms.llm(self.settings.EXPERIMENT_PLAN_MODEL)
            result = model.respond(prompt, config={"temperature": 0.0})
            return remove_thinking_blocks(result.content)
        except Exception as e:
            print(f"ERROR: Failed to generate experiment plan: {e}")
            traceback.print_exc()
            raise
    
    def save_experiment_plan(
        self,
        experiment_plan: str
    ) -> str:
        """Save an experiment plan to a file."""

        file_path = save_markdown(experiment_plan, EXPERIMENT_PLAN_FILE, self.base_output_dir)
        
        return file_path
    
    def load_experiment_plan(
        self
    ) -> str:
        """Load an experiment plan from a file."""

        file_path = os.path.join(self.base_output_dir, EXPERIMENT_PLAN_FILE)

        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Experiment plan not found: {file_path}")

        plan_content = load_markdown(path_obj.name, str(path_obj.parent))

        return plan_content
    
    def load_experiment_code(self) -> str:
        """Load experiment code from a file."""
        
        filename = "experiment.py"
        file_path = os.path.join(self.base_output_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment code not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def save_hypothesis_evaluation(
        self,
        evaluation: HypothesisEvaluation
    ) -> str:
        """Save hypothesis evaluation (proven/disproven/inconclusive)."""

        eval_data = {
            "hypothesis_id": evaluation.hypothesis_id,
            "verdict": evaluation.verdict,
            "reasoning": evaluation.reasoning
        }

        filename = "hypothesis_evaluation.json"
        eval_path = save_json(eval_data, filename, self.base_output_dir)

        return eval_path
    
    @staticmethod
    def load_previous_results(
        run_id: Optional[int] = None,
        base_dir: str = "output/experiments"
    ) -> dict[str, Any]:
        """Load previous experiment results for comparison."""

        result_data = {}
        eval_path = os.path.join(base_dir, "hypothesis_evaluation.json")
        if os.path.exists(eval_path):
            path_obj = Path(eval_path)
            result_data = load_json(path_obj.name, str(path_obj.parent))

        return result_data

    
    def load_experiment_files(self) -> ExperimentFiles:
        """Load plan and experiment code files."""
        
        return ExperimentFiles(
            experiment_plan=self.load_experiment_plan(),
            experiment_code=self.load_experiment_code(),
            plan_file_path=os.path.join(self.base_output_dir, EXPERIMENT_PLAN_FILE),
            code_file_path=os.path.join(self.base_output_dir, "experiment.py")
        )
    
    def save_experiment_result(
        self,
        experiment_result: ExperimentResult
    ) -> str:
        """Save ExperimentResult to JSON file."""

        # Convert ExperimentResult to dictionary
        result_dict = self._experiment_result_to_dict(experiment_result)

        filename = "experiment_result.json"
        file_path = save_json(result_dict, filename, self.base_output_dir)

        return file_path
    
    def _experiment_result_to_dict(self, result: ExperimentResult) -> dict:
        """Convert ExperimentResult to dictionary for JSON serialization."""
        
        def convert_value(value):
            """Recursively convert values to JSON-serializable format."""
            if isinstance(value, BaseModel):
                # Pydantic model
                return value.model_dump() if hasattr(value, 'model_dump') else value.dict()
            elif is_dataclass(value):
                # Dataclass - convert to dict recursively
                return {k: convert_value(v) for k, v in asdict(value).items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value
        
        # Start with the dataclass as a dict
        result_dict = asdict(result)
        # Recursively convert nested objects
        return convert_value(result_dict)
    
    @staticmethod
    def load_experiment_result(file_path: str) -> ExperimentResult:
        """Load ExperimentResult from JSON file."""

        path_obj = Path(file_path)
        data = load_json(path_obj.name, str(path_obj.parent))
        
        # Reconstruct nested objects
        hypothesis = Hypothesis(**data['hypothesis'])
        
        execution_result = ExecutionResult(**data['execution_result'])
        
        validation_result = ValidationResult(**data['validation_result'])
        
        hypothesis_evaluation = HypothesisEvaluation(**data['hypothesis_evaluation'])
        
        plots = [Plot(**plot_data) for plot_data in data.get('plots', [])]
        
        # Load experiment_code from saved data, or try to load from file if not present
        experiment_code = data.get('experiment_code')
        if not experiment_code:
            # Try to load from the experiment code file
            import os
            base_dir = os.path.dirname(file_path)
            code_file_path = os.path.join(base_dir, "experiment.py")
            if os.path.exists(code_file_path):
                try:
                    with open(code_file_path, 'r', encoding='utf-8') as f:
                        experiment_code = f.read()
                except Exception:
                    experiment_code = ""
            else:
                experiment_code = ""
        
        experiment_result = ExperimentResult(
            hypothesis=hypothesis,
            experiment_plan=data['experiment_plan'],
            experiment_code=experiment_code,
            execution_result=execution_result,
            validation_result=validation_result,
            hypothesis_evaluation=hypothesis_evaluation,
            plots=plots,
            fix_attempts=data.get('fix_attempts', 0),
            validation_attempts=data.get('validation_attempts', 0),
            execution_time=data.get('execution_time')
        )
        
        return experiment_result
    
    def _determine_verdict(
        self,
        hypothesis: Hypothesis,
        stdout_summary: str,
        plot_captions: list[Plot],
        validation_warning: str = ""
    ) -> Tuple[str, str]:
        """Determine verdict and reasoning using LLM."""
        
        # Build plot captions text
        plot_captions_text = ""
        if plot_captions:
            plot_captions_text = "\n\nGenerated Plot Captions:\n"
            for i, plot in enumerate(plot_captions, 1):
                plot_captions_text += f"{i}. {os.path.basename(plot.filename)}: {plot.caption}\n"
        
        # Build context for verdict determination
        verdict_prompt = textwrap.dedent(f"""\
            [ROLE]
            You are evaluating the results of a scientific experiment to test a hypothesis.

            [HYPOTHESIS]
            Description: {hypothesis.description}
            Success Criteria: {hypothesis.success_criteria}

            [EXPERIMENT OUTPUT]
            {stdout_summary}

            [PLOT CAPTIONS]
            {plot_captions_text}
            {validation_warning}

            [TASK]
        Determine if the hypothesis is PROVEN, DISPROVEN, or INCONCLUSIVE based on the evidence.
        
        [GUIDELINES]
        1. Apply scientific judgement. Do not be robotically strict about exact numerical thresholds if the trend is overwhelming and statistically significant.
        2. If the results strongly support the core hypothesis but miss a specific metric by a negligible margin (e.g. 0.89 vs 0.90), rule PROVEN.
        3. If the results match the expected behavior/trend described in the hypothesis, favor PROVEN.
        4. Only rule DISPROVEN if the results directly contradict the hypothesis.
        5. Only rule INCONCLUSIVE if the data is messy, contradictory, or the code failed to produce meaningful metrics.
        6. Your goal is to validate the scientific discovery, not to act as a harsh gatekeeper. If the experiment works, say so!

        Focus on whether the CORE SCIENTIFIC CLAIM is supported by the data.
            
            Provide:
            1. Your verdict: 'proven', 'disproven', or 'inconclusive'
            2. Brief reasoning based on the success criteria and observed results
        """)
        
        try:
            model = lms.llm(self.settings.EXPERIMENT_VERDICT_MODEL)
            result = model.respond(verdict_prompt, response_format=VerdictResult)
            parsed_dict = result.parsed
            
            verdict_result = VerdictResult(**parsed_dict)
            verdict = verdict_result.verdict.strip().lower()
            reasoning = verdict_result.reasoning
            
            # Validate verdict
            if verdict not in ["proven", "disproven", "inconclusive"]:
                verdict = "inconclusive"
                reasoning += f"\\n\\nNote: Invalid verdict '{verdict_result.verdict}' was returned, defaulting to 'inconclusive'."
            
            return verdict, reasoning
        except Exception as e:
            print(f"ERROR: Failed to get verdict: {e}")
            traceback.print_exc()
            return "inconclusive", f"Failed to determine verdict: {str(e)}"

    def run_experiment(
        self,
        hypothesis: Hypothesis,
        research_context: ResearchContext,
        load_existing_plan: bool = False,
        load_existing_code: bool = False,
        status_callback: callable = None
    ) -> ExperimentResult:
        """Run experiment to test hypothesis.
        
        Args:
            status_callback: Optional callback function(str) for progress updates.
        """
        
        # Ensure output directory exists
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Clear plots directory if it exists to prevent mixing results
        plots_dir = os.path.join(self.base_output_dir, "plots")
        if os.path.exists(plots_dir):
            for file in os.listdir(plots_dir):
                file_path = os.path.join(plots_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Warning: Failed to delete {file_path}: {e}")
        else:
             os.makedirs(plots_dir, exist_ok=True)
        
        # Load paper specification and user code
        if status_callback:
            status_callback("Loading paper specification and code")
        paper_specification = None
        user_code = None
        try:
            paper_specification = PaperSpecification.load("user_files/paper_specification.md")
        except FileNotFoundError:
            print("Paper specification file not found, proceeding without it")
        except Exception as e:
            print(f"Warning: Failed to load paper specification: {e}")
        
        try:
            code_analyzer = CodeAnalyzer(model_name=self.settings.CODE_ANALYSIS_MODEL)
            user_code = code_analyzer.load_code_files("user_files")
            # Analyze semantic content and extract signatures
            if user_code:
                user_code = code_analyzer.analyze_all_files(user_code)
                
                # Clean old user code files before copying new ones (prevent stale imports)
                import shutil
                for existing_file in os.listdir(self.base_output_dir):
                    if existing_file.endswith('.py') and existing_file != 'experiment.py':
                        try:
                            os.unlink(os.path.join(self.base_output_dir, existing_file))
                            print(f"Removed old user file: {existing_file}")
                        except Exception as e:
                            print(f"Warning: Failed to delete old file {existing_file}: {e}")
                
                # Copy user code files to experiment directory so they can be imported
                print(f"Copying {len(user_code)} user file(s) to experiment directory...")
                for code_file in user_code:
                    src_path = code_file.file_path
                    dest_path = os.path.join(self.base_output_dir, code_file.file_name)
                    try:
                        shutil.copy2(src_path, dest_path)
                    except Exception as e:
                        print(f"Error copying {code_file.file_name}: {e}")
                        
        except Exception as e:
            print(f"Warning: Failed to load user code files: {e}")
            user_code = None
        
        # Generate or load experiment plan
        try:
            plan_file_path = os.path.join(self.base_output_dir, EXPERIMENT_PLAN_FILE)
            if load_existing_plan and os.path.exists(plan_file_path):
                if status_callback:
                    status_callback("Loading existing experiment plan")
                print(f"Loading existing experiment plan...")
                experiment_plan = self.load_experiment_plan()
            else:
                if load_existing_plan:
                    print(f"Experiment plan not found, generating new plan...")
                else:
                    if status_callback:
                        status_callback("Generating experiment plan")
                    print(f"Generating new experiment plan...")
                experiment_plan = self._generate_experiment_plan(
                    hypothesis, 
                    research_context,
                    paper_specification=paper_specification,
                    user_code=user_code
                )
                self.save_experiment_plan(experiment_plan)
        except Exception as e:
            print(f"ERROR: Failed to generate/load experiment plan: {e}")
            traceback.print_exc()
            raise
        
        # Generate or load experiment code
        code_file_path = os.path.join(self.base_output_dir, "experiment.py")
        code_file_path = os.path.abspath(code_file_path)
        
        if load_existing_code and os.path.exists(code_file_path):
            if status_callback:
                status_callback("Executing existing code")
            print(f"Loading existing experiment code...")
            # Load existing code and execute it
            print(f"Executing loaded code: {code_file_path}")
            execution_result = self.executor.execute_file(code_file_path, output_dir=self.base_output_dir)
            if execution_result.return_code != 0:
                print(f"Code execution failed with return code {execution_result.return_code}")
                print(f"STDERR: {execution_result.stderr[:500] if execution_result.stderr else 'None'}")
                print(f"STDOUT: {execution_result.stdout[:500] if execution_result.stdout else 'None'}")
            else:
                print(f"Code executed successfully. Generated {len(execution_result.plot_files)} plot(s) and {len(execution_result.result_files)} result file(s)")
                if len(execution_result.plot_files) == 0 and len(execution_result.result_files) == 0:
                    # Check if files exist but weren't detected as new
                    plots_dir = os.path.join(self.base_output_dir, "plots")
                    results_file = os.path.join(self.base_output_dir, "results.json")
                    existing_plots = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.pdf'))] if os.path.exists(plots_dir) else []
                    existing_results = os.path.exists(results_file)
                    print(f"  Note: Found {len(existing_plots)} existing plot(s) and {'1' if existing_results else '0'} existing result file(s) (may have been created in previous run)")
            write_result = CodeGenerationResult(
                code_file_path=code_file_path,
                execution_result=execution_result
            )
        else:
            if load_existing_code:
                print(f"Experiment code not found, generating new code...")
            # Generate new code
            if status_callback:
                status_callback("Generating experiment code")
            write_result = self._write_experiment_code(
                experiment_plan,
                hypothesis,
                research_context,
                self.base_output_dir,
                paper_specification=paper_specification,
                user_code=user_code,
                status_callback=status_callback
            )
        
        code_file_path = write_result.code_file_path
        execution_result = write_result.execution_result
        
        # Track total attempts across all loops
        total_fix_attempts = 0
        total_validation_attempts = 0
        validation_result = None
        validation_warning = None
        
        # Fix code if execution failed
        max_fix_attempts = 5
        fix_attempt = 0
        fix_chat = None  # Chat object to keep conversation across fix attempts
        
        while execution_result.return_code != 0 and fix_attempt < max_fix_attempts:
            fix_attempt += 1
            total_fix_attempts += 1
            
            if status_callback:
                status_callback(f"Fixing code errors (attempt {fix_attempt}/{max_fix_attempts})")
            
            # Extract error information
            error_message = execution_result.stderr or "Unknown error"
            stdout = execution_result.stdout
            stderr = execution_result.stderr
            
            error_lines = stderr.strip().split('\n') if stderr else []
            # Get the last few lines of error (usually the most relevant)
            concise_error = '\n'.join(error_lines[-15:]) if len(error_lines) > 15 else stderr
            print(f"Code execution failed (attempt {fix_attempt}/{max_fix_attempts}):")
            if concise_error:
                print(f"  Error: {concise_error}")
            else:
                print(f"  Error: {error_message}")
            
            fix_result, fix_chat = self._fix_experiment_code(
                code_file_path,
                error_message,
                stdout,
                stderr,
                hypothesis,
                self.base_output_dir,
                user_code=user_code,
                chat=fix_chat,
                fix_attempt=fix_attempt,
                max_attempts=max_fix_attempts
            )
            
            execution_result = fix_result
        
        # Initialize verdict and reasoning (will be set in branches below)
        verdict = "inconclusive"
        reasoning = ""
        plot_captions = []  # Will be populated if plots exist and execution succeeds
        validation_warning = ""  # Will be populated if validation has warnings
        
        # Validate and improve results if execution succeeded
        if execution_result.return_code == 0:
            # Validate results and improve if needed
            max_validation_attempts = 3
            validation_attempt = 0
            validation_passed = False
            
            print("Validating experiment results...")
            if status_callback:
                status_callback("Validating experiment results")
            while not validation_passed and validation_attempt < max_validation_attempts:
                validation_attempt += 1
                total_validation_attempts += 1
                if status_callback:
                    status_callback(f"Validating results (attempt {validation_attempt}/{max_validation_attempts})")
                validation_result = self._validate_experiment_results(
                    execution_result,
                    experiment_plan,
                    hypothesis,
                    code_file_path,
                    user_code=user_code
                )
                
                if validation_result.is_valid:
                    validation_passed = True
                    print("Results validation passed.")
                    print(f"  Reasoning: {validation_result.reasoning}")
                else:
                    print(f"Results validation failed (attempt {validation_attempt}/{max_validation_attempts}):")
                    print(f"  Reasoning: {validation_result.reasoning}")
                    if validation_result.issues:
                        print(f"  Issues: {validation_result.issues}")
                    
                    if validation_attempt < max_validation_attempts:
                        # Improve code based on validation feedback
                        if status_callback:
                            status_callback(f"Improving code based on feedback")
                        improvement_result = self._improve_experiment_code(
                            code_file_path,
                            validation_result,
                            hypothesis,
                            self.base_output_dir,
                            user_code=user_code
                        )
                        execution_result = improvement_result
                        
                        # If improvement broke the code, go back to fix loop
                        if execution_result.return_code != 0:
                            print("Improvement introduced errors. Entering fix loop...")
                            # Re-enter fix loop (reuse fix_chat if available, otherwise create new)
                            fix_attempt = 0
                            nested_fix_chat = fix_chat  # Reuse existing chat if available
                            while execution_result.return_code != 0 and fix_attempt < max_fix_attempts:
                                fix_attempt += 1
                                total_fix_attempts += 1
                                
                                if status_callback:
                                    status_callback(f"Fixing code errors (attempt {fix_attempt}/{max_fix_attempts})")
                                
                                # Extract error information
                                error_message = execution_result.stderr or "Unknown error"
                                stdout = execution_result.stdout
                                stderr = execution_result.stderr
                                
                                # Print concise error message
                                error_lines = stderr.strip().split('\n') if stderr else []
                                concise_error = '\n'.join(error_lines[-5:]) if len(error_lines) > 5 else stderr
                                print(f"Code execution failed (attempt {fix_attempt}/{max_fix_attempts}):")
                                if concise_error:
                                    print(f"  Error: {concise_error}")
                                else:
                                    print(f"  Error: {error_message}")
                                
                                fix_result, nested_fix_chat = self._fix_experiment_code(
                                    code_file_path,
                                    error_message,
                                    stdout,
                                    stderr,
                                    hypothesis,
                                    self.base_output_dir,
                                    user_code=user_code,
                                    chat=nested_fix_chat,
                                    fix_attempt=fix_attempt,
                                    max_attempts=max_fix_attempts
                                )
                                
                                execution_result = fix_result
                            fix_chat = nested_fix_chat  # Update main fix_chat
                            
                            # If still broken after fix attempts, break validation loop
                            if execution_result.return_code != 0:
                                break
            
            # Post-loop check: If execution succeeded but validation hasn't run yet (e.g. strange loop exit)
            if execution_result.return_code == 0 and validation_result is None:
                 # Check if we should validate the final fixed code
                 print("Validating final fixed code...")
                 validation_result = self._validate_experiment_results(
                    execution_result,
                    experiment_plan,
                    hypothesis,
                    code_file_path,
                    user_code=user_code
                 )
                 if validation_result.is_valid:
                     validation_passed = True
                     print("Final validation passed.")
                     if validation_result.issues:
                         validation_warning = validation_result.issues
                 else:
                     print("Final validation failed.")
            
            # Determine verdict and reasoning
            if execution_result.return_code == 0:
                # Generate plot captions if plots exist (always generate if we have plots and valid execution)
                if execution_result.plot_files:
                    if status_callback:
                        status_callback("Generating plot captions")
                    print("Generating captions for plots...")
                    plot_captions = self._generate_plot_captions(
                        execution_result.plot_files,
                        hypothesis,
                        experiment_plan,
                        execution_result.stdout
                    )
                    print(f"Generated {len(plot_captions)} plot caption(s)")
                
                # Successful execution (even if validation warned or failed) - get verdict from LLM
                if status_callback:
                    status_callback("Determining verdict")
                print("Code executed successfully. Determining verdict...")
                
                # Truncate stdout if too long to prevent context overflow
                stdout_summary = execution_result.stdout
                if len(stdout_summary) > 2000:
                    stdout_summary = stdout_summary[:500] + "\n...[truncated output]...\n" + stdout_summary[-1500:]
                
                # Build context for verdict determination
                if validation_passed:
                    verdict, reasoning = self._determine_verdict(
                        hypothesis,
                        stdout_summary,
                        plot_captions,
                        validation_warning
                    )
                else:
                    # Execution succeeded but validation failed
                    print(f"Results validation failed after {max_validation_attempts} attempts.")
                    reasoning = f"Code executed successfully but results validation failed. Last validation reasoning: {validation_result.reasoning if validation_result else 'None'}"
                    if validation_result and validation_result.issues:
                        reasoning += f"\nIssues: {validation_result.issues}"
                    verdict = "inconclusive"
            else:
                # Execution failed after all retries
                print(f"Code execution failed after {max_fix_attempts} fix attempts.")
                reasoning = f"Experiment code failed to execute after {max_fix_attempts} fix attempts. Last error: {execution_result.stderr or 'Unknown error'}"
                verdict = "inconclusive"
                # Create a validation result for failed execution
                if validation_result is None:
                    validation_result = ValidationResult(
                        is_valid=False,
                        reasoning=reasoning,
                        issues="Execution failed"
                    )
        
        # Read experiment code from file
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                experiment_code = f.read()
        except Exception as e:
            experiment_code = f"Error reading code file: {e}"
        
        # Create hypothesis evaluation
        try:
            evaluation = HypothesisEvaluation(
                hypothesis_id=hypothesis.id,
                verdict=verdict,
                reasoning=reasoning
            )
            
            self.save_hypothesis_evaluation(evaluation)
        except Exception as e:
            print(f"ERROR: Failed to save evaluation: {e}")
            traceback.print_exc()
            # Return evaluation anyway if we can create it
            evaluation = HypothesisEvaluation(
                hypothesis_id=hypothesis.id,
                verdict=verdict,
                reasoning=reasoning
            )
        
        # Ensure validation_result is set
        if validation_result is None:
            validation_result = ValidationResult(
                is_valid=False,
                reasoning="Validation was not performed",
                issues="Execution did not succeed"
            )
        
        # Create and return ExperimentResult
        experiment_result = ExperimentResult(
            hypothesis=hypothesis,
            experiment_plan=experiment_plan,
            experiment_code=experiment_code,
            execution_result=execution_result,
            validation_result=validation_result,
            hypothesis_evaluation=evaluation,
            plots=plot_captions,
            fix_attempts=total_fix_attempts,
            validation_attempts=total_validation_attempts,
            execution_time=None  # Could be tracked if needed
        )
        
        # Save experiment result
        try:
            saved_path = self.save_experiment_result(experiment_result)
            print(f"Saved experiment result to {saved_path}")
        except Exception as e:
            print(f"ERROR: Failed to save experiment result: {e}")
            traceback.print_exc()
        
        return experiment_result

    @staticmethod
    def generate_new_experiment_plan(hypothesis: Hypothesis, status_callback: callable = None) -> str:
        """
        Generate and save experiment plan.
        
        This is the centralized orchestrator method that handles:
        1. Loading research context
        2. Loading paper specification (optional)
        3. Loading user code files (optional)
        4. Generating experiment plan using LLM
        5. Saving the result
        
        Args:
            hypothesis: The hypothesis to generate a plan for.
            status_callback: Optional callback function(str) for progress updates.
            
        Returns:
            The generated experiment plan as a string.
        """
        
        if status_callback:
            status_callback("Loading research context")
        research_context = ResearchContextGenerator.load_research_context("output/research_context.md")
        
        if status_callback:
            status_callback("Loading paper specification")
        paper_specification = None
        try:
            paper_specification = PaperSpecification.load("user_files/paper_specification.md")
        except FileNotFoundError:
            pass
        
        if status_callback:
            status_callback("Loading code files")
        user_code = None
        try:
            code_analyzer = CodeAnalyzer(model_name=Settings.CODE_ANALYSIS_MODEL)
            user_code = code_analyzer.load_code_files("user_files")
        except Exception:
            pass
        
        if status_callback:
            status_callback("Generating experiment plan")
        runner = ExperimentRunner()
        experiment_plan = runner._generate_experiment_plan(
            hypothesis, 
            research_context,
            paper_specification=paper_specification,
            user_code=user_code
        )
        runner.save_experiment_plan(experiment_plan)
        return experiment_plan

    @staticmethod
    def run_new_experiment(status_callback: callable = None) -> "ExperimentResult":
        """
        Run experiment and save results.
        
        This handles:
        1. Loading hypothesis
        2. Loading research context
        3. Running the experiment (using existing plan)
        4. Saving the result
        
        Args:
            status_callback: Optional callback function(str) for progress updates.
            
        Returns:
            The ExperimentResult object.
        """
        
        if status_callback:
            status_callback("Loading hypothesis")
        hypothesis = HypothesisBuilder.load_hypothesis("output/hypothesis.md")
        if hypothesis is None:
            raise ValueError("No hypothesis found")
        
        if status_callback:
            status_callback("Loading research context")
        research_context = ResearchContextGenerator.load_research_context("output/research_context.md")
        
        if status_callback:
            status_callback("Running experiment")
        runner = ExperimentRunner()
        return runner.run_experiment(
            hypothesis, 
            research_context,
            load_existing_plan=True,
            load_existing_code=False,
            status_callback=status_callback
        )

