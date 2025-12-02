import os
import textwrap
import traceback
import re
import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple, List
from pathlib import Path
from pydantic import BaseModel
from utils.file_utils import save_json, load_json, save_markdown, load_markdown
import lmstudio as lms
from phases.context_analysis.paper_conception import PaperConcept
from phases.hypothesis_generation.hypothesis_models import Hypothesis
from phases.experimentation.experiment_state import (
    HypothesisEvaluation, ExecutionResult, CodeGenerationResult,
    ExperimentFiles, ValidationResult, VerdictResult, ExperimentResult, Plot
)
from phases.experimentation.code_executor import CodeExecutor
from phases.experimentation.results_manager import ResultsManager
from utils.llm_utils import remove_thinking_blocks
import lmstudio as lms


EXPERIMENTAL_PLAN_FILE = "experimental_plan.md"


class ExperimentRunner:
    """Conducts experiments to test hypotheses."""
    
    def __init__(self, base_output_dir: str = "output/experiments"):
        from settings import Settings
        self.settings = Settings
        self.executor = CodeExecutor()
        self.results_manager = ResultsManager(base_output_dir)
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
    
    def _write_experiment_code(
        self,
        experimental_plan: str,
        hypothesis: Hypothesis,
        paper_concept: PaperConcept,
        output_dir: str
    ) -> CodeGenerationResult:
        """Generate experiment code in chunks, save to file, execute, and return results."""
        
        try:
            code_snippets_section = ""
            if paper_concept.code_snippets:
                code_snippets_section = f"\n\nAvailable Code Snippets (use only if helpful):\n{paper_concept.code_snippets}"
            
            # System prompt for all chunks
            system_prompt = textwrap.dedent(f"""\
                [ROLE]
                You are an expert at writing scientific experiment code in Python.

                [TASK]
                Generate experiment code in logical chunks to test a given hypothesis.

                [REQUIREMENTS]
                - Write clean, concise Python code
                - Save plots to plots/ directory (create with os.makedirs if needed)
                - Save results to JSON in current directory
                - Print concise, meaningful output (~100-200 lines max)
                - Output ONLY Python code, NO markdown formatting

                [CONSTRAINTS]
                Code MUST complete in under 5 minutes. Reduce iterations, computations, or parameter combinations if needed. Optimize loops and maintain scientific validity.

                [AVAILABLE PACKAGES]
                The following Python packages are available (optional - use if helpful):
                - numpy: Numerical computing, arrays, mathematical operations
                - matplotlib: Plotting and visualization
                - seaborn: Statistical data visualization (built on matplotlib)
                - pygame: Game development and interactive simulations
                These packages are available but not required - use them only if they help test the hypothesis.

                [HYPOTHESIS]
                Description: {hypothesis.description}
                Rationale: {hypothesis.rationale}
                Expected Improvement: {hypothesis.expected_improvement}
                Baseline: {hypothesis.baseline_to_beat}

                [CODE_SNIPPETS]
                {code_snippets_section if code_snippets_section else "No code snippets provided"}

                [EXPERIMENTAL_PLAN]
                {experimental_plan}
            """)

            chat = lms.Chat(system_prompt)
            current_code = ""
            
            # Chunk 1/4: Imports and structures
            try:
                print(f"Generating imports and data structures...")
                chat.add_user_message("""Generate ONLY imports and data structure definitions.

                Include:
                - All necessary imports
                - Any classes or data structures needed
                - Global constants

                Do NOT include algorithm implementations, experiment logic, or visualization yet.""")

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.4})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating imports chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Chunk 2/4: Algorithms
            try:
                print(f"Generating algorithm implementations...")
                chat.add_user_message("""Implement the code for the algorithm(s) being tested and merge it with the previous response.

                Include everything from the previous response, then add:
                - The proposed method/algorithm being tested (as described in the experimental plan)
                - The baseline/comparison method (as described in the experimental plan)
                - Any helper functions needed for the algorithms
                The most important part is to implement the algorithms as described in the experimental plan.

                Output the COMPLETE code so far (imports and data structures + algorithms).""")

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.4})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating algorithms chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Chunk 3/4: Experiment setup and execution
            try:
                print(f"Generating experiment execution logic...")
                chat.add_user_message(textwrap.dedent("""\
                    Implement the code for the experiment setup and execution and merge it with the previous response.

                    Include everything from the previous response, then add:
                    - Experiment setup and execution (as described in the experimental plan)
                    - Running the proposed method and baseline/comparison method
                    - Metric collection and measurement
                    - Save results to JSON file in current directory
                    - Concise stdout output with key metrics

                    Output the COMPLETE code so far (imports and data structures + algorithms + experiment).
                    Do NOT include visualization yet.
                """))

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.4})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating experiment chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Chunk 4/4: Visualization/Plotting
            try:
                print(f"Generating visualization code...")
                chat.add_user_message(textwrap.dedent("""\
                    Generate the COMPLETE final code including everything from before PLUS visualization and summary.

                    Include everything from the previous response, then add:
                    - Create plots/ directory
                    - Generate comparison plots
                    - Save plots to plots/
                    - Print concise summary of the results (NEVER guess the results, only print the actual results)

                    Output the COMPLETE, FINAL code (imports & data structures + algorithms + experiment + visualization).
                """))

                model = lms.llm(self.settings.EXPERIMENT_CODE_WRITE_MODEL)
                response = model.respond(chat, config={"temperature": 0.4})
                current_code = self._remove_markdown_formatting(remove_thinking_blocks(response.content))
            except Exception as e:
                error_msg = f"ERROR generating visualization chunk: {e}"
                print(error_msg)
                traceback.print_exc()
                raise
            
            # Save the complete code (from final chunk)
            code_file_path = os.path.join(output_dir, f"experiment_{hypothesis.id}.py")
            code_file_path = os.path.abspath(code_file_path)
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(current_code)
            print(f"Code saved to {code_file_path}")
            
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
        chat: Optional[lms.Chat] = None,
        fix_attempt: int = 1,
        max_attempts: int = 5
    ) -> Tuple[ExecutionResult, lms.Chat]:
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
            
            # Truncate long outputs to avoid context truncation
            stdout_preview = stdout[:1000] if len(stdout) > 2000 else stdout
            stderr_preview = stderr[:1000] if len(stderr) > 2000 else stderr
            
            user_message = textwrap.dedent(f"""\
                [TASK]
                Fix the errors in this Python code.

                {attempt_context}

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
                
                [OUTPUT_REQUIREMENT]
                IMPORTANT: Output the COMPLETE fixed Python code file from start to finish. Do not truncate or omit any parts.
            """)

            print(f"Fixing experiment code (attempt {fix_attempt}/{max_attempts}): {code_file_path}")
            chat.add_user_message(user_message)
            model = lms.llm(self.settings.EXPERIMENT_CODE_FIX_MODEL)
            result = model.respond(chat, config={"temperature": 0.4})
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
        experimental_plan: str,
        hypothesis: Hypothesis,
        code_file_path: str
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
            You are an expert at validating scientific experiments.

            [TASK]
            Evaluate whether an experiment's results are valid and meaningful based on the experimental plan and hypothesis.

            [VALIDATION_CRITERIA]
            1. Did the experiment actually test the hypothesis?
            2. Were the expected outputs generated (plots, JSON results)?
            3. Are the results meaningful and complete?
            4. Are the algorithms and experiment logic correct and complete?
            5. Are there any obvious issues that make the experiment invalid?
        """)
                
        # User message with execution results and code
        validation_prompt = textwrap.dedent(f"""\
            [TASK]
            Evaluate whether this experiment produced valid and meaningful results.

            [HYPOTHESIS]
            Description: {hypothesis.description}
            Rationale: {hypothesis.rationale}
            Expected Improvement: {hypothesis.expected_improvement}
            Baseline to Beat: {hypothesis.baseline_to_beat or "N/A"}

            [EXPERIMENTAL_PLAN]
            {experimental_plan}

            [EXECUTION_RESULTS]
            - Return Code: {execution_result.return_code}
            - Generated Plots: {plot_count} file(s) {f"({', '.join([os.path.basename(p) for p in execution_result.plot_files])})" if execution_result.plot_files else ""}
            - Generated JSON results: {result_file_count} file(s)

            [CODE_EXECUTED]
            ```python
            {code_content}
            ```

            [STDOUT_OUTPUT]
            {stdout_summary}

            [OUTPUT_REQUIREMENTS]
            Provide a structured evaluation with:
            - reasoning: Detailed explanation of why results are valid or invalid
            - If invalid: concisely describe specific issues that need to be addressed in a few sentences.
        """)
        
        try:
            chat = lms.Chat(system_prompt)
            chat.add_user_message(validation_prompt)
            model = lms.llm(self.settings.EXPERIMENT_VALIDATION_MODEL)
            result = model.respond(chat, response_format=ValidationResult)
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
        output_dir: str
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
                2. Ensure the code actually tests the hypothesis as described in the experimental plan
                3. Ensure plots are saved to "plots/" directory (relative to execution directory) - create this directory if needed using os.makedirs("plots", exist_ok=True)
                4. Save detailed results/metrics to JSON file in the current directory (do NOT create an "output" directory - the code already runs from the output directory)
                5. Ensure stdout output is concise and meaningful - key metrics, conclusions and results only, avoid loop spam
                6. Make sure the experiment is complete and meaningful (e.g., not too short, collects proper metrics, etc.)
                7. Preserve any working, valid parts of the code

                [AVAILABLE_PACKAGES]
                The following Python packages are available (optional - use if helpful):
                - numpy: Numerical computing, arrays, mathematical operations
                - matplotlib: Plotting and visualization
                - seaborn: Statistical data visualization (built on matplotlib)
                - pygame: Game development and interactive simulations
                These packages are available but not required - use them only if they help test the hypothesis.

                [HYPOTHESIS]
                Description: {hypothesis.description}
                Rationale: {hypothesis.rationale}
                Expected Improvement: {hypothesis.expected_improvement}
                Baseline to Beat: {hypothesis.baseline_to_beat or "N/A"}

                [CURRENT_CODE]
                ```python
                {current_code}
                ```

                [VALIDATION_FEEDBACK]
                {feedback_text}

                [OUTPUT_FORMAT]
                Output ONLY the improved Python code, NO further markdown or explanations. Your answer will be saved to a code file.
            """)

            model = lms.llm(self.settings.EXPERIMENT_CODE_IMPROVE_MODEL)
            result = model.respond(prompt, config={"temperature": 0.4})
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
        plot_files: List[str],
        hypothesis: Hypothesis,
        experimental_plan: str,
        stdout: str
    ) -> List[Plot]:
        """Generate captions for plot files using LM Studio VLM API."""
        
        if not plot_files:
            return []
        
        plots = []
        
        # System prompt for plot caption generation
        system_prompt = """[ROLE]
        You are an expert at writing scientific figure captions for research papers.

        [TASK]
        Generate concise, informative captions for scientific plots/figures generated from experiments.

        [GUIDELINES]
        1. Describe what the plot shows clearly and concisely
        2. Relate it to the hypothesis being tested
        3. Use professional scientific language suitable for research papers
        4. Keep captions typically 1-2 sentences
        5. Be informative but concise
        6. Focus on the key findings or comparisons shown in the plot"""
        
        for plot_file in plot_files:
            # Extract just the filename from the full path
            filename = os.path.basename(plot_file)
            
            # Prepare the image for VLM
            try:
                image_handle = lms.prepare_image(plot_file)
            except Exception as e:
                print(f"ERROR: Failed to prepare image {filename}: {e}")
                traceback.print_exc()
                # Fallback caption without image
                plots.append(Plot(filename=plot_file, caption=f"Figure showing results for hypothesis: {hypothesis.description}"))
                continue
            
            user_message = textwrap.dedent(f"""\
                [TASK]
                Generate a caption for this scientific plot.

                [HYPOTHESIS]
                Description: {hypothesis.description}
                Rationale: {hypothesis.rationale}
                Expected Improvement: {hypothesis.expected_improvement}
                Baseline to Beat: {hypothesis.baseline_to_beat or "N/A"}

                [EXPERIMENTAL_PLAN]
                {experimental_plan}

                [CODE_EXECUTION_OUTPUT]
                {stdout[:1000] if len(stdout) > 1000 else stdout}

                [PLOT_FILENAME]
                {filename}

                [OUTPUT_REQUIREMENT]
                Generate a professional scientific figure caption for this plot. Output ONLY the caption text, no additional formatting or explanations.
            """)
                        
            try:
                chat = lms.Chat(system_prompt)
                chat.add_user_message(user_message, images=[image_handle])
                model = lms.llm(self.settings.EXPERIMENT_PLOT_CAPTION_MODEL)
                result = model.respond(chat, config={"temperature": 0.4})
                caption = remove_thinking_blocks(result.content)
                plots.append(Plot(filename=plot_file, caption=caption))
            except Exception as e:
                print(f"ERROR: Failed to generate caption for {filename}: {e}")
                traceback.print_exc()
                # Fallback caption
                plots.append(Plot(filename=plot_file, caption=f"Figure showing results for hypothesis: {hypothesis.description}"))
        
        return plots
    
    def _generate_experimental_plan(
        self,
        hypothesis: Hypothesis,
        paper_concept: PaperConcept
    ) -> str:
        """Generate a detailed experimental plan for testing a hypothesis."""

        prompt = textwrap.dedent(f"""\
            [TASK]
            Create a detailed, concise experimental plan for testing a given hypothesis.
            The plan will be used to generate the experiment code in Python.

            [PLAN_REQUIREMENTS]
            Include:
            - Objective and success criteria
            - Required mathematical formulas/technical details
            - Experimental setup
            - Metrics to measure
            - Implementation approach
            - Output requirements: 
              * Detailed results/metrics stored in JSON file
              * Concise, meaningful output to stdout (key metrics, conclusions)
              * Plot(s) for visualization

            [CONSTRAINTS]
            Experiment MUST complete in under 5 minutes. Use reasonable parameter ranges and reduce iterations/computations/parameter combinations if needed.

            [AVAILABLE_PACKAGES]
            The following Python packages are available (optional - use if helpful):
            - numpy: Numerical computing, arrays, mathematical operations
            - matplotlib: Plotting and visualization
            - seaborn: Statistical data visualization (built on matplotlib)
            - pygame: Game development and interactive simulations
            These packages are available but not required - use them only if they help test the hypothesis.

            [RESEARCH_CONTEXT]
            {paper_concept.description}

            [CODE_SNIPPETS]
            {paper_concept.code_snippets if paper_concept.code_snippets else "No code snippets provided"}

            [HYPOTHESIS]
            Description: {hypothesis.description}
            Rationale: {hypothesis.rationale}
            Expected Improvement: {hypothesis.expected_improvement}
            Baseline to Beat: {hypothesis.baseline_to_beat or "N/A"}

            [INSTRUCTIONS]
            Be specific and actionable. Only use information actually present in the research context.""")

        try:
            model = lms.llm(self.settings.EXPERIMENT_PLAN_MODEL)
            result = model.respond(prompt, config={"temperature": 0.4})
            return remove_thinking_blocks(result.content)
        except Exception as e:
            print(f"ERROR: Failed to generate experimental plan: {e}")
            traceback.print_exc()
            raise
    
    def save_experimental_plan(
        self,
        experimental_plan: str
    ) -> str:
        """Save an experimental plan to a file."""

        file_path = save_markdown(experimental_plan, EXPERIMENTAL_PLAN_FILE, self.base_output_dir)
        
        return file_path
    
    def load_experimental_plan(
        self
    ) -> str:
        """Load an experimental plan from a file."""

        file_path = os.path.join(self.base_output_dir, EXPERIMENTAL_PLAN_FILE)

        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Experimental plan not found: {file_path}")

        plan_content = load_markdown(path_obj.name, str(path_obj.parent))

        return plan_content
    
    def load_experiment_code(
        self,
        hypothesis_id: str
    ) -> str:
        """Load experiment code from a file."""
        
        filename = f"experiment_{hypothesis_id}.py"
        file_path = os.path.join(self.base_output_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment code not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        return code_content
    
    def load_experiment_files(
        self,
        hypothesis_id: str
    ) -> ExperimentFiles:
        """Load both experimental plan and experiment code files."""
        
        return ExperimentFiles(
            experimental_plan=self.load_experimental_plan(),
            experiment_code=self.load_experiment_code(hypothesis_id),
            plan_file_path=os.path.join(self.base_output_dir, EXPERIMENTAL_PLAN_FILE),
            code_file_path=os.path.join(self.base_output_dir, f"experiment_{hypothesis_id}.py")
        )
    
    def save_experiment_result(
        self,
        experiment_result: ExperimentResult
    ) -> str:
        """Save ExperimentResult to JSON file."""

        # Convert ExperimentResult to dictionary
        result_dict = self._experiment_result_to_dict(experiment_result)

        filename = f"experiment_result_{experiment_result.hypothesis.id}.json"
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
            hypothesis_id = hypothesis.id
            code_file_path = os.path.join(base_dir, f"experiment_{hypothesis_id}.py")
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
            experimental_plan=data['experimental_plan'],
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
    
    def run_experiment(
        self,
        hypothesis: Hypothesis,
        paper_concept: PaperConcept,
        load_existing_plan: bool = False,
        load_existing_code: bool = False
    ) -> ExperimentResult:
        """Run experiment to test hypothesis."""
        
        # Ensure output directory exists
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Generate or load experimental plan
        try:
            plan_file_path = os.path.join(self.base_output_dir, EXPERIMENTAL_PLAN_FILE)
            if load_existing_plan and os.path.exists(plan_file_path):
                print(f"Loading existing experimental plan...")
                experimental_plan = self.load_experimental_plan()
            else:
                if load_existing_plan:
                    print(f"Experimental plan not found, generating new plan...")
                else:
                    print(f"Generating new experimental plan...")
                experimental_plan = self._generate_experimental_plan(hypothesis, paper_concept)
                self.save_experimental_plan(experimental_plan)
        except Exception as e:
            print(f"ERROR: Failed to generate/load experimental plan: {e}")
            traceback.print_exc()
            raise
        
        # Generate or load experiment code
        code_file_path = os.path.join(self.base_output_dir, f"experiment_{hypothesis.id}.py")
        code_file_path = os.path.abspath(code_file_path)
        
        if load_existing_code and os.path.exists(code_file_path):
            print(f"Loading existing experiment code for {hypothesis.id}...")
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
                    existing_plots = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.svg', '.pdf'))] if os.path.exists(plots_dir) else []
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
            write_result = self._write_experiment_code(
                experimental_plan,
                hypothesis,
                paper_concept,
                self.base_output_dir
            )
        
        code_file_path = write_result.code_file_path
        execution_result = write_result.execution_result
        
        # Track total attempts across all loops
        total_fix_attempts = 0
        total_validation_attempts = 0
        validation_result = None
        
        # Fix code if execution failed
        max_fix_attempts = 5
        fix_attempt = 0
        fix_chat = None  # Chat object to keep conversation across fix attempts
        
        while execution_result.return_code != 0 and fix_attempt < max_fix_attempts:
            fix_attempt += 1
            total_fix_attempts += 1
            
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
                chat=fix_chat,
                fix_attempt=fix_attempt,
                max_attempts=max_fix_attempts
            )
            
            execution_result = fix_result
        
        # Initialize verdict and reasoning (will be set in branches below)
        verdict = "inconclusive"
        reasoning = ""
        plot_captions = []  # Will be populated if plots exist and execution succeeds
        
        # Validate and improve results if execution succeeded
        if execution_result.return_code == 0:
            # Validate results and improve if needed
            max_validation_attempts = 3
            validation_attempt = 0
            validation_passed = False
            
            print("Validating experiment results...")
            while not validation_passed and validation_attempt < max_validation_attempts:
                validation_attempt += 1
                total_validation_attempts += 1
                validation_result = self._validate_experiment_results(
                    execution_result,
                    experimental_plan,
                    hypothesis,
                    code_file_path
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
                        improvement_result = self._improve_experiment_code(
                            code_file_path,
                            validation_result,
                            hypothesis,
                            self.base_output_dir
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
                                    chat=nested_fix_chat,
                                    fix_attempt=fix_attempt,
                                    max_attempts=max_fix_attempts
                                )
                                
                                execution_result = fix_result
                            fix_chat = nested_fix_chat  # Update main fix_chat
                            
                            # If still broken after fix attempts, break validation loop
                            if execution_result.return_code != 0:
                                break
            
            # Determine verdict and reasoning
            if execution_result.return_code == 0 and validation_passed:
                # Generate plot captions if plots exist
                if execution_result.plot_files:
                    print("Generating captions for plots...")
                    plot_captions = self._generate_plot_captions(
                        execution_result.plot_files,
                        hypothesis,
                        experimental_plan,
                        execution_result.stdout
                    )
                    print(f"Generated {len(plot_captions)} plot caption(s)")
                
                # Successful execution with valid results - get verdict from LLM
                print("Code executed successfully with valid results. Determining verdict...")
                
                # Truncate stdout if too long to prevent context overflow
                stdout_summary = execution_result.stdout
                if len(stdout_summary) > 1500:
                    stdout_summary = stdout_summary[:1500] + "\n...[truncated]..."
                
                # Build plot captions text for prompt
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
                    Rationale: {hypothesis.rationale}
                    Expected Improvement: {hypothesis.expected_improvement}
                    Baseline to Beat: {hypothesis.baseline_to_beat or "N/A"}

                    [STDOUT_OUTPUT]
                    {stdout_summary}

                    [PLOT_CAPTIONS]
                    {plot_captions_text}

                    [TASK]
                    Based on the execution results and generated plots, provide:
                    1. A concise reasoning about whether the hypothesis is proven, disproven, or inconclusive
                    2. Your concise analysis of the results and observations of the experiment
                    Then determine the verdict with a single word: 'proven', 'disproven', or 'inconclusive'.
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
                        reasoning += f"\n\nNote: Invalid verdict '{verdict_result.verdict}' was returned, defaulting to 'inconclusive'."
                except Exception as e:
                    print(f"ERROR: Failed to get verdict: {e}")
                    traceback.print_exc()
                    verdict = "inconclusive"
                    reasoning = f"Failed to determine verdict: {str(e)}"
            elif execution_result.return_code == 0 and not validation_passed:
                # Execution succeeded but validation failed
                print(f"Results validation failed after {max_validation_attempts} attempts.")
                reasoning = f"Code executed successfully but results validation failed after {max_validation_attempts} attempts. Last validation reasoning: {validation_result.reasoning}"
                if validation_result.issues:
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
            
            self.results_manager.save_hypothesis_evaluation(evaluation)
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
            experimental_plan=experimental_plan,
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

