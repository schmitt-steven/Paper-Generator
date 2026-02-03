# Experimentation Phase

Runs experiments to test research hypotheses.

## Components

### Data Classes ([experiment_state.py](experiment_state.py))

- `ExecutionResult` - Stdout, stderr, return code, and generated files
- `ValidationResult` - Whether experiment results are valid/meaningful
- `HypothesisEvaluation` - Verdict (proven/disproven/inconclusive) with reasoning
- `Plot` - Plot filename with generated caption
- `ExperimentResult` - Aggregation of the above objects

### `CodeExecutor` ([code_executor.py](code_executor.py))

Executes experiment code in a subprocess with timeout handling.

### `ExperimentRunner` ([experiment_runner.py](experiment_runner.py))

Handles the complete experimentation process:
- Generates experiment plan from hypothesis and research context
- Writes experiment code using LLM
- Executes code and handles errors with automatic fixing
- Validates results and improves code if needed
- Generates plot captions using VLM
- Evaluates hypothesis (proven/disproven/inconclusive)

## Output

`output/experiments/` containing:
- `experiment_plan.md` - Detailed experiment plan
- `experiment.py` - Generated experiment code
- `plots/` - Generated plots (PNG, SVG, etc.)
- `plot_captions.json` - VLM-generated captions for plots
- `hypothesis_evaluation.json` - Final verdict on hypothesis
- `experiment_result.json` - Detailed execution and validation results
