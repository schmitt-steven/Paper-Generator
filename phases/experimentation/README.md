# Experimentation Phase

Runs experiment to test a research hypothesis.

## Components

- **`ExperimentRunner`**: Orchestrates hypothesis testing workflow
- **`CodeExecutor`**: Executes generated experiment code
- **`ResultsManager`**: Manages experiment outputs and plots

## Process

1. Generate experiment plan
2. Generate and validate code
3. Execute code
4. If broken: fix/regenerate code
5. If results are unusable: improve code
6. Generate plots with captions
7. Evaluate results and generate verdict

## Output

- `output/experiments/` - Experiment code, results, and plots

