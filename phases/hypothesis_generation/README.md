# Hypothesis Generation Phase

Generates a structured research hypothesis from user input and paper concept.

## Components

### `Hypothesis` ([hypothesis_builder.py](hypothesis_builder.py))

Data class representing a testable research hypothesis.

- `description` - What the hypothesis claims
- `rationale` - Why this hypothesis is worth testing
- `success_criteria` - How to determine if test results prove the hypothesis

### `HypothesisBuilder` ([hypothesis_builder.py](hypothesis_builder.py))

Uses LLM to convert user-provided hypothesis text into a `Hypothesis` object.

- Takes raw user input from the paper specification and paper concept as context
- Outputs a structured hypothesis with description, rationale and success criteria

## Output

`output/hypothesis.md` containing:
- Hypothesis description
- Rationale
- Success criteria
