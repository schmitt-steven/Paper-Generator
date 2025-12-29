# Context Analysis Phase

Analyzes user code and requirements to generate a paper concept.

## Components

### `CodeAnalyzer` ([user_code_analysis.py](user_code_analysis.py))

Analyzes user code files for novel concepts and research relevance.

- Loads code files from `user_files/`
- Uses LLM to identify novel concepts and research relevance
- Extracts important code snippets with explanations

### `UserRequirements` ([user_requirements.py](user_requirements.py))

Parses structured user requirements from `user_files/user_requirements.md`.

- Topic and hypothesis
- Section-specific requirements (abstract, introduction, methods, etc.)

### `PaperConception` ([paper_conception.py](paper_conception.py))

Generates the paper concept using code analysis and user requirements.

- Builds paper description from analyzed code and requirements
- Identifies open questions to guide literature search
- Formats code snippets for inclusion in the paper

## Output

`output/paper_concept.md` containing:
- Paper description
- Open questions for literature search
- Code snippets with explanations
