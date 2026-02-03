# Context Analysis Phase

Analyzes user code and requirements to generate a research context.

## Components

### `CodeAnalyzer` ([user_code_analysis.py](user_code_analysis.py))

Analyzes user code files for novel concepts and research relevance.

- Loads code files from `user_files/`
- Uses LLM to identify novel concepts and research relevance
- Extracts important code snippets with explanations

### `PaperSpecification` ([paper_specification.py](paper_specification.py))

Parses structured paper specification from `user_files/paper_specification.md`.

- Topic and hypothesis
- Section-specific requirements (abstract, introduction, methods, etc.)

### `ResearchContextGenerator` ([research_context_generator.py](research_context_generator.py))

Generates the research context using code analysis and paper specification.

- Builds paper description from analyzed code and requirements
- Identifies open questions to guide literature search
- Formats code snippets for inclusion in the paper

## Output

`output/research_context.md` containing:
- Paper description
- Open questions for literature search
- Code snippets with explanations
