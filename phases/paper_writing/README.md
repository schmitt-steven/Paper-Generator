# Section Writing Phase

System for generating academic paper sections with indirect citations. Inspired by [PaperQA](https://github.com/Future-House/paper-qa).

## Components

### `Data Models` ([data_models.py](data_models.py))

- `Section` - Enum of paper sections (Abstract, Introduction, etc.)
- `PaperDraft` - Container for all generated sections
- `PaperChunk` - Indexed chunk with embedding for retrieval
- `Evidence` - Scored chunk with summary and source query
- `SectionCritique` - Structured critic output with improvements and search queries

### `PaperIndexer` ([paper_indexer.py](paper_indexer.py))

Splits papers into overlapping text chunks and creates their embeddings.

- First preprocesses markdown, strips references/bibliography
- Then splits the text into overlapping chunks with configurable token size and overlap
- Lastly embeds the chunks in batches, caching them to `output/paper_embeddings.json`

### `EvidenceGatherer` ([evidence_gatherer.py](evidence_gatherer.py))

Handles evidence retrieval for the paper writing pipeline.
- `batch_search`: Executes a list queries from the critic (sequentially but with internal batching).
- Internal `_search_evidence` pipeline: vector search → combined summarization & scoring → filter.

### `SectionCritic` ([section_critic.py](section_critic.py))

Analyzes draft sections and returns structured feedback:
- `improvements`: List of specific suggestions (positively framed)
- `search_queries`: Targeted queries to fill missing evidence gaps

### `SectionGuidelinesLoader` ([style_guidelines.py](style_guidelines.py))

Loads user-defined style guidelines from `user_files/style_guidelines.md`.

### `PaperWriter` ([paper_writer.py](paper_writer.py))

Generates and rewrites paper sections.
- `generate_initial_section`: Creates initial draft using a catalog of titles/abstracts/conclusions.
- `rewrite_section`: Refines the draft using critique feedback and gathered evidence.
- `generate_title`: Generates a paper title from the full draft.
- `generate_acknowledgements`: Formats user-provided acknowledgements.
- `generate_paper_sections`: Orchestrates the writing of all sections in order.

### `PaperWritingPipeline` ([paper_writing_pipeline.py](paper_writing_pipeline.py))

High-level orchestrator for the entire paper writing workflow:
1. Index papers
2. Loop through sections (Methods → Results → Discussion → etc.)
3. For each section:
   - Draft (zero-shot from catalog)
   - Critique (identify gaps)
   - Search (fill gaps with evidence)
   - Rewrite (incorporate feedback)

## Output

- `output/paper_embeddings.json` - Cached chunk embeddings
- `output/section_writing_prompts.json` - Prompts used for each section
- `output/paper_draft.md` - Generated paper draft
