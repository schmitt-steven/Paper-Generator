# Paper Writing Phase

RAG-based system for generating academic paper sections with indirect citations. Inspired by [PaperQA](https://github.com/Future-House/paper-qa).

## Components

### Data Models ([data_models.py](data_models.py))

- `Section` - Enum of paper sections (Abstract, Introduction, etc.)
- `PaperDraft` - Container for all generated sections
- `PaperChunk` - Indexed chunk with embedding for retrieval
- `Evidence` - Scored chunk with summary and source query

### `PaperIndexer` ([paper_indexer.py](paper_indexer.py))

Splits papers into overlapping text chunks and creates their embeddings.

- First preprocesses markdown, strips references/bibliography
- Then splits the text into overlapping chunks with configurable token size and overlap
- Lastly embeds the chunks in batches, caching them to `output/paper_embeddings.json`

### `EvidenceGatherer` ([evidence_gatherer.py](evidence_gatherer.py))

LLM agent that iteratively calls `search_evidence` tool until sufficient evidence is found (or max iterations reached).

Each `search_evidence` call runs a retrieval pipeline:
1. Vector search (cosine similarity) to find relevant chunks
2. Batch summarization (LLM summarizes retrieved chunks)
3. Batch scoring (LLM scores relevance 0.0-1.0)
4. Combined scoring: `0.3 * vector_score + 0.7 * llm_score`
5. Filtering: keep only chunks with highest combined scores

### `EvidenceManager` ([evidence_manager.py](evidence_manager.py))

Utilities for saving/loading evidence by section to JSON.

### `SectionGuidelinesLoader` ([section_guidelines.py](section_guidelines.py))

Loads user-defined writing guidelines from `user_files/section_guidelines.md`.

### `PaperWriter` ([paper_writer.py](paper_writer.py))

Generates paper sections in order: Methods → Results → Discussion → Introduction → Related Work → Conclusion → Abstract → Title

- Builds prompts with context, evidence, and guidelines
- Integrates figures/plots in Results section
- Generates title from abstract, introduction, and conclusion (unles user provided one)

### `PaperWritingPipeline` ([paper_writing_pipeline.py](paper_writing_pipeline.py))

Handles the complete paper writing workflow: indexing → evidence gathering → section writing.

## Output

- `output/paper_embeddings.json` - Cached chunk embeddings
- `output/evidence.json` - Gathered evidence by section
- `output/section_writing_prompts.md` - Prompts used for each section
- `output/paper_draft.md` - Generated paper draft
