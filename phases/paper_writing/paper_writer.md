# Process Description: RAG-Based Citation Retrieval for Academic Paper Section Generation

## Overview

Adapted from PaperQA2's approach for generating academic paper sections with indirect citations. Uses an agentic 3-phase process: index papers, gather evidence (with LLM-driven search), generate sections. The evidence gathering phase employs an LLM agent that can iteratively search for evidence using custom queries, similar to PaperQA's approach.

---

## Phase 1: Paper Indexing (One-time Setup)

### Input
- `papers: List[Paper]` - Paper objects with `markdown_text` field populated
- `embedding_model` - LM Studio embedding model

### Process
1. For each `Paper`:
   - Parse `paper.markdown_text` to identify sections (Abstract, Introduction, Methods, Results, Discussion)
   - Split each section into chunks (~300-500 tokens, ~100 token overlap)
   - Preserve section boundaries and metadata
   - Create embeddings for each chunk
   - Store: `(chunk_text, embedding, paper_reference, section_type, chunk_index)`

### Output
- Indexed corpus: List of `PaperChunk` objects with embeddings

### Data Structure
```python
PaperChunk {
chunk_id: str
paper: Paper # Reference to full Paper object
section_type: str
chunk_text: str
chunk_index: int
embedding: List[float]
}
```

**Note:** All metadata (title, authors, year, bibtex, doi, etc.) is accessed via `chunk.paper.*` - no duplication needed.

**Section Type:** Stored as metadata indicating where the chunk came from in the source paper. Not used for filtering - semantic relevance is prioritized.

---

## Phase 2: Agentic Evidence Gathering (Per Section Generation)

### Overview
An LLM agent iteratively gathers evidence by:
1. Starting with default queries (provided as "starting information")
2. Analyzing what evidence is present and what's missing
3. Creating custom search queries to fill gaps
4. Repeating until sufficient evidence is gathered or max iterations reached

The agent has access to a `search_evidence` tool that performs vector search + LLM scoring over the indexed corpus.

### Input
- `section_type: Section` - Section being generated
- `default_queries: List[str]` - Pre-constructed queries (see Section-Specific Query Construction)
- `context: PaperConcept` - Research context
- `experiment: ExperimentResult` - Experiment data (if applicable)
- `indexed_corpus: List[PaperChunk]` - Indexed paper chunks
- `llm_model` - For agent reasoning, summarization, and scoring
- `embedding_model` - For vector search
- `max_search_iterations: int` - Maximum number of custom searches (default: 5)

### Process

#### Step 2.1: Initial Evidence from Default Queries
- Execute Phase 2 search process (Steps 2.1-2.3 below) for each default query
- Aggregate all evidence from default queries
- Deduplicate chunks from same paper
- Output: `initial_evidence: List[Evidence]`

#### Step 2.2: Agent Setup
- Build agent prompt with:
  - Section type and requirements
  - Research context (paper concept, hypothesis, experiment)
  - Initial evidence from default queries (formatted with summaries)
  - Instructions on what evidence is needed for this section type
  - Tool documentation for `search_evidence`
- Provide agent with `search_evidence` tool

#### Step 2.3: Agentic Iterative Search
- Agent analyzes initial evidence and identifies gaps
- Agent can invoke `search_evidence(query: str)` tool with custom queries
- Each tool call returns evidence using the search process
- Agent tracks what's been retrieved to avoid redundancy
- Continues until:
  - Agent determines sufficient evidence gathered, OR
  - Maximum iterations reached (default: 5 custom searches)
- Output: `final_evidence: List[Evidence]` (aggregated from all searches)

### Search Evidence Tool

The `search_evidence(query: str)` tool performs the following sub-process:

#### Sub-Step 2.1: Vector Search (Initial Retrieval)
- Embed the query using embedding model
- Compute cosine similarity against all chunks in indexed corpus
- Select top `top_k_initial` chunks by similarity score (default: 20)
- Output: List of `(chunk, vector_score)` tuples

#### Sub-Step 2.2: Contextual Summarization
- For each candidate chunk:
  - Prompt LLM to summarize the chunk in the context of the query
  - Focus: Why this chunk is relevant to the query
  - Output: 2-3 sentence summary explaining relevance
- Output: List of `(chunk, vector_score, summary)` tuples

#### Sub-Step 2.3: LLM Re-scoring
- For each summarized chunk:
  - Prompt LLM to score relevance (0.0-1.0 scale)
  - Input: query + summary + paper metadata (`chunk.paper.title`, `chunk.paper.authors`)
  - Output: LLM relevance score
- Output: List of `(chunk, vector_score, summary, llm_score)` tuples

#### Sub-Step 2.4: Combined Scoring & Selection
- Compute combined score: `0.4 * vector_score + 0.6 * llm_score`
- Sort by combined score (descending)
- Select top `top_k_final` chunks (default: 5-8)
- Output: List of `Evidence` objects

### Output
- `evidence: List[Evidence]` - Aggregated evidence from default queries + agent searches
- Each `Evidence` contains:
  - `chunk: PaperChunk` - Original paper chunk (with `paper` reference)
  - `summary: str` - Contextual summary
  - `vector_score: float` - Initial similarity score
  - `llm_score: float` - LLM relevance score
  - `combined_score: float` - Weighted final score
  - `source_query: str` - Which query retrieved this evidence (for tracking)

### Agent Behavior Guidelines
- **Query Diversity:** Agent should create queries that differ meaningfully from defaults
- **Gap Identification:** Agent should identify specific missing aspects (methods, results, comparisons, etc.)
- **Iteration Control:** Agent should stop when evidence is sufficient, not always use max iterations
- **Deduplication:** System tracks retrieved chunks to avoid redundant results across searches

---

## Phase 3: Generate Section with Citations

### Input
- `section_type: Section`(an enum) - Section to generate (material and methods/methodology, "results", "discussion", "introduction", "conclusion", "abstract")
- `context: PaperConcept` - Research context
- `experiment: ExperimentResult` - Experiment data
- `evidence: List[Evidence]` - From Phase 2
- `llm_model` - For generation

### Process

#### Step 3.1: Build Generation Prompt
- Section-specific instructions (varies by section type)
- Research context (paper concept, hypothesis)
- Formatted evidence (for indirect citation)
- Style guidelines (academic writing, citation format)
- Output: Complete prompt string

#### Step 3.2: Generate Section
- Send prompt to LLM
- LLM generates section integrating evidence indirectly
- Citations appear naturally in narrative: "(smith2024quantum)"
- Output: Generated section text

### Output
- `section_text: str` - Complete section with integrated citations

---

## Section-Specific Query Construction (Default Queries)

These queries serve as "starting information" for the agent. The agent can use them as-is or create additional custom queries to fill gaps.

### Introduction
- **Default Query:** `{paper_concept.description} + {paper_concept.open_questions} + {hypothesis.description}`
- **Focus:** Background, problem statement, related work
- **Agent Can Search For:** Specific related work gaps, alternative problem formulations, historical context

### Methods
- **Default Query:** `{experimental_plan} + {hypothesis.method_combination} + {experiment_code}`
- **Focus:** Similar methodologies, technical approaches
- **Agent Can Search For:** Specific implementation details, alternative methods, parameter choices, evaluation protocols

### Results
- **Default Query:** `{experiment.execution_result.stdout} + {experiment.metrics} + {experiment.plots}`
- **Focus:** Comparable results, benchmarks, experimental outcomes
- **Agent Can Search For:** Specific benchmarks, baseline comparisons, similar experimental setups, performance metrics

### Discussion
- **Default Query:** `{experiment.verdict} + {experiment.reasoning} + {experiment.results}`
- **Focus:** Related findings, comparisons, limitations
- **Agent Can Search For:** Conflicting results, alternative interpretations, limitations of similar work, future directions

### Abstract
- **Default Query:** `{paper_concept.description} + {experiment.summary}`
- **Focus:** High-level overview, key contributions
- **Agent Can Search For:** Positioning within field, contribution clarity, related high-level work

### Conclusion
- **Default Query:** `{experiment.verdict} + {experiment.reasoning} + {paper_concept.description}`
- **Focus:** Summary, implications, future work
- **Agent Can Search For:** Future work directions, broader implications, connections to other fields

---

## Key Design Decisions

### Agentic Evidence Gathering
- LLM agent decides what evidence is needed and creates custom search queries
- Default queries provide strong starting point and anchor the search space
- Agent can invoke search tool multiple times in any order (similar to PaperQA)
- Iterative refinement allows discovering evidence that default queries miss

### Semantic Relevance Over Section Matching
- Vector similarity + LLM scoring determines relevance
- Related content can appear in any section of source papers
- Query itself captures semantic intent

### Two-Stage Retrieval (Per Search)
- Initial vector search finds candidates
- LLM re-scoring refines selection
- Combined scoring balances both approaches

### Contextual Summarization
- Chunks summarized in context of query
- Helps LLM understand relevance
- Enables better integration into narrative

### Indirect Citations
- Summaries integrated into narrative
- Citations appear naturally: "[Author et al., Year]"
- No direct quotes - academic writing style

---

## Data Flow Summary

```
[Papers with markdown]
→ [Chunking]
→ [Embedding]
→ [Indexed Corpus]

[Section Context + Default Queries]
→ [Execute Default Queries → Initial Evidence]
→ [Agent Setup with Initial Evidence + search_evidence Tool]
→ [Agent Iteratively Calls search_evidence(query)]
  → [Vector Search - All Chunks]
  → [Top K Candidates]
  → [Contextual Summarization]
  → [LLM Re-scoring]
  → [Top Evidence by Combined Score]
→ [Aggregate All Evidence]
→ [Final Evidence Set]

[Evidence + Section Context]
→ [Build Prompt]
→ [LLM Generation]
→ [Section with Citations]
```


---

## Quality Control Points

1. **Diversity:** Avoid multiple chunks from the same paper in one section (deduplication)
2. **Citation Balance:** Ensure mix of high-scoring and diverse papers
3. **Summary Quality:** Ensure summaries are concise and contextually relevant
4. **Citation Formatting:** Consistent citation keys and formatting across sections
5. **Query Quality:** Agent queries should be semantically distinct from defaults
6. **Iteration Efficiency:** Agent should stop when sufficient evidence is gathered
7. **Search Tracking:** Track which queries retrieved which evidence to understand agent reasoning

---

## Implementation Notes

- **Chunking:** Use markdown section headers to preserve structure
- **Math/Code:** Keep equations and code blocks intact as single units
- **Performance:** Batch embedding creation during indexing
- **Caching:** Cache search results if query is similar (optional optimization)
- **Citation Management:** Track which papers are cited in each section
- **Tool Implementation:** Use LM Studio's `model.act()` with `search_evidence` as a tool function
- **Iteration Limits:** Enforce max_search_iterations to control cost and latency
- **Evidence Deduplication:** Track retrieved chunk IDs across all searches to avoid duplicates
- **Agent Prompting:** Provide clear instructions on when to search vs. when evidence is sufficient