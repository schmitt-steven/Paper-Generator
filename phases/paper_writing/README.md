# Paper Writing System

An agentic, RAG-based system for generating academic paper sections with indirect citations. Uses a 3-phase approach:
1. Indexing a corpus of research papers into searchable chunks
2. Searching for relevant information automatically and via an LLM agent for each chapter
3. Generating complete chapters (with citations and figure integration)

This system is inspired by **PaperQA**, a RAG-based, agent-driven tool that is capable of answering scientific questions, summarizing papers and more. Check it out here: https://github.com/Future-House/paper-qa?tab=readme-ov-file

## Components

- **`PaperIndexer`**: Chunks and embeds papers into a searchable corpus
- **`QueryBuilder`**: Constructs section-specific default queries
- **`EvidenceGatherer`**: Performs vector search, summarization, and LLM scoring; orchestrates agentic iterative search
- **`PaperWriter`**: Generates paper sections from evidence
- **`PaperWritingPipeline`**: Orchestrates the entire process


## Phase 1: Paper Indexing

Converts papers into searchable chunks with embeddings for semantic retrieval.

### Process

1. **Preprocessing**: Clean markdown text (remove artifacts from PDF conversion)
2. **Whole-Document Chunking**: 
   - Split entire document into chunks (~300-500 tokens)
   - Preserve code blocks and equations as single units
   - Use ~100 token overlap between chunks
3. **Embedding**: Create embeddings for each chunk using an embedding model
4. **Indexing**: Store chunks with metadata

### Output
- `List[PaperChunk]` - Indexed corpus with embeddings
- Each chunk contains: `chunk_id`, `paper` reference, `chunk_text`, `chunk_index`, `embedding`

### Data Structure
```python
PaperChunk {
    chunk_id: str
    paper: Paper  # Reference to full Paper object
    chunk_text: str
    chunk_index: int
    embedding: List[float]
}
```


## Phase 2: Agentic Evidence Gathering (Per Section)

An LLM agent iteratively gathers evidence by:
1. Starting with default queries (provided as "starting information")
2. Analyzing what evidence is present and what's missing
3. Creating custom search queries to fill gaps
4. Repeating until sufficient evidence is gathered or max iterations reached

The agent can use the `search_evidence` tool that performs a vector search + LLM scoring.

### Input
- `section_type: Section` - Section being generated
- `default_queries: List[str]` - Pre-constructed queries
- `context: PaperConcept` - Research context
- `experiment: ExperimentResult` - Experiment data (if applicable)
- `indexed_corpus: List[PaperChunk]` - Indexed paper chunks
- `max_iterations: int` - Times the agent can search for information

### Process

#### Step 2.1: Execute Default Queries
- Run search process for each default query sequentially

#### Step 2.2: Agent Setup
- Build agent prompt with:
  - Section type and objectives
  - Research context (paper concept, hypothesis, experiment)
  - Initial evidence from default queries (formatted with summaries)
  - Instructions on what evidence is needed
  - Tool documentation for `search_evidence`
- Provide agent with `search_evidence` tool

#### Step 2.3: Agentic Iterative Search
- Agent analyzes initial evidence and identifies gaps
- Agent calls `search_evidence(query: str, target_section: Section)` with his own queries
- Each tool call uses `exclude_chunk_ids` to prevent retrieving chunks already seen
- After each search, `_deduplicate_evidence` merges new evidence with existing evidence
- Keeps highest-scoring instance if same chunk appears multiple times
- Continues until:
  - Agent determines enough evidence was gathered, or
  - Maximum iterations reached

### Search Evidence Tool Process

The `search_evidence(query: str, target_section: Section)` tool performs:

#### Sub-Step 2.1: Vector Search
- Embed query using embedding model
- Compute cosine similarity against all chunks
- Select top `top_k_initial` chunks
- Exclude already-seen chunks (per-section deduplication)

#### Sub-Step 2.2: Contextual Summarization
- For each candidate chunk:
  - Prompt LLM to summarize chunk in context of query
  - Focus: Why this chunk is relevant to the query
  - Output: Short summary of the chunk

#### Sub-Step 2.3: LLM Re-scoring
- For each summarized chunk:
  - Prompt LLM to score relevance (0.0-1.0 scale)
  - Input: query + summary + target section + chunk source section
  - Uses structured output (`ScoreResult`)
  - Output: LLM relevance score

#### Sub-Step 2.4: Combined Scoring & Selection
- Compute combined score: `0.4 * vector_score + 0.6 * llm_score`
- Sort by combined score (descending)
- Select top `top_k_final` chunks
- Return `Evidence` objects

### Output
- `evidence: List[Evidence]` - Aggregated evidence from default queries + agent searches
- `final_prompt: str` - Complete prompt with tool call results (for debugging)

### Evidence Data Structure
```python
class Evidence(BaseModel):
    chunk: PaperChunk         # Origin
    summary: str              # LLM summary
    source_query: str         # Query that retrieved this evidence
    vector_score: float       # Similarity to query
    llm_score: float          # LLM relevance score
    combined_score: float     # Weighted final score
    
```

## Phase 3: Section Generation

Sections are generated in this order:
1. **Methods**
2. **Results** (with figure integration)
3. **Discussion**
4. **Introduction**
5. **Related Work**
6. **Conclusion**
7. **Abstract**

After all sections are generated, the **Title** is created using abstract, introduction, and conclusion as context.

### Process

#### Step 3.1: Build Generation Prompt
For each section:
- **Role**: Expert academic writer
- **Task**: Write the complete section
- **Section Type**: Which section is being generated
- **Research Context**: Paper concept, hypothesis, experiment data
- **Evidence**: Formatted evidence with summaries and citation keys
- **Section Guidelines**: Section-specific writing instructions
- **Requirements**: General writing requirements (citation style, flow, etc.)

**Special Handling for Results Section**:
- If plots are available, adds figure integration instructions
- Includes plot filenames and captions
- Provides example format for markdown image syntax

#### Step 3.2: Generate Section
- Send prompt to LLM
- LLM generates section integrating evidence
- Citations appear naturally: `(smith2024quantum)`
- For Results: Figures integrated using `![alt text](filename.png)` followed by `*Figure N: Caption text*`

#### Step 3.3: Generate Title
- After all sections complete
- Uses abstract, introduction, and conclusion as context
- Generates concise, informative title

### Result
- `PaperDraft` object containing:
  - `title: str`
  - `abstract: str`
  - `introduction: str`
  - `related_work: str`
  - `methods: str`
  - `results: str`
  - `discussion: str`
  - `conclusion: str`
