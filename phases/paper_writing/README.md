# Paper Writing Pipeline

An agentic, RAG-based system for generating academic paper sections with indirect citations. Uses a 3-phase approach:
1. Indexing a corpus of research papers into searchable chunks
2. Agentic evidence search for each section of the paper
3. Writing each section of the paper, one by one

This system is inspired by **PaperQA**, an agent-driven tool that uses RAG to gather information. It's capable of answering scientific questions, summarizing papers and more. Check it out here: https://github.com/Future-House/paper-qa?tab=readme-ov-file

## Components

- **`PaperIndexer`**: Chunks and embeds papers into a searchable corpus
- **`QueryBuilder`**: Constructs default queries for each section of the paper
- **`EvidenceGatherer`**: Vector search, summarizing and scoring chunks, and agentic search
- **`PaperWriter`**: Generates paper sections
- **`PaperWritingPipeline`**: Orchestrates the entire process

## 1. PaperIndexer Class

Converts papers into searchable chunks with embeddings.

### Process
1. **Preprocessing**: Clean markdown text
2. **Chunking**: Split papers into chunks with small token overlap
3. **Embedding**: Create embeddings for chunks in batches (batch size can be changed in settings)
4. **Caching**: Embeddings are saved automatically and can be loaded again (via LOAD_PAPER_EMBEDDINGS setting)

### Output
- `List[PaperChunk]` with `chunk_id`, `paper`, `chunk_text`, `chunk_index`, `embedding`

### Data Structure
```python
PaperChunk {
    chunk_id: str
    paper: Paper  # source
    chunk_text: str
    chunk_index: int
    embedding: List[float]
}
```
## 2. EvidenceGatherer Class

An LLM agent gathers evidence:
1. Execute default queries (one per section)
2. Agent analyzes evidence and identifies gaps
3. Agent calls `search_evidence` tool with custom queries
4. Continues until sufficient evidence or max iterations reached

### Settings
- `EVIDENCE_INITIAL_CHUNKS`: Chunks retrieved from vector search
- `EVIDENCE_FILTERED_CHUNKS`: Final chunks after LLM filtering
- `EVIDENCE_AGENTIC_ITERATIONS`: Max agent tool calls

### Search Evidence Process

#### 1. Vector Search
- Embed query, compute cosine similarity
- Select most relevant chunks
- Exclude already-seen chunks

#### 2. Batch Summarization
- Process chunks in batches
- LLM summarizes each chunk in context of query
- Uses structured output

#### 3. Batch Scoring
- Process summarized chunks in batches
- LLM scores relevance (0.0-1.0)
- Uses structured output

#### 4. Combined Scoring & Selection
- Combined score: `0.3 * vector_score + 0.7 * llm_score`
- Sort by combined score, select top `filtered_chunks`
- Return `Evidence` objects

### Output
- `evidence: List[Evidence]` - Aggregated evidence from default queries + agent searches

### Evidence Structure
```python
class Evidence:
    chunk: PaperChunk         # Source chunk
    summary: str              # LLM summary
    source_query: str         # Query that retrieved this
    vector_score: float       # Cosine similarity
    llm_score: float          # LLM relevance score
    combined_score: float     # Weighted final score
```

## PaperWriter Class

Sections are generated in order: Methods → Results → Discussion → Introduction → Related Work → Conclusion → Abstract

### Process
1. **Build Prompt**: Includes role, task, section type, research context, evidence, and guidelines
2. **Generate**: LLM generates section with integrated citations
3. **Title**: Generated after all sections using abstract, introduction, and conclusion

### Special Handling
- **Results Section**: Includes figure integration instructions and captions


### Output
- `PaperDraft` with all sections and title
