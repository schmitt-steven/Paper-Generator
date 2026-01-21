"""
Test script for evidence search pipeline.

This script allows you to test and debug the evidence retrieval process,
showing exactly what is retrieved, summarized, scored, and what ends up
in the final writing prompt.

Usage:
    python tests/test_evidence_search.py

Note: This test handles JIT model loading (one model at a time).
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phases.paper_search.paper import Paper
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_writing.paper_indexer import PaperIndexer
from phases.paper_writing.evidence_gatherer import EvidenceGatherer
from phases.paper_writing.data_models import Section, Evidence, PaperChunk
from phases.paper_writing.paper_writer import PaperWriter
from settings import Settings
import lmstudio as lms


# Match real pipeline settings
# Match real pipeline settings
INITIAL_CHUNKS = 10  # Retrieved per query
FILTERED_CHUNKS = 5  # Final per query
BATCH_SIZE = 5
MAX_CHUNKS_PER_PAPER = 2


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*60}")
    print(f" {title}")
    print(f"{'-'*60}\n")


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for display."""
    text = text.replace('\n', ' ').strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def load_papers() -> list[Paper]:
    """Load papers from the output folder."""
    papers_file = Path("output/papers.json")
    if not papers_file.exists():
        print(f"Error: {papers_file} not found. Please run the paper search first.")
        sys.exit(1)
    
    papers = LiteratureSearch.load_papers(str(papers_file))
    papers_with_text = [p for p in papers if p.markdown_text and p.markdown_text.strip()]
    
    print(f"Loaded {len(papers)} papers ({len(papers_with_text)} with text content)")
    return papers_with_text


def index_papers(papers: list[Paper], force_reindex: bool = False) -> list[PaperChunk]:
    """Index papers into chunks with embeddings."""
    print_header("STEP 1: INDEXING PAPERS")
    
    indexer = PaperIndexer(
        max_tokens_per_chunk=700,
        min_tokens_per_chunk=500,
        overlap_tokens=50,
    )
    
    embeddings_file = Path(indexer.EMBEDDINGS_FILE)
    
    if force_reindex and embeddings_file.exists():
        print(f"Force reindex: deleting existing {embeddings_file}")
        embeddings_file.unlink()
        indexed_corpus = indexer.index_papers(papers)
    elif embeddings_file.exists():
        # Load existing embeddings and build corpus manually
        print(f"Loading existing embeddings from {embeddings_file}")
        from utils.file_utils import load_json
        existing_embeddings = load_json(embeddings_file.name, str(embeddings_file.parent))
        
        # Build chunk definitions (no embedding generation)
        chunk_definitions = indexer._create_chunk_definitions(papers)
        
        indexed_corpus = []
        missing_count = 0
        for paper, chunk_idx, chunk_id, chunk_text in chunk_definitions:
            if chunk_id in existing_embeddings:
                indexed_corpus.append(
                    PaperChunk(
                        chunk_id=chunk_id,
                        paper=paper,
                        chunk_text=chunk_text,
                        chunk_index=chunk_idx,
                        embedding=existing_embeddings[chunk_id],
                    )
                )
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} chunks missing embeddings (will be skipped)")
        print(f"Loaded {len(indexed_corpus)} chunks from existing embeddings")
    else:
        # No existing embeddings, generate new ones
        print("No existing embeddings found. Generating new ones...")
        indexed_corpus = indexer.index_papers(papers)
    
    print(f"\nTotal chunks indexed: {len(indexed_corpus)}")
    print("\nSample chunks:")
    for i, chunk in enumerate(indexed_corpus[:3]):
        print(f"  [{i}] {chunk.paper.title[:50]}... (chunk {chunk.chunk_index})")
        print(f"      Text preview: {truncate(chunk.chunk_text, 100)}")
        print(f"      Embedding dims: {len(chunk.embedding)}")
    
    return indexed_corpus


def vector_search_with_precomputed_embedding(
    gatherer: EvidenceGatherer,
    query_embedding: np.ndarray,
    top_k: int,
    exclude_chunk_ids: Optional[set[str]] = None,
) -> list[tuple[PaperChunk, float]]:
    """Vector search using a pre-computed query embedding."""
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    normalized_query = query_embedding / query_norm

    scored_chunks = []
    for normalized_chunk in gatherer._normalized_chunks:
        if exclude_chunk_ids and normalized_chunk.chunk.chunk_id in exclude_chunk_ids:
            continue
        
        score = float(np.dot(normalized_query, normalized_chunk.vector))
        scored_chunks.append((score, normalized_chunk.chunk))

    import heapq
    top_chunks = heapq.nlargest(top_k, scored_chunks, key=lambda item: item[0])
    return [(chunk, score) for score, chunk in top_chunks]


def run_full_pipeline(
    gatherer: EvidenceGatherer,
    query: str,
    section_type: Section,
):
    """Run the complete evidence search pipeline step by step."""
    print_header("FULL PIPELINE TEST")
    print(f"Query: \"{query}\"")
    print(f"Target Section: {section_type.value}")
    print(f"Initial retrieval: {INITIAL_CHUNKS} chunks")
    print(f"Final count: {FILTERED_CHUNKS} chunks")
    
    # PHASE 1: Load embedding model and compute query embedding
    print_subheader("PHASE 1: Computing Query Embedding")
    print(f"Loading embedding model: {Settings.PAPER_INDEXING_EMBEDDING_MODEL}")
    embedding_model = lms.embedding_model(Settings.PAPER_INDEXING_EMBEDDING_MODEL)
    
    query_embedding = np.array(embedding_model.embed(query), dtype=np.float32)
    print(f"Query embedding computed (dims: {len(query_embedding)})")
    
    # Run vector search
    print_subheader(f"Vector Search Results")
    vector_results = vector_search_with_precomputed_embedding(
        gatherer, query_embedding, INITIAL_CHUNKS
    )
    
    print(f"Retrieved {len(vector_results)} chunks:\n")
    for i, (chunk, score) in enumerate(vector_results):
        print(f"  [{i+1}] Score: {score:.4f}")
        print(f"      Paper: {chunk.paper.title[:60]}...")
        print(f"      Chunk {chunk.chunk_index}: {truncate(chunk.chunk_text, 150)}")
        print()
    
    if not vector_results:
        print("No results found. Stopping.")
        return
    
    # Unload embedding model (JIT will handle this)
    del embedding_model
    
    # PHASE 2: Load LLM for summarization and scoring
    print_subheader("PHASE 2: LLM Summarization & Scoring")
    print(f"Loading LLM model: {Settings.PAPER_WRITING_MODEL}")
    llm_model = lms.llm(Settings.PAPER_WRITING_MODEL)
    
    # Combined Summarization & Scoring
    print("\n--- Combined Summarization & Scoring ---")
    processed = gatherer._process_chunks_combined(
        query, section_type, vector_results, batch_size=BATCH_SIZE, llm_model=llm_model
    )
    
    print(f"\nProcessed {len(processed)} chunks:")
    for i, (chunk, vector_score, summary, llm_score) in enumerate(processed):
        print(f"  [{i+1}] {chunk.paper.title[:50]}...")
        print(f"      Vector Score: {vector_score:.4f} | LLM Score: {llm_score:.4f}")
        print(f"      Summary: {truncate(summary, 200)}")
        print()
    
    # PHASE 3: Combine scores and show final result
    print_subheader("PHASE 3: Selection (Primary: LLM Score, Tie-break: Vector)")
    print(f"Max chunks per paper cap: {MAX_CHUNKS_PER_PAPER}")
    
    evidence = gatherer._combine_scores(query, processed, FILTERED_CHUNKS, MAX_CHUNKS_PER_PAPER)
    
    print(f"Top {len(evidence)} evidence pieces (after filtering):\n")
    for i, ev in enumerate(evidence):
        print(f"  [{i+1}] Combined: {ev.combined_score:.4f} (vec={ev.vector_score:.4f}, llm={ev.llm_score:.4f})")
        print(f"      Paper: {ev.chunk.paper.title[:50]}...")
        print(f"      Summary: {truncate(ev.summary, 200)}")
        print()
    
    # Show prompt format
    print_header("FINAL: EVIDENCE IN WRITING PROMPT")
    formatted = PaperWriter._format_evidence_for_prompt(evidence)
    print("The following is what gets added to the writing prompt:\n")
    print("-" * 60)
    print(formatted)
    print("-" * 60)


def run_batch_search_test(
    gatherer: EvidenceGatherer,
    queries: list[str],
    section_type: Section,
):
    """Test the batch_search method used by the critique pipeline."""
    print_header("BATCH SEARCH TEST")
    print(f"Target Section: {section_type.value}")
    print(f"Queries: {len(queries)}")
    print(f"Chunks per query: {FILTERED_CHUNKS}")
    
    for i, q in enumerate(queries):
        print(f"  [{i+1}] {q}")
    
    print("\nRunning batch search (this uses real pipeline settings)...")
    evidence = gatherer.batch_search(
        queries, section_type, chunks_per_query=FILTERED_CHUNKS, max_chunks_per_paper=MAX_CHUNKS_PER_PAPER
    )
    
    print_subheader("Batch Search Results")
    print(f"Total evidence retrieved: {len(evidence)}")
    
    for i, ev in enumerate(evidence):
        print(f"\n  [{i+1}] Combined: {ev.combined_score:.4f}")
        print(f"      Query: {truncate(ev.source_query, 60)}")
        print(f"      Paper: {ev.chunk.paper.title[:50]}...")
        print(f"      Summary: {truncate(ev.summary, 150)}")
    
    # Show prompt format
    print_header("FINAL: EVIDENCE IN WRITING PROMPT")
    formatted = PaperWriter._format_evidence_for_prompt(evidence)
    print("The following is what gets added to the writing prompt:\n")
    print("-" * 60)
    print(formatted)
    print("-" * 60)


def main():
    """Main entry point."""
    print_header("EVIDENCE SEARCH PIPELINE TEST")
    print(f"Settings: {INITIAL_CHUNKS} chunks retrieved, {FILTERED_CHUNKS} final per query")
    
    # Load papers
    papers = load_papers()
    if not papers:
        print("No papers with text content found. Exiting.")
        return
    
    # Ask about embeddings
    embeddings_file = Path("output/paper_embeddings.json")
    if embeddings_file.exists():
        print(f"\nExisting embeddings found: {embeddings_file}")
        reindex_choice = input("Regenerate embeddings? (y/N): ").strip().lower()
        force_reindex = reindex_choice == 'y'
    else:
        print("\nNo existing embeddings found. Will generate new ones.")
        force_reindex = True
    
    indexed_corpus = index_papers(papers, force_reindex=force_reindex)
    if not indexed_corpus:
        print("No chunks indexed. Exiting.")
        return
    
    # Create gatherer
    gatherer = EvidenceGatherer(indexed_corpus)
    
    # Interactive mode: let user choose what to test
    print_header("TEST OPTIONS")
    print("1. Run full pipeline with a single query (step-by-step)")
    print("2. Run batch search with multiple queries (uses real batch_search)")
    print("3. Custom query (interactive)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Default test query
        query = "What are the key challenges in offline reinforcement learning?"
        section_type = Section.RELATED_WORK
        
        run_full_pipeline(gatherer, query, section_type)
    
    elif choice == "2":
        # Batch search test - this uses the real batch_search method
        queries = [
            "What are the main approaches to offline RL?",
            "How do value-based methods handle distribution shift?",
        ]
        section_type = Section.RELATED_WORK
        
        run_batch_search_test(gatherer, queries, section_type)
    
    elif choice == "3":
        # Interactive mode
        query = input("Enter your search query: ").strip()
        if not query:
            print("Empty query. Exiting.")
            return
        
        print("\nSelect target section:")
        sections = [s for s in Section]
        for i, s in enumerate(sections):
            print(f"  {i+1}. {s.value}")
        
        section_choice = input(f"Select section (1-{len(sections)}): ").strip()
        try:
            section_type = sections[int(section_choice) - 1]
        except (ValueError, IndexError):
            section_type = Section.RELATED_WORK
            print(f"Invalid choice, defaulting to {section_type.value}")
        
        run_full_pipeline(gatherer, query, section_type)
    
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
