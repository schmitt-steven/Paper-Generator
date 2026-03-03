from __future__ import annotations

import json
import heapq
import textwrap
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from phases.paper_writing.data_models import Evidence, PaperChunk, Section, BatchResult
from utils.llm_utils import remove_thinking_blocks
from settings import Settings
import lmstudio as lms


@dataclass
class _NormalizedChunk:
    chunk: PaperChunk
    vector: np.ndarray


class EvidenceGatherer:
    """Retrieves and scores evidence chunks for a given query."""

    def __init__(
        self,
        indexed_corpus: Sequence[PaperChunk],
    ) -> None:
        self.indexed_corpus = list(indexed_corpus)
        self._normalized_chunks = self._normalize_corpus(indexed_corpus)

    def _search_evidence(
        self,
        query: str,
        target_section: Section,
        initial_chunks: int,
        filtered_chunks: int,
        batch_size: int,
        exclude_chunk_ids: Optional[set[str]] = None,
        llm_model=None,
        embedding_model=None,
        max_chunks_per_paper: int = 2,
    ) -> list[Evidence]:
        """Run the full evidence retrieval pipeline for a query."""

        retrieved_chunks = self._vector_search(query, initial_chunks, exclude_chunk_ids, embedding_model)
        processed_chunks = self._process_chunks_combined(query, target_section, retrieved_chunks, batch_size=batch_size, llm_model=llm_model)
        
        return self._combine_scores(query, processed_chunks, filtered_chunks, max_chunks_per_paper)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        exclude_chunk_ids: Optional[set[str]] = None,
        embedding_model=None,
    ) -> list[tuple[PaperChunk, float]]:
        """Return top_k chunks by cosine similarity to the query embedding."""

        if embedding_model is None:
            raise ValueError("embedding_model must be provided to avoid model loading conflicts")
        
        query_embedding = np.array(embedding_model.embed(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        normalized_query = query_embedding / query_norm

        scored_chunks: list[tuple[float, PaperChunk]] = []
        for normalized_chunk in self._normalized_chunks:
            if exclude_chunk_ids and normalized_chunk.chunk.chunk_id in exclude_chunk_ids:
                continue
            
            score = float(np.dot(normalized_query, normalized_chunk.vector))
            scored_chunks.append((score, normalized_chunk.chunk))

        top_chunks = heapq.nlargest(top_k, scored_chunks, key=lambda item: item[0])
        return [(chunk, score) for score, chunk in top_chunks]

    def _process_chunks_combined(
        self,
        query: str,
        target_section: Section,
        chunks: list[tuple[PaperChunk, float]],
        batch_size: int = 5,
        llm_model=None,
    ) -> list[tuple[PaperChunk, float, str, float]]:
        """Summarize and score chunks in a single pass."""
        
        if not chunks:
            return []
        
        if llm_model is None:
            raise ValueError("llm_model must be provided")

        results: list[tuple[PaperChunk, float, str, float]] = []
        total = len(chunks)
        print(f"    Processing {total} chunks in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            prompt = self._build_combined_prompt(query, target_section, batch)
            
            try:
                response = llm_model.respond(
                    prompt,
                    response_format=BatchResult,
                    config={"temperature": 0.2, "maxTokens": 1500},
                )
                
                content = remove_thinking_blocks(response.content)
                parsed = json.loads(content)
                batch_results = parsed.get('results', [])
                
                # Map results
                for j, (chunk, vector_score) in enumerate(batch):
                    if j < len(batch_results):
                        item = batch_results[j]
                        # Handling item as dict or object (usually dict with parsed)
                        summary = item.get('summary') if isinstance(item, dict) else getattr(item, 'summary', "Summary missing")
                        score_val = item.get('score') if isinstance(item, dict) else getattr(item, 'score', 0.0)
                        
                        score = self._clamp_score(float(score_val))
                        results.append((chunk, vector_score, summary, score))
                    else:
                        print(f"[WARNING] LLM returned fewer results than request.")
                        results.append((chunk, vector_score, "Processing failed", 0.0))
            except Exception as e:
                 print(f"[WARNING] Batch processing failed: {e}")
                 for chunk, vector_score in batch:
                     results.append((chunk, vector_score, "Processing failed", 0.0))
                     
        return results

    def _build_combined_prompt(self, query: str, target_section: Section, batch: list[tuple[PaperChunk, float]]) -> str:
        items_text = []
        for j, (chunk, _) in enumerate(batch):
            content = chunk.chunk_text
            items_text.append(textwrap.dedent(f"""\
                <text>
                  <title>{chunk.paper.title}</title>
                  <content>
                    {content}
                  </content>
                </text>
            """))

        return textwrap.dedent(f"""\
            [ROLE]
            You are assisting with academic literature review.

            [TASK]
            For each provided text chunk:
            1. Summarize the content in a few sentences.
            2. Rate the relevance of the chunk for the target section and query.
            3. Provide a short, concise reasoning.

            [INSTRUCTIONS]
            Your response MUST be a JSON object conforming to the `BatchResult` schema.
            You MUST return exactly {len(batch)} results, one for each item, in the same order as provided.
            Never skip any items.
            
            [SUMMARY RULES]
            - Content must be completely self-contained.
            - REMOVE all in-text citations (e.g., [11], [Sutton1990]).
            - REMOVE references to specific authors or papers mentioned in the text (e.g., avoid "As Sutton states..." or "In [1] it is shown...").
            - Focus purely on the contexts, findings, and arguments presented in the provided chunk.
            - Do not mention that the text is "citing" other works.
            - If the text says "We propose", summarize it as "The study proposes" or "The authors propose".
            - Do not bring in any outside knowledge, facts, or citations that are not explicitly present in the text chunk.
            - Do not generate any new citation keys (e.g. [Pearl2009]) even if you know the work being described.
            
            [SCORING RULES]
            Rate relevance on a scale of 0.0 to 1.0 (float):
            - 1.0: Extremely relevant. Directly answers the query with specific facts/findings.
            - 0.8: Highly relevant. Provides strong evidence or context.
            - 0.5: Moderately relevant. Somewhat related but vague or peripheral.
            - 0.1: Minimally relevant. Only mentions keywords in passing.
            - 0.0: Irrelevant.

            Target Section: {target_section.value}
            Query: "{query}"

            [CHUNKS]
            {"".join(items_text)}"""
        )

    @staticmethod
    def _normalize_corpus(indexed_corpus: Sequence[PaperChunk]) -> list[_NormalizedChunk]:
        normalized: list[_NormalizedChunk] = []
        for chunk in indexed_corpus:
            if not chunk.embedding:
                continue

            vector = np.array(chunk.embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue

            normalized.append(_NormalizedChunk(chunk=chunk, vector=vector / norm))
        return normalized

    @staticmethod
    def _clamp_score(score: float) -> float:
        """Clamp score value between 0 and 1."""
        return float(min(max(score, 0.0), 1.0))

    def _combine_scores(
        self,
        query: str,
        chunks: list[tuple[PaperChunk, float, str, float]],
        filtered_chunks: int,
        max_chunks_per_paper: int = 2,
    ) -> list[Evidence]:
        """Combine vector and LLM scores and return top evidence."""

        weighted: list[Evidence] = []
        for chunk, vector_score, summary, llm_score in chunks:
            # User request: Use LLM score as primary metric.
            # Vector score is only used for tie-breaking.
            
            # We set combined_score to llm_score so it reflects the primary metric
            combined = llm_score
            
            weighted.append(
                Evidence(
                    chunk=chunk,
                    summary=summary,
                    vector_score=vector_score,
                    llm_score=llm_score,
                    combined_score=combined,
                    source_query=query,
                )
            )

        # Sort by LLM score (descending), then Vector score (descending)
        weighted.sort(key=lambda ev: (ev.llm_score, ev.vector_score), reverse=True)
        
        # Filter by max chunks per paper
        final_list = []
        paper_counts = {}
        
        for ev in weighted:
            if len(final_list) >= filtered_chunks:
                break
            
            # Use citation key as ID, fallback to title
            paper_id = ev.chunk.paper.citation_key or ev.chunk.paper.title
            
            current_count = paper_counts.get(paper_id, 0)
            if current_count < max_chunks_per_paper:
                final_list.append(ev)
                paper_counts[paper_id] = current_count + 1
                
        return final_list

    @staticmethod
    def _deduplicate_evidence(evidence_list: Sequence[Evidence]) -> list[Evidence]:
        """Keep the highest scoring evidence per chunk."""

        best_by_chunk: dict[str, Evidence] = {}
        for evidence in evidence_list:
            chunk_id = evidence.chunk.chunk_id
            existing = best_by_chunk.get(chunk_id)
            
            # Use LLM score (primary) and Vector score (secondary) for comparison
            if existing is None:
                best_by_chunk[chunk_id] = evidence
            else:
                # Compare (llm, vector) tuples
                current_key = (evidence.llm_score, evidence.vector_score)
                existing_key = (existing.llm_score, existing.vector_score)
                
                if current_key > existing_key:
                    best_by_chunk[chunk_id] = evidence

        deduplicated = list(best_by_chunk.values())
        # Sort by LLM score (descending), then Vector score (descending)
        deduplicated.sort(key=lambda ev: (ev.llm_score, ev.vector_score), reverse=True)
        return deduplicated

    def batch_search(
        self,
        queries: list[str],
        section_type: Section,
        chunks_per_query: int = 3,
        max_chunks_per_paper: int = 2,
    ) -> list[Evidence]:
        """
        Execute multiple search queries in batch mode (non-agentic).
        
        This is used by the critique-based pipeline to search for evidence
        based on the critic's suggested queries.
        
        Args:
            queries: List of search queries from the SectionCritic
            section_type: Target section for relevance scoring
            chunks_per_query: Number of evidence chunks to return per query
            max_chunks_per_paper: Maximum number of chunks to return from a single paper
            
        Returns:
            Combined, deduplicated list of Evidence
        """
        if not queries:
            return []
        
        # Load models once for all queries
        embedding_model = lms.embedding_model(Settings.PAPER_INDEXING_EMBEDDING_MODEL)
        llm_model = lms.llm(Settings.PAPER_WRITING_MODEL)
        
        all_evidence: list[Evidence] = []
        seen_chunk_ids: set[str] = set()
        
        print(f"  [Batch Search] Executing {len(queries)} queries...")
        
        for idx, query in enumerate(queries):
            query = query.strip()
            if not query:
                continue
                
            print(f"    Query {idx + 1}/{len(queries)}: \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
            
            evidence = self._search_evidence(
                query=query,
                target_section=section_type,
                initial_chunks=chunks_per_query * 2,  # Retrieve more, filter down
                filtered_chunks=chunks_per_query,
                batch_size=chunks_per_query,          # Use chunks_per_query as batch size
                exclude_chunk_ids=seen_chunk_ids,
                llm_model=llm_model,
                embedding_model=embedding_model,
                max_chunks_per_paper=max_chunks_per_paper,
            )
            
            all_evidence.extend(evidence)
            seen_chunk_ids.update(ev.chunk.chunk_id for ev in evidence)
            print(f"      Found {len(evidence)} evidence chunks")
        
        # Deduplicate and sort by score
        deduplicated = self._deduplicate_evidence(all_evidence)
        print(f"  [Batch Search] Total: {len(deduplicated)} unique evidence chunks")
        
        return deduplicated
