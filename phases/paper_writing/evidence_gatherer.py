from __future__ import annotations

import heapq
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from phases.context_analysis.paper_conception import PaperConcept
from phases.context_analysis.user_requirements import UserRequirements
from phases.experimentation.experiment_state import ExperimentResult
from phases.paper_writing.data_models import Evidence, PaperChunk, Section, ScoreResult, SummaryBatchResult, ScoreBatchResult
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

    def search_evidence(
        self,
        query: str,
        target_section: Section,
        initial_chunks: int = 10,
        filtered_chunks: int = 5,
        exclude_chunk_ids: Optional[Set[str]] = None,
        llm_model=None,
        embedding_model=None,
    ) -> List[Evidence]:
        """Run the full evidence retrieval pipeline for a query."""

        retrieved_chunks = self._vector_search(query, initial_chunks, exclude_chunk_ids, embedding_model)
        summarized_chunks = self._summarize_chunks_batch(query, retrieved_chunks, llm_model=llm_model)
        rescored_chunks = self._score_chunks_batch(query, target_section, summarized_chunks, llm_model=llm_model)
        
        return self._combine_scores(query, rescored_chunks, filtered_chunks)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        exclude_chunk_ids: Optional[Set[str]] = None,
        embedding_model=None,
    ) -> List[Tuple[PaperChunk, float]]:
        """Return top_k chunks by cosine similarity to the query embedding."""

        if embedding_model is None:
            raise ValueError("embedding_model must be provided to avoid model loading conflicts")
        
        query_embedding = np.array(embedding_model.embed(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        normalized_query = query_embedding / query_norm

        scored_chunks: List[Tuple[float, PaperChunk]] = []
        for normalized_chunk in self._normalized_chunks:
            if exclude_chunk_ids and normalized_chunk.chunk.chunk_id in exclude_chunk_ids:
                continue
            
            score = float(np.dot(normalized_query, normalized_chunk.vector))
            scored_chunks.append((score, normalized_chunk.chunk))

        top_chunks = heapq.nlargest(top_k, scored_chunks, key=lambda item: item[0])
        return [(chunk, score) for score, chunk in top_chunks]

    def _summarize_chunks_batch(
        self,
        query: str,
        chunks: List[Tuple[PaperChunk, float]],
        batch_size: int = 5,
        llm_model=None,
    ) -> List[Tuple[PaperChunk, float, str]]:
        """Summarize chunks in batches."""
        
        if not chunks:
            return []
        
        if llm_model is None:
            raise ValueError("llm_model must be provided")
        results: List[Tuple[PaperChunk, float, str]] = []
        total = len(chunks)
        print(f"    Summarizing {total} chunks in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            # print(f"      Summarizing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
            
            prompt = self._build_batch_summary_prompt(query, batch)
            
            try:
                response = llm_model.respond(
                    prompt,
                    response_format=SummaryBatchResult,
                    config={"temperature": 0.3},
                )
                
                batch_results = response.parsed.get('results', [])
                
                if len(batch_results) < len(batch):
                    print(f"[DEBUG] Batch mismatch! Expected {len(batch)}, got {len(batch_results)}")
                    print(f"[DEBUG] Raw response content: {response.content}")
                    print(f"[DEBUG] Parsed response: {response.parsed}")
                
                # Map results by index (assuming order is preserved)
                for j, (chunk, vector_score) in enumerate(batch):
                    if j < len(batch_results):
                        item = batch_results[j]
                        summary = item.get('summary') if isinstance(item, dict) else getattr(item, 'summary', None)
                            
                        if summary:
                            results.append((chunk, vector_score, summary))
                        else:
                             results.append((chunk, vector_score, "Summary missing"))
                    else:
                        print(f"[WARNING] LLM returned fewer summaries than requested ({len(batch_results)} vs {len(batch)}). Using fallback.")
                        results.append((chunk, vector_score, "Summary failed"))
                        
            except Exception as e:
                print(f"[WARNING] Batch summarization failed: {e}")
                for chunk, vector_score in batch:
                    results.append((chunk, vector_score, "Batch summary failed"))
                    
        return results

    def _score_chunks_batch(
        self,
        query: str,
        target_section: Section,
        chunks: List[Tuple[PaperChunk, float, str]],
        batch_size: int = 5,
        llm_model=None,
    ) -> List[Tuple[PaperChunk, float, str, float]]:
        """Score chunks in batches."""
        
        if llm_model is None:
            raise ValueError("llm_model must be provided")
        results: List[Tuple[PaperChunk, float, str, float]] = []
        total = len(chunks)
        print(f"    Scoring {total} chunks in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            # print(f"      Scoring batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
            
            prompt = self._build_batch_scoring_prompt(query, target_section, batch)
            
            try:
                response = llm_model.respond(
                    prompt,
                    response_format=ScoreBatchResult,
                    config={"temperature": 0.2, "maxTokens": 1000},
                )
                
                # User confirmed response.parsed is always a dict when schema is used
                batch_results = response.parsed.get('results', [])
                
                # Map results by index
                for j, (chunk, vector_score, summary) in enumerate(batch):
                    if j < len(batch_results):
                        item = batch_results[j]
                        score_val = item.get('score') if isinstance(item, dict) else getattr(item, 'score', None)
                            
                        if score_val is not None:
                            score = self._clamp_score(float(score_val))
                            results.append((chunk, vector_score, summary, score))
                        else:
                            results.append((chunk, vector_score, summary, 0.0))
                    else:
                        print(f"[WARNING] LLM returned fewer scores than requested ({len(batch_results)} vs {len(batch)}). Using fallback.")
                        results.append((chunk, vector_score, summary, 0.0))
                        
            except Exception as e:
                print(f"[WARNING] Batch scoring failed: {e}")
                for chunk, vector_score, summary in batch:
                    results.append((chunk, vector_score, summary, 0.0))
                    
        return results

    def _build_batch_summary_prompt(self, query: str, batch: List[Tuple[PaperChunk, float]]) -> str:
        items_text = []
        for j, (chunk, _) in enumerate(batch):
            content = chunk.chunk_text
            items_text.append(textwrap.dedent(f"""\
                <chunk>
                  <title>{chunk.paper.title}</title>
                  <content>
                    {content}
                  </content>
                </chunk>
            """))
        
        return textwrap.dedent(f"""\
            [ROLE]
            You are assisting with academic literature review.

            [TASK]
            For each provided chunk, provide a summary of the content in a few sentences.

            [INSTRUCTIONS]
            Your response MUST be a JSON object conforming to the `SummaryBatchResult` schema.
            You MUST return exactly {len(batch)} summaries, one for each item, in the same order as provided.
            Never skip any items.
            IMPORTANT:
            - Content must be completely self-contained.
            - REMOVE all in-text citations (e.g., [11], [Sutton1990]).
            - REMOVE references to specific authors or papers mentioned in the text (e.g., avoid "As Sutton states..." or "In [1] it is shown...").
            - Focus purely on the concepts, findings, and arguments presented.
            - Do not mention that the text is "citing" other works.
            - If the text says "We propose", summarize it as "The study proposes" or "The authors propose".

            [CHUNKS]
            {"".join(items_text)}"""
        )

    def _build_batch_scoring_prompt(self, query: str, target_section: Section, batch: List[Tuple[PaperChunk, float, str]]) -> str:
        items_text = []
        for j, (chunk, _, _) in enumerate(batch):
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
            You are rating the relevance of text chunks for academic writing.

            [INSTRUCTIONS]
            Rate how relevant the each text chunks is for the target section and query.
            For each item, provide:
            1. score: 0.0 (not relevant) to 1.0 (highly relevant).
            2. reason: Brief reason (1 sentence).
            
            Your response MUST be a JSON object conforming to the `ScoreBatchResult` schema.
            Ensure you return exactly one score for each item, in the same order.

            [QUERY]
            {query}

            [TARGET SECTION]
            {target_section.value}

            [TEXT CHUNKS]
            {"".join(items_text)}
        """)

    @staticmethod
    def _normalize_corpus(indexed_corpus: Sequence[PaperChunk]) -> List[_NormalizedChunk]:
        normalized: List[_NormalizedChunk] = []
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
        chunks: List[Tuple[PaperChunk, float, str, float]],
        filtered_chunks: int,
    ) -> List[Evidence]:
        """Combine vector and LLM scores and return top evidence."""

        weighted: List[Evidence] = []
        for chunk, vector_score, summary, llm_score in chunks:
            combined = (0.3 * vector_score) + (0.7 * llm_score)
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

        weighted.sort(key=lambda ev: ev.combined_score, reverse=True)
        return weighted[:filtered_chunks]

    @staticmethod
    def _deduplicate_evidence(evidence_list: Sequence[Evidence]) -> List[Evidence]:
        """Keep the highest scoring evidence per chunk."""

        best_by_chunk: Dict[str, Evidence] = {}
        for evidence in evidence_list:
            chunk_id = evidence.chunk.chunk_id
            existing = best_by_chunk.get(chunk_id)
            if existing is None or evidence.combined_score > existing.combined_score:
                best_by_chunk[chunk_id] = evidence

        deduplicated = list(best_by_chunk.values())
        deduplicated.sort(key=lambda ev: ev.combined_score, reverse=True)
        return deduplicated

    def gather_evidence(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        default_queries: Sequence[str],
        max_iterations: int = 5,
        initial_chunks: int = 20,
        filtered_chunks: int = 10,
        user_requirements: Optional[UserRequirements] = None,
    ) -> Tuple[List[Evidence], str]:
        """Gather evidence using agentic iterative search with default queries as starting point."""

        # Step 1: Create embedding model FIRST and do all initial vector searches
        embedding_model = lms.embedding_model(Settings.PAPER_INDEXING_EMBEDDING_MODEL)
        
        collected_evidence: List[Evidence] = []
        seen_chunk_ids: Set[str] = set()
        all_retrieved_chunks: List[Tuple[str, Section, List[Tuple[PaperChunk, float]]]] = []
        
        for query in default_queries:
            if not query:
                continue
            # Do vector search with embedding model
            retrieved_chunks = self._vector_search(query, initial_chunks, seen_chunk_ids, embedding_model)
            all_retrieved_chunks.append((query, section_type, retrieved_chunks))
            # Update seen_chunk_ids based on retrieved chunks
            seen_chunk_ids.update(chunk.chunk_id for chunk, _ in retrieved_chunks)
        
        # Step 2: Create LLM model ONCE (this unloads embedding model)
        llm_model = lms.llm(Settings.EVIDENCE_GATHERING_MODEL)
        
        # Step 3: Now do all summarization and scoring (using LLM model)
        for query, target_section, retrieved_chunks in all_retrieved_chunks:
            summarized_chunks = self._summarize_chunks_batch(query, retrieved_chunks, llm_model=llm_model)
            rescored_chunks = self._score_chunks_batch(query, target_section, summarized_chunks, llm_model=llm_model)
            new_evidence = self._combine_scores(query, rescored_chunks, filtered_chunks)
            collected_evidence.extend(new_evidence)
            seen_chunk_ids.update(ev.chunk.chunk_id for ev in new_evidence)

        collected_evidence = self._deduplicate_evidence(collected_evidence)
        seen_chunk_ids.update(ev.chunk.chunk_id for ev in collected_evidence)
        executed_queries = {query.strip().lower() for query in default_queries if query}
        tool_calls = 0
        tool_results: List[str] = []

        def search_evidence_tool(query: str) -> str:
            """Search for additional evidence using a custom query string."""

            nonlocal collected_evidence, tool_calls, seen_chunk_ids, tool_results, llm_model, embedding_model

            if tool_calls >= max_iterations:
                result = "Tool call limit reached; reuse existing evidence."
                tool_results.append(result)
                return result

            cleaned_query = (query or "").strip()
            if not cleaned_query:
                result = "Query was empty; no search performed."
                tool_results.append(result)
                return result
            query_key = cleaned_query.lower()
            if query_key in executed_queries:
                result = "Query already executed; provide a new angle."
                tool_results.append(result)
                return result

            tool_calls += 1
            executed_queries.add(query_key)
            
            print(f"  [Agent Tool Call #{tool_calls}] Query: \"{cleaned_query}\"")
            
            # With LMSJITSettings, we can use the existing models without reloading!
            new_evidence = self.search_evidence(
                cleaned_query,
                section_type,
                initial_chunks=initial_chunks,
                filtered_chunks=filtered_chunks,
                exclude_chunk_ids=seen_chunk_ids,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
            
            collected_evidence = self._deduplicate_evidence(collected_evidence + new_evidence)
            seen_chunk_ids.update(ev.chunk.chunk_id for ev in new_evidence)
            result = self._summarize_tool_results(new_evidence)
            tool_results.append(result)
            print(f"    Added {len(new_evidence)} new evidence items")
            
            return result

        initial_prompt = self.build_agent_prompt(section_type, context, experiment, collected_evidence, user_requirements)

        try:
            # print(f"  [Agentic Search] Starting iterative evidence gathering...")
            llm_model.act(
                initial_prompt,
                tools=[search_evidence_tool],
                config={"temperature": 0.4},
            )
        except Exception as exc:
            # Surface issue but return whatever evidence we have
            print(f"[EvidenceGatherer] Agentic search failed: {exc}")

        # Reconstruct the full prompt as the LLM saw it: initial prompt + tool results
        final_prompt = self._reconstruct_full_prompt(initial_prompt, tool_results)
        
        return collected_evidence, final_prompt

    @staticmethod
    def _reconstruct_full_prompt(initial_prompt: str, tool_results: List[str]) -> str:
        """Reconstruct the full prompt as the LLM saw it, including tool call results."""
        if not tool_results:
            return initial_prompt
        
        tool_results_section = ""
        for idx, result in enumerate(tool_results, 1):
            tool_results_section += f"\n{'─' * 80}\n"
            tool_results_section += f"TOOL CALL #{idx}\n"
            tool_results_section += f"{'─' * 80}\n\n"
            tool_results_section += f"{result}\n"
        
        return initial_prompt + tool_results_section

    def build_agent_prompt(
        self,
        section_type: Section,
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
        initial_evidence: Sequence[Evidence],
        user_requirements: Optional[UserRequirements] = None,
    ) -> str:
        """Construct the system prompt used to guide the agentic evidence search."""

        objectives = self.get_section_objectives(section_type)
        context_text = self._format_section_context(context, experiment)
        evidence_text = self._format_evidence_for_prompt(initial_evidence)
        
        # Map Section enum to UserRequirements field
        section_to_requirement = {
            Section.ABSTRACT: "abstract",
            Section.INTRODUCTION: "introduction",
            Section.RELATED_WORK: "related_work",
            Section.METHODS: "methods",
            Section.RESULTS: "results",
            Section.DISCUSSION: "discussion",
            Section.CONCLUSION: "conclusion",
        }
        
        # Get section-specific user requirements if available
        user_requirements_block = ""
        if user_requirements:
            requirement_field = section_to_requirement.get(section_type)
            if requirement_field:
                requirement_text = getattr(user_requirements, requirement_field, None)
                if requirement_text and requirement_text.strip():
                    user_requirements_block = f"""[USER REQUIREMENTS]\n{requirement_text.strip()}"""

        return textwrap.dedent(f"""\
            [ROLE]
            You are an autonomous research assistant tasked with gathering evidence for an academic paper section.
            Decide whether additional searches are required, and call tools only when necessary.
            Focus on filling remaining gaps needed to draft a rigorous section of an academic paper.

            [SECTION TYPE]
            {section_type.value}

            [SECTION OBJECTIVES]
            {objectives}

            [RESEARCH CONTEXT]
            {context_text}

            [CURRENT EVIDENCE]
            {evidence_text or 'No evidence yet.'}

            {user_requirements_block}

            [AVAILABLE TOOL]
            search_evidence(query: str): retrieve additional information related to the query.
            The query is used to generate an embedding, which is then used to find the most semantically similar chunks in a corpus of academic papers.
            Use this tool when additional supporting material is required.

            [RESPONSIBILITIES]
            - Review existing evidence and identify missing aspects.
            - Formulate focused queries when additional evidence is needed.
            - Avoid redundant searches—do not repeat queries that were already covered.
            - When done searching, respond with ONLY "done" to indicate evidence gathering is complete."""
        )

    @staticmethod
    def _format_evidence_for_prompt(evidence: Sequence[Evidence]) -> str:

        if not evidence:
            return "No evidence available."
        
        lines: List[str] = []
        for idx, item in enumerate(evidence, 1):
            citation_key = item.chunk.paper.citation_key or "unknown"
            year = item.chunk.paper.published or "n.d."
            
            lines.append(f"[{citation_key}]")
            lines.append(f"    Title: {item.chunk.paper.title}")
            lines.append(f"    Year: {year}")
            lines.append(f"    Content: {item.summary}")
            lines.append("")  # Blank line between entries
        
        return "\n".join(lines).strip()

    @staticmethod
    def _format_section_context(
        context: PaperConcept,
        experiment: Optional[ExperimentResult],
    ) -> str:
        context_parts = [
            ("Paper concept description", context.description),
            ("Open research questions", context.open_questions),
        ]

        if experiment:
            context_parts.extend(
                [
                    ("Hypothesis", getattr(experiment.hypothesis, "description", "")),
                    ("Methods", getattr(experiment.hypothesis, "methods", "")),
                    ("Success criteria", getattr(experiment.hypothesis, "success_criteria", "")),
                    ("Experiment plan", experiment.experiment_plan),
                    ("Key execution outcome", getattr(experiment.execution_result, "stdout", "")),
                    ("Verdict", getattr(experiment.hypothesis_evaluation, "verdict", "")),
                ]
            )

        formatted_parts = [
            f"[{label.upper()}]\n{value.strip()}"
            for label, value in context_parts
            if isinstance(value, str) and value.strip()
        ]
        return "\n\n".join(formatted_parts)

    @staticmethod
    def get_section_objectives(section_type: Section) -> str:
        objectives_map = {
            Section.ABSTRACT: 
            """Capture the overarching motivation and key findings.
             Highlight the methodology and main results succinctly.""",
            Section.INTRODUCTION: 
            """Establish background and motivation.
            Clarify the gap the paper addresses and its hypothesis.""",
            Section.RELATED_WORK:
            """Review and synthesize existing work in the field.
            Position this research relative to prior contributions, identify gaps, and compare approaches.
            Organize by themes, methods, or chronological development.""",
            Section.METHODS: 
            """Describe the experiment setup and methodology.
            Reference comparable techniques or baselines.""",
            Section.RESULTS: 
            """Compare the experiment outcomes with related benchmarks.
            Highlight quantitative or qualitative findings.""",
            Section.DISCUSSION: 
            """Interpret results, limitations, and implications.
            Contrast findings with prior work and suggest future research.""",
            Section.CONCLUSION: 
            """Summarize contributions and key takeaways.
            Outline broader impact and if senseful, next steps.""",
        }

        return objectives_map.get(section_type, "")

    @staticmethod
    def _summarize_tool_results(evidence: Sequence[Evidence]) -> str:
        """Return a detailed summary of retrieved evidence for the agent."""
        if not evidence:
            return "No relevant evidence retrieved."

        lines = ["[RETRIEVED EVIDENCE]"]
        for idx, item in enumerate(evidence, 1):
            lines.append(f"{idx}. {item.chunk.paper.title}")
            lines.append(
                f"   Scores → vector: {item.vector_score:.3f}, "
                f"llm: {item.llm_score:.3f}, combined: {item.combined_score:.3f}"
            )
            lines.append(f"   Summary: {item.summary}")
            lines.append("")
        return "\n".join(lines).strip()

