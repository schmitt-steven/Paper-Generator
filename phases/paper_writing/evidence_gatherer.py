from __future__ import annotations

import heapq
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from phases.context_analysis.paper_conception import PaperConcept
from phases.experimentation.experiment_state import ExperimentResult
from phases.paper_writing.data_models import Evidence, PaperChunk, Section, ScoreResult
from utils.lazy_model_loader import LazyModelMixin, LazyEmbeddingMixin


@dataclass
class _NormalizedChunk:
    chunk: PaperChunk
    vector: np.ndarray


class EvidenceGatherer(LazyModelMixin, LazyEmbeddingMixin):
    """Retrieves and scores evidence chunks for a given query."""

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        indexed_corpus: Sequence[PaperChunk],
    ) -> None:
        self.model_name = llm_model_name  # For LazyModelMixin
        self._model = None  # Lazy-loaded via LazyModelMixin (accessed as self.model)
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None  # Lazy-loaded via LazyEmbeddingMixin

        self.indexed_corpus = list(indexed_corpus)
        self._normalized_chunks = self._normalize_corpus(indexed_corpus)
    
    @property
    def llm_model(self):
        """Alias for model property (for backward compatibility)."""
        return self.model

    def search_evidence(
        self,
        query: str,
        target_section: Section,
        top_k_initial: int = 20,
        top_k_final: int = 5,
        exclude_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Evidence]:
        """Run the full evidence retrieval pipeline for a query."""

        vector_candidates = self._vector_search(query, top_k_initial, exclude_chunk_ids)
        summarized_candidates = self._contextual_summarize(query, vector_candidates)
        rescored_candidates = self._llm_rescore(query, target_section, summarized_candidates)
        return self._combine_scores(query, rescored_candidates, top_k_final)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        exclude_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[PaperChunk, float]]:
        """Return top_k chunks by cosine similarity to the query embedding."""

        query_embedding = np.array(self.embedding_model.embed(query), dtype=np.float32)
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

    def _contextual_summarize(
        self,
        query: str,
        candidates: List[Tuple[PaperChunk, float]],
    ) -> List[Tuple[PaperChunk, float, str]]:
        """Summarize how each candidate chunk relates to the query context."""

        summaries: List[Tuple[PaperChunk, float, str]] = []
        for chunk, vector_score in candidates:
            prompt = self._build_summary_prompt(query, chunk)
            response = self.llm_model.respond(
                prompt,
                config={"temperature": 0.3, "maxTokens": 220},
            )
            summary_text = self._extract_response_text(response)
            summaries.append((chunk, vector_score, summary_text))
        return summaries

    def _llm_rescore(
        self,
        query: str,
        target_section: Section,
        candidates: List[Tuple[PaperChunk, float, str]],
    ) -> List[Tuple[PaperChunk, float, str, float]]:
        """Assign LLM relevance scores (0.0-1.0) to summarized evidence."""

        rescored: List[Tuple[PaperChunk, float, str, float]] = []
        for chunk, vector_score, summary in candidates:
            prompt = self._build_scoring_prompt(query, target_section, chunk, summary)
            response = self.llm_model.respond(
                prompt,
                response_format=ScoreResult,
                config={"temperature": 0.2, "maxTokens": 120},
            )
            parsed_dict = response.parsed
            score = self._clamp_score(float(parsed_dict.get("score", 0.0)))
            rescored.append((chunk, vector_score, summary, score))
        return rescored

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
    def _extract_response_text(response) -> str:
        """Safely convert an LM Studio response to plain text."""

        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()

    def _build_summary_prompt(self, query: str, chunk: PaperChunk) -> str:
        """Construct a prompt asking the LLM to summarize chunk relevance."""

        authors = ", ".join(chunk.paper.authors) if chunk.paper.authors else "Unknown authors"
        published = chunk.paper.published or "Unknown year"
        excerpt = self._truncate_chunk(chunk.chunk_text)

        return textwrap.dedent(f"""\
            [ROLE]
            You are assisting with academic literature review. Summarize the following paper chunk in the context of the research query. Follow these rules:
            - Use ~3 sentences.
            - Emphasize why the chunk is relevant to the query.
            - Do not quote verbatim; paraphrase.
            - Mention the key idea and its relation to the query.

            [QUERY]
            {query}

            [PAPER]
            {chunk.paper.title} ({authors}, {published})

            [CHUNK]
            \"\"\"
            {excerpt}
            \"\"\"
        """)

    @staticmethod
    def _truncate_chunk(chunk_text: str, max_chars: int = 3500) -> str:
        if len(chunk_text) <= max_chars:
            return chunk_text
        return chunk_text[: max_chars - 3].rstrip() + "..."

    def _build_scoring_prompt(self, query: str, target_section: Section, chunk: PaperChunk, summary: str) -> str:
        authors = ", ".join(chunk.paper.authors) if chunk.paper.authors else "Unknown authors"
        published = chunk.paper.published or "Unknown year"

        return textwrap.dedent(f"""\
            [ROLE]
            You are rating the relevance of evidence for academic writing.
            Rate how well the evidence summary supports the research query for the target section.
            Provide a score between 0 and 1, where 1 indicates highly relevant and 0 indicates not relevant.

            [TARGET SECTION]
            {target_section.value}

            [QUERY]
            {query}

            [PAPER]
            {chunk.paper.title} ({authors}, {published})

            [EVIDENCE SUMMARY]
            {summary}
        """)

    @staticmethod
    def _clamp_score(score: float) -> float:
        """Clamp score value between 0 and 1."""
        return float(min(max(score, 0.0), 1.0))

    def _combine_scores(
        self,
        query: str,
        candidates: List[Tuple[PaperChunk, float, str, float]],
        top_k_final: int,
    ) -> List[Evidence]:
        """Combine vector and LLM scores and return top evidence."""

        weighted: List[Evidence] = []
        for chunk, vector_score, summary, llm_score in candidates:
            combined = (0.4 * vector_score) + (0.6 * llm_score)
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
        return weighted[:top_k_final]

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
        top_k_initial: int = 20,
        top_k_final: int = 5,
    ) -> List[Evidence]:
        """Gather evidence using agentic iterative search with default queries as starting point."""

        # Step 1: Execute default queries to get initial evidence
        collected_evidence: List[Evidence] = []
        seen_chunk_ids: Set[str] = set()
        for query in default_queries:
            if not query:
                continue
            new_evidence = self.search_evidence(
                query,
                section_type,
                top_k_initial,
                top_k_final,
                exclude_chunk_ids=seen_chunk_ids,
            )
            collected_evidence.extend(new_evidence)
            seen_chunk_ids.update(ev.chunk.chunk_id for ev in new_evidence)

        collected_evidence = self._deduplicate_evidence(collected_evidence)
        seen_chunk_ids.update(ev.chunk.chunk_id for ev in collected_evidence)
        executed_queries = {query.strip().lower() for query in default_queries if query}
        tool_calls = 0
        tool_results: List[str] = []

        def search_evidence_tool(query: str) -> str:
            """Search for additional evidence using a custom query string."""

            nonlocal collected_evidence, tool_calls, seen_chunk_ids, tool_results

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

            new_evidence = self.search_evidence(
                cleaned_query,
                section_type,
                top_k_initial=top_k_initial,
                top_k_final=top_k_final,
                exclude_chunk_ids=seen_chunk_ids,
            )
            collected_evidence = self._deduplicate_evidence(collected_evidence + new_evidence)
            seen_chunk_ids.update(ev.chunk.chunk_id for ev in new_evidence)
            result = self._summarize_tool_results(new_evidence)
            tool_results.append(result)
            return result

        initial_prompt = self.build_agent_prompt(section_type, context, experiment, collected_evidence)

        try:
            self.llm_model.act(
                initial_prompt,
                tools=[search_evidence_tool],
                config={"temperature": 0.2},
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
    ) -> str:
        """Construct the system prompt used to guide the agentic evidence search."""

        objectives = self.get_section_objectives(section_type)
        context_text = self._format_section_context(context, experiment)
        evidence_text = self._format_evidence_for_prompt(initial_evidence)

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

            [CURRENT EVIDENCE] ({len(initial_evidence)} items)
            {evidence_text or 'No evidence yet.'}

            [AVAILABLE TOOL]
            search_evidence(query: str): retrieve additional literature evidence related to the query.
            Use this when additional supporting material is required.

            [RESPONSIBILITIES]
            - Review existing evidence and identify missing aspects.
            - Formulate focused queries when additional evidence is needed.
            - Avoid redundant searches—do not repeat queries that were already covered.
            - When done searching, respond with ONLY "done" to indicate evidence gathering is complete.
        """)

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
                    ("Expected improvement", getattr(experiment.hypothesis, "expected_improvement", "")),
                    ("Experimental plan", experiment.experimental_plan),
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
            """Describe the experimental setup and methodology.
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

