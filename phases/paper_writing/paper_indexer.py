from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import lmstudio as lms

from phases.literature_review.arxiv_api import Paper
from phases.paper_writing.data_models import PaperChunk, Section
from utils.file_utils import preprocess_markdown


class PaperIndexer:
    """Builds an indexed corpus of paper chunks using whole-document chunking."""

    CODE_BLOCK_PATTERN = re.compile(r"```.+?```", re.DOTALL)

    def __init__(
        self,
        embedding_model_name: str,
        max_tokens_per_chunk: int = 500,
        min_tokens_per_chunk: int = 300,
        overlap_tokens: int = 100,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.embedding_model = lms.embedding_model(embedding_model_name)
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self.overlap_tokens = overlap_tokens

    def index_papers(self, papers: Sequence[Paper]) -> List[PaperChunk]:
        """Parse and chunk papers into indexed PaperChunk records."""

        chunk_specs: List[tuple[Paper, int, str, str]] = []
        for paper in papers:
            if not paper.markdown_text:
                continue

            cleaned_markdown = preprocess_markdown(paper.markdown_text)
            chunks = self._chunk_document(cleaned_markdown)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = self._build_chunk_id(paper.id, chunk_idx)
                chunk_specs.append((paper, chunk_idx, chunk_id, chunk_text))

        if not chunk_specs:
            return []

        embeddings = self._embed_texts([spec[3] for spec in chunk_specs])

        indexed_chunks: List[PaperChunk] = []
        for spec, embedding in zip(chunk_specs, embeddings):
            paper, chunk_idx, chunk_id, chunk_text = spec
            indexed_chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper=paper,
                    section_type=Section.INTRODUCTION,  # Default value, not used for filtering
                    chunk_text=chunk_text,
                    chunk_index=chunk_idx,
                    embedding=list(embedding),
                )
            )

        return indexed_chunks

    def _chunk_document(self, document_text: str) -> List[str]:
        """Chunk document text into overlapping windows while preserving structures."""

        blocks = self._split_into_blocks(document_text)
        if not blocks:
            return []

        chunks: List[str] = []
        current_blocks: List[str] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self._estimate_tokens(block)

            if current_blocks and current_tokens + block_tokens > self.max_tokens_per_chunk:
                chunks.append(self._join_blocks(current_blocks))

                overlap_blocks = self._collect_overlap_blocks(current_blocks)
                current_blocks = overlap_blocks + [block]
                current_tokens = sum(self._estimate_tokens(b) for b in current_blocks)
            else:
                current_blocks.append(block)
                current_tokens += block_tokens

        if current_blocks:
            chunks.append(self._join_blocks(current_blocks))

        if len(chunks) >= 2 and self._estimate_tokens(chunks[-1]) < self.min_tokens_per_chunk:
            merged_chunk = f"{chunks[-2]}\n\n{chunks[-1]}".strip()
            chunks[-2] = merged_chunk
            chunks.pop()

        return chunks

    def _split_into_blocks(self, text: str) -> List[str]:
        """Split text into blocks, keeping code fences intact."""

        blocks: List[str] = []
        last_end = 0

        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            pre_block = text[last_end : match.start()]
            blocks.extend(self._split_paragraph_blocks(pre_block))
            blocks.append(match.group().strip())
            last_end = match.end()

        tail = text[last_end:]
        blocks.extend(self._split_paragraph_blocks(tail))

        return [block for block in blocks if block]

    @staticmethod
    def _split_paragraph_blocks(text: str) -> List[str]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
        return [paragraph for paragraph in paragraphs if paragraph]

    def _collect_overlap_blocks(self, blocks: Sequence[str]) -> List[str]:
        """Collect blocks from the end until the overlap token budget is met."""

        overlap_blocks: List[str] = []
        accumulated_tokens = 0

        for block in reversed(blocks):
            overlap_blocks.insert(0, block)
            accumulated_tokens += self._estimate_tokens(block)
            if accumulated_tokens >= self.overlap_tokens:
                break

        return overlap_blocks

    @staticmethod
    def _join_blocks(blocks: Iterable[str]) -> str:
        return "\n\n".join(blocks).strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Approximate tokens using word count (average 1 token ≈ 0.75 words)
        return max(1, int(len(text.split()) / 0.75))

    @staticmethod
    def _build_chunk_id(paper_id: str, chunk_idx: int) -> str:
        safe_paper_id = paper_id.replace("/", "_").replace(":", "_")
        return f"{safe_paper_id}_chunk{chunk_idx:02d}"

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts sequentially."""

        return [self.embedding_model.embed(text) for text in texts]


