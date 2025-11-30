from __future__ import annotations
import re
import time
from typing import List, Sequence, Iterable, Optional
from pathlib import Path
import lmstudio as lms
from settings import Settings
from utils.file_utils import save_json, load_json
from phases.paper_search.paper import Paper
from phases.paper_writing.data_models import PaperChunk



class PaperIndexer:
    """Builds an indexed corpus of paper chunks using whole-document chunking."""
    
    EMBEDDINGS_FILE = "output/paper_embeddings.json"

    CODE_BLOCK_PATTERN = re.compile(r"```.+?```", re.DOTALL)

    def __init__(
        self,
        max_tokens_per_chunk: int = 700,
        min_tokens_per_chunk: int = 500,
        overlap_tokens: int = 50,
    ) -> None:
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self.overlap_tokens = overlap_tokens

    def index_papers(self, papers: Sequence[Paper]) -> List[PaperChunk]:
        """Parse and chunk papers into indexed PaperChunk records."""

        print(f"\n{'='*80}")
        print(f"INDEXING {len(papers)} PAPERS")
        print(f"{'='*80}\n")
        
        # Try to load existing embeddings if enabled
        if Settings.LOAD_PAPER_EMBEDDINGS:
            existing_embeddings = self.load_embeddings()
            if existing_embeddings:
                # We have embeddings, but we still need to rebuild chunk_specs to create PaperChunk objects
                # Do minimal processing - just chunk without printing stats
                chunk_specs: List[tuple[Paper, int, str, str]] = []
                for paper in papers:
                    if not paper.markdown_text:
                        continue
                    cleaned_markdown = preprocess_markdown(paper.markdown_text)
                    cleaned_markdown = self._strip_references_section(cleaned_markdown)
                    chunks = self._chunk_document(cleaned_markdown)
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_id = self._build_chunk_id(paper.id, chunk_idx)
                        chunk_specs.append((paper, chunk_idx, chunk_id, chunk_text))
                
                if len(existing_embeddings) == len(chunk_specs):
                    print(f"Loaded {len(existing_embeddings)} existing embeddings from {self.EMBEDDINGS_FILE}")
                    embeddings = existing_embeddings
                else:
                    print(f"Found {len(existing_embeddings)} embeddings but need {len(chunk_specs)}. Re-generating...")
                    # Fall through to regenerate
                    chunk_specs, embeddings = self._process_and_embed_papers(papers)
            else:
                # No embeddings file found, do full processing
                chunk_specs, embeddings = self._process_and_embed_papers(papers)
        else:
            # Not loading embeddings, do full processing
            chunk_specs, embeddings = self._process_and_embed_papers(papers)
        
        if not chunk_specs:
            return []
        
        print(f"\nBuilding indexed corpus from {len(chunk_specs)} chunks...")

        indexed_chunks: List[PaperChunk] = []
        for spec, embedding in zip(chunk_specs, embeddings):
            paper, chunk_idx, chunk_id, chunk_text = spec
            indexed_chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper=paper,
                    chunk_text=chunk_text,
                    chunk_index=chunk_idx,
                    embedding=list(embedding),
                )
            )

        return indexed_chunks
    
    def _process_and_embed_papers(self, papers: Sequence[Paper]) -> tuple[List[tuple[Paper, int, str, str]], List[List[float]]]:
        """Process papers (preprocess, strip refs, chunk) and generate embeddings."""
        chunk_specs: List[tuple[Paper, int, str, str]] = []
        total_tokens_saved = 0
        papers_with_refs_stripped = 0
        
        for paper in papers:
            if not paper.markdown_text:
                continue

            cleaned_markdown = preprocess_markdown(paper.markdown_text)
            
            # Strip references and acknowledgments sections
            original_tokens = self._estimate_tokens(cleaned_markdown)
            cleaned_markdown = self._strip_references_section(cleaned_markdown)
            final_tokens = self._estimate_tokens(cleaned_markdown)
            tokens_saved = original_tokens - final_tokens
            if tokens_saved > 0:
                total_tokens_saved += tokens_saved
                papers_with_refs_stripped += 1
            chunks = self._chunk_document(cleaned_markdown)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = self._build_chunk_id(paper.id, chunk_idx)
                chunk_specs.append((paper, chunk_idx, chunk_id, chunk_text))

        if not chunk_specs:
            return [], []
        
        # Print summary of reference stripping
        print(f"PREPROCESSING SUMMARY:")
        print(f"  Papers processed: {len(papers)}")
        print(f"  References stripped: {papers_with_refs_stripped}/{len(papers)} papers")
        print(f"  Tokens saved: {total_tokens_saved:,}")
        print(f"  Total chunks created: {len(chunk_specs)}\n")
        
        print(f"Creating embeddings for {len(chunk_specs)} chunks...")
        embeddings = self._embed_texts([spec[3] for spec in chunk_specs])
        self.save_embeddings(embeddings)
        
        return chunk_specs, embeddings

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
        return max(1, int(len(text.split()) / 0.75))

    @staticmethod
    def _build_chunk_id(paper_id: str, chunk_idx: int) -> str:
        safe_paper_id = paper_id.replace("/", "_").replace(":", "_")
        return f"{safe_paper_id}_chunk{chunk_idx:02d}"

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts in batches."""
        if not texts:
            return []
        
        embedding_model = lms.embedding_model(Settings.PAPER_INDEXING_EMBEDDING_MODEL)
        batch_size = Settings.PAPER_EMBEDDING_BATCH_SIZE
        all_embeddings: List[List[float]] = []
        
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            batch_num = (i // batch_size) + 1
            
            start_time = time.time()
            batch_embeddings = embedding_model.embed(batch)
            elapsed = time.time() - start_time
            
            print(f"  Embedding batch {batch_num}/{num_batches} ({len(batch)} items)... Done in {elapsed:.2f}s")
            all_embeddings.extend(batch_embeddings)
                
        return all_embeddings



    def save_embeddings(self, embeddings: List[List[float]]) -> None:
        """Save embeddings to JSON file."""
        try:
            path_obj = Path(self.EMBEDDINGS_FILE)
            save_json(embeddings, path_obj.name, str(path_obj.parent))
            print(f"Saved {len(embeddings)} embeddings to {self.EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self) -> Optional[List[List[float]]]:
        """Load embeddings from JSON file if it exists."""
        path_obj = Path(self.EMBEDDINGS_FILE)
        if not path_obj.exists():
            return None

        try:
            embeddings = load_json(path_obj.name, str(path_obj.parent))
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None

    def _strip_references_section(self, text: str) -> str:
        """Remove references, acknowledgments, and bibliography sections from text."""
        import re
        
        # Pattern matches common reference section headers (case-insensitive, with optional markdown formatting)
        # Matches: REFERENCES, References, **References**, ACKNOWLEDGMENTS, etc.
        pattern = r'^\s*(?:\*\*)?(?:\d+\.?\s*)?(?:REFERENCES?|ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS?|BIBLIOGRAPHY)(?:\*\*)?(?:\s+.*)?$'
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.match(pattern, line, re.IGNORECASE):
                # Truncate at this line
                return '\n'.join(lines[:i])
        
        return text
