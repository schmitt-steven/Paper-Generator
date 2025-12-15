from __future__ import annotations
import re
import time
from typing import List, Sequence, Iterable, Optional
from pathlib import Path
import lmstudio as lms
import numpy as np  # Added for array handling if needed, though list is used for storage
from settings import Settings
from utils.file_utils import save_json, load_json, preprocess_markdown
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

    def index_papers(self, papers: Sequence[Paper]) -> list[PaperChunk]:
        """Parse and chunk papers into indexed PaperChunk records."""

        print(f"\n{'='*80}")
        print(f"INDEXING {len(papers)} PAPERS")
        print(f"{'='*80}\n")
        
        # 1. Generate all chunk definitions first
        chunk_definitions = self._create_chunk_definitions(papers)
        if not chunk_definitions:
            return []
            
        # 2. Load existing embeddings (dict)
        existing_embeddings: dict[str, list[float]] = {}
        # 2. Load existing embeddings (dict)
        existing_embeddings: dict[str, list[float]] = {}
        loaded = self.load_embeddings()
        if loaded:
            existing_embeddings = loaded
            print(f"Loaded {len(existing_embeddings)} existing embeddings.")
        
        # 3. Identify missing chunks
        missing_chunks: list[tuple[Paper, int, str, str]] = []
        for defn in chunk_definitions:
            chunk_id = defn[2] # (paper, idx, id, text)
            if chunk_id not in existing_embeddings:
                missing_chunks.append(defn)
        
        # 4. Embed missing chunks
        if missing_chunks:
            print(f"Found {len(missing_chunks)} new chunks to embed.")
            full_texts = [defn[3] for defn in missing_chunks]
            new_embeddings_list = self._embed_texts(full_texts)
            
            if len(new_embeddings_list) != len(missing_chunks):
                 print(f"Error: Mismatch in embeddings count. Expected {len(missing_chunks)}, got {len(new_embeddings_list)}")
                 # Handle error or continue carefully? 
                 # For now, zip will stop at shortest, which is safer than crashing but might lose data.
            
            # Update dictionary
            for defn, embedding in zip(missing_chunks, new_embeddings_list):
                 chunk_id = defn[2]
                 existing_embeddings[chunk_id] = embedding
            
            # Save updated dictionary
            self.save_embeddings(existing_embeddings)
        else:
            print("All chunks have existing embeddings. Skipping embedding generation.")

        # 5. Build final list of PaperChunks
        print(f"\nBuilding indexed corpus from {len(chunk_definitions)} chunks...")
        indexed_chunks: list[PaperChunk] = []
        
        for paper, chunk_idx, chunk_id, chunk_text in chunk_definitions:
            if chunk_id in existing_embeddings:
                indexed_chunks.append(
                    PaperChunk(
                        chunk_id=chunk_id,
                        paper=paper,
                        chunk_text=chunk_text,
                        chunk_index=chunk_idx,
                        embedding=existing_embeddings[chunk_id],
                    )
                )
            else:
                # Should not happen unless embedding failed
                print(f"Warning: No embedding found for {chunk_id}")

        return indexed_chunks
    
    def _create_chunk_definitions(self, papers: Sequence[Paper]) -> list[tuple[Paper, int, str, str]]:
        """Process papers and create chunk definitions (without embedding)."""
        chunk_definitions: list[tuple[Paper, int, str, str]] = []
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
                chunk_definitions.append((paper, chunk_idx, chunk_id, chunk_text))

        if not chunk_definitions:
            return []
        
        # Print summary of reference stripping (only if actually processing)
        if chunk_definitions:  # Always print summary if we processed something
            print(f"PREPROCESSING SUMMARY:")
            print(f"  Papers processed: {len(papers)}")
            print(f"  References stripped: {papers_with_refs_stripped}/{len(papers)} papers")
            print(f"  Tokens saved: {total_tokens_saved:,}")
            print(f"  Total chunks created: {len(chunk_definitions)}\n")
            
        return chunk_definitions

    def _chunk_document(self, document_text: str) -> list[str]:
        """Chunk document text into overlapping windows while preserving structures."""

        blocks = self._split_into_blocks(document_text)
        if not blocks:
            return []

        chunks: list[str] = []
        current_blocks: list[str] = []
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

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into blocks, keeping code fences intact."""

        blocks: list[str] = []
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
    def _split_paragraph_blocks(text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
        return [paragraph for paragraph in paragraphs if paragraph]

    def _collect_overlap_blocks(self, blocks: Sequence[str]) -> list[str]:
        """Collect blocks from the end until the overlap token budget is met."""

        overlap_blocks: list[str] = []
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

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches."""
        if not texts:
            return []
        
        embedding_model = lms.embedding_model(Settings.PAPER_INDEXING_EMBEDDING_MODEL)
        batch_size = Settings.PAPER_EMBEDDING_BATCH_SIZE
        all_embeddings: list[list[float]] = []
        
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



    def save_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Save embeddings to JSON file."""
        try:
            path_obj = Path(self.EMBEDDINGS_FILE)
            save_json(embeddings, path_obj.name, str(path_obj.parent))
            print(f"Saved {len(embeddings)} embeddings to {self.EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self) -> Optional[dict[str, list[float]]]:
        """Load embeddings from JSON file if it exists."""
        path_obj = Path(self.EMBEDDINGS_FILE)
        if not path_obj.exists():
            return None

        try:
            embeddings = load_json(path_obj.name, str(path_obj.parent))
            # Validate structure - simple check if it looks like a dict
            if isinstance(embeddings, dict):
                 return embeddings
            else:
                 print(f"Warning: Embeddings file format mismatch (expected dict, got {type(embeddings)}). Ignoring.")
                 return None
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
