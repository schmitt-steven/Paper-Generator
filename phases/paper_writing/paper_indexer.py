from __future__ import annotations
import re
import time
from typing import List, Sequence, Iterable, Optional
from pathlib import Path
import lmstudio as lms
import numpy as np  # Added for array handling if needed, though list is used for storage
from settings import Settings
from utils.file_utils import save_json, load_json, preprocess_markdown
from phases.literature_search.paper import Paper
from phases.paper_writing.data_models import PaperChunk



class PaperIndexer:
    """Builds an indexed corpus of paper chunks using whole-document chunking."""
    
    EMBEDDINGS_FILE = "output/paper_embeddings.json"

    CODE_BLOCK_PATTERN = re.compile(r"```.+?```", re.DOTALL)
    
    # Patterns to identify abstract and conclusion sections
    # These are excluded from embeddings since they're always in the prompt
    ABSTRACT_PATTERN = re.compile(
        r'^\s*(?:#+\s*)?(?:\*\*)?ABSTRACT(?:\*\*)?(?:[:\.]|\s+|$)',
        re.IGNORECASE | re.MULTILINE
    )
    CONCLUSION_PATTERN = re.compile(
        r'^\s*(?:#+\s*)?(?:\*\*)?(?:(?:[IVXivx]+\.?\s*)|(?:\d+\.?\s*))?(?:\*\*)?\s*(?:\*\*)?(?:DISCUSSION\s+AND\s+)?(?:CONCLUSIONS?|CONCLUDING\s+REMARKS?|CLOSING\s+REMARKS?)(?:[:\.\s][^\n]*)?(?:\*\*)?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    # Pattern for next section after abstract (to know where abstract ends)
    NEXT_SECTION_PATTERN = re.compile(
        r'^\s*(?:#+\s*)?(?:\*\*)?(?:(?:[IVXivx]+\.?\s+)|(?:\d+\.?\s+))?(?:\*\*)?\s*(?:\*\*)?(?:INTRODUCTION|BACKGROUND|RELATED|METHODS?|PRELIMINAR)(?:[:\.\s][^\n]*)?(?:\*\*)?\s*$',
        re.IGNORECASE | re.MULTILINE
    )

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

        papers_with_content = [p for p in papers if p.markdown_text and p.markdown_text.strip()]
        
        print(f"\n{'='*80}")
        print(f"INDEXING {len(papers_with_content)}/{len(papers)} PAPERS (skipping {len(papers) - len(papers_with_content)} without text)")
        print(f"{'='*80}\n")
        
        # 1. Generate all chunk definitions first
        chunk_definitions = self._create_chunk_definitions(papers)
        if not chunk_definitions:
            return []
            
        # 2. Load existing embeddings (dict)
        existing_embeddings: dict[str, list[float]] = {}
        # 2. Start fresh (do not load existing embeddings)
        existing_embeddings: dict[str, list[float]] = {}
        
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

            # Strip abstract and conclusion before chunking
            # (theyre already included in the prompts by default)
            markdown = self._strip_abstract_conclusion(paper.markdown_text)
                
            chunks = self._chunk_document(markdown)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = self._build_chunk_id(paper.id, chunk_idx)
                chunk_definitions.append((paper, chunk_idx, chunk_id, chunk_text))

        if not chunk_definitions:
            return []
        
        print(f"INDEXING SUMMARY:")
        print(f"  Papers processed: {len(papers)}")
        print(f"  Total chunks created: {len(chunk_definitions)}\n")
            
        return chunk_definitions

    @classmethod
    def _strip_abstract_conclusion(cls, text: str) -> str:
        """Remove Abstract and Conclusion sections from text for embeddings."""
        # 1. Find Abstract
        abstract_match = cls.ABSTRACT_PATTERN.search(text)
        intro_match = cls.NEXT_SECTION_PATTERN.search(text)
        
        # 2. Find Conclusion
        conclusion_match = cls.CONCLUSION_PATTERN.search(text)
        
        # Build new text without these sections
        # We process conclusion first if its at the end, then abstract
        
        # Calculate cut ranges
        cuts = [] # list of (start, end) to remove
        
        if abstract_match:
            # We want to remove everything before the abstract too (Title, Authors, etc.)
            # So start is 0
            start = 0
            
            # End is start of next section (Intro) OR some heuristic length if intro not found
            if intro_match and intro_match.start() > abstract_match.start():
                end = intro_match.start()
            else:
                # Fallback
                remaining = text[abstract_match.end():]
                # Look for next Markdown header (#) OR bold header (**)
                next_heading = re.search(r'^\s*(?:#|\*\*)', remaining, re.MULTILINE)
                if next_heading:
                     end = abstract_match.end() + next_heading.start()
                else:
                     end = abstract_match.start() # Minimal safe cut (Title + Authors)
            
            if end > start:
                cuts.append((start, end))
                
        if conclusion_match:
            start = conclusion_match.start()
            # removes everything from conclusion onwards
            end = len(text)
            
            cuts.append((start, end))
            
        # Apply cuts in reverse order to keep indices valid
        cuts.sort(key=lambda x: x[0], reverse=True)
        
        current_text = text
        for start, end in cuts:
             # Leave a marker so we know something was removed (optional, but good for debugging)
             # current_text = current_text[:start] + "\n\n[SECTION REDACTED_FOR_EMBEDDING]\n\n" + current_text[end:]
             current_text = current_text[:start] + current_text[end:]
             
        return current_text

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
        """
        Estimate tokens via simple word count division. 
        Note: While lms.embedding_model().tokenize() provides exact counts, this approximation is used because the chunking algo evaluates token counts thousands of times per paper. 
        Exact tokenization over HTTP adds unnecessary overhead and precise chunk sizes dont matter enough anyways to justify the performance loss.
        """
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
        batch_size = 32
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
