import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel
import pymupdf
from textwrap import dedent

from phases.paper_search.paper import Paper
from phases.paper_search.semantic_scholar_api import SemanticScholarAPI
from utils.lazy_model_loader import LazyModelMixin
from phases.latex_generation.bibliography import generate_bibtex_entry


class PaperTitle(BaseModel):
    """Structured output for title extraction"""
    title: str


# Regex pattern for arXiv IDs: YYMM.NNNNN or YYMM.NNNNNvN (to strip version suffix)
ARXIV_ID_PATTERN = re.compile(r'^(\d{4}\.\d{4,5})(v\d+)?$')


class UserPaperLoader(LazyModelMixin):
    """Load and process user-provided PDF papers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def _get_id_variants_from_filename(self, filename: str) -> List[str]:
        """Generate potential paper ID variants from filename to try on S2."""
        stem = Path(filename).stem
        variants = [stem]
        
        # If it looks like an arXiv ID, also try with ARXIV: prefix
        match = ARXIV_ID_PATTERN.match(stem)
        if match:
            arxiv_base = match.group(1)
            variants.append(f"ARXIV:{arxiv_base}")
        
        return variants
    
    def _search_paper_on_s2(self, pdf_path: Path, s2_api: SemanticScholarAPI) -> Tuple[Optional[Paper], str]:
        """Search for paper on Semantic Scholar."""
        # Try filename-based ID variants first
        id_variants = self._get_id_variants_from_filename(pdf_path.name)
        
        for variant in id_variants:
            print(f"    Trying ID: {variant}")
            paper = s2_api.get_paper_by_id(variant)
            if paper:
                return paper, f"id_lookup:{variant}"
        
        # Fall back to title extraction and search
        print(f"    ID lookup failed, extracting title...")
        title = self._extract_title_from_pdf(str(pdf_path))
        print(f"    Extracted title: {title[:80]}...")
        
        # For user-provided papers, we don't care about open access status since we already have the PDF
        # We just need the metadata, so search without open access filter
        results = s2_api.search_papers(title, max_results=1, open_access_only=False)
        if results:
            return results[0], "title_search"
        
        return None, "not_found"
    
    def _extract_title_from_pdf(self, pdf_path: str) -> str:
        """Extract title from PDF using LLM."""
        try:
            doc = pymupdf.open(pdf_path)
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            first_page = doc[0]
            first_page_text = first_page.get_text()
            doc.close()
            
            if not first_page_text or not first_page_text.strip():
                raise ValueError("First page has no text")
            
            prompt = dedent(f"""\
                Extract the paper title from the following text from the first page of a research paper:

                {first_page_text[:1000]}

                Return ONLY the exact title of the paper.""")
            
            result = self.model.respond(
                history=prompt,
                response_format=PaperTitle,
                config={'temperature': 0.1}
            )
            
            return result.parsed['title'].strip()
            
        except Exception as e:
            print(f"Error extracting title from {pdf_path}: {e}")
            return Path(pdf_path).stem
    
    def load_user_paper(self, pdf_path: Path, s2_api: Optional[SemanticScholarAPI] = None) -> Optional[Paper]:
        """
        Load and process a single user-provided PDF paper.
        Uses filename-based ID: user_{filename_stem}
        Copies PDF to output/literature/{paper_id}/
        
        Args:
            pdf_path: Path to the PDF file
            s2_api: SemanticScholarAPI instance (optional)
            
        Returns:
            Paper object or None if processing failed
        """
        if s2_api is None:
            s2_api = SemanticScholarAPI()
        
        # Paper ID is always based on filename
        paper_id = f"user_{pdf_path.stem}"
        
        print(f"  Processing {pdf_path.name} -> {paper_id}")
        
        try:
            # Search for paper metadata on Semantic Scholar
            s2_paper, search_method = self._search_paper_on_s2(pdf_path, s2_api)
            
            # Copy PDF to output/literature/{paper_id}/
            output_folder = Path("output/literature") / paper_id
            output_folder.mkdir(parents=True, exist_ok=True)
            dest_path = output_folder / f"{paper_id}.pdf"
            shutil.copy2(pdf_path, dest_path)
            
            if s2_paper:
                print(f"    Found paper on Semantic Scholar: {s2_paper.id}")
                
                paper = Paper(
                    id=paper_id,
                    title=s2_paper.title,
                    published=s2_paper.published,
                    authors=s2_paper.authors,
                    summary=s2_paper.summary,
                    pdf_url=s2_paper.pdf_url,
                    doi=s2_paper.doi,
                    fields_of_study=s2_paper.fields_of_study,
                    venue=s2_paper.venue,
                    citation_count=s2_paper.citation_count,
                    bibtex=s2_paper.bibtex,
                    is_open_access=s2_paper.is_open_access,
                    user_provided=True,
                    pdf_path=str(dest_path.resolve().relative_to(Path.cwd()))
                )
            else:
                print(f"    Paper not found on Semantic Scholar")
                title = self._extract_title_from_pdf(str(pdf_path))
                
                paper = Paper(
                    id=paper_id,
                    title=title,
                    published=str(datetime.now().year),
                    authors=[],
                    summary="",
                    pdf_url=None,
                    doi=None,
                    fields_of_study=[],
                    venue=None,
                    citation_count=None,
                    bibtex=None,
                    is_open_access=False,
                    user_provided=True,
                    pdf_path=str(dest_path.resolve().relative_to(Path.cwd()))
                )
                paper.bibtex = generate_bibtex_entry(paper)
            
            print(f"    Done")
            return paper
            
        except Exception as e:
            print(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_user_papers(
        self, 
        folder_path: str = "user_files/papers/",
        s2_api: Optional[SemanticScholarAPI] = None
    ) -> List[Paper]:
        """
        Load and process user-provided PDF papers from a folder.
        
        Args:
            folder_path: Path to folder containing user PDFs
            s2_api: SemanticScholarAPI instance (optional)
            
        Returns:
            List of Paper objects
        """
        if s2_api is None:
            s2_api = SemanticScholarAPI()
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"User papers folder not found: {folder_path}")
            return []
        
        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return []
        
        print(f"[UserPaperLoader] Loading {len(pdf_files)} user-provided PDF(s)...")
        
        papers = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"  [{i}/{len(pdf_files)}]", end="")
            paper = self.load_user_paper(pdf_path, s2_api)
            if paper:
                papers.append(paper)
        
        print(f"Loaded {len(papers)} user-provided paper(s)\n")
        return papers
