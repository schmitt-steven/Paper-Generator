import os
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import pymupdf  # PyMuPDF (fitz)

from phases.paper_search.paper import Paper
from phases.paper_search.semantic_scholar_api import SemanticScholarAPI
from utils.lazy_model_loader import LazyModelMixin
from phases.latex_generation.bibliography import generate_bibtex_entry


class PaperTitle(BaseModel):
    """Structured output for title extraction"""
    title: str


class UserPaperLoader(LazyModelMixin):
    """Load and process user-provided PDF papers"""
    
    def __init__(self, model_name: str):
        """
        Initialize UserPaperLoader with a language model.
        
        Args:
            model_name: Name of the LLM model to use for title extraction
        """
        self.model_name = model_name
        self._model = None  # Lazy-loaded via LazyModelMixin
    
    def _extract_title_from_pdf(self, pdf_path: str) -> str:
        """
        Extract title from PDF using LLM.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted title string
        """
        # Extract first page text using PyMuPDF
        try:
            doc = pymupdf.open(pdf_path)
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Get first page text
            first_page = doc[0]
            first_page_text = first_page.get_text()
            doc.close()
            
            if not first_page_text or not first_page_text.strip():
                raise ValueError("First page has no text")
            
            # Use LLM with structured output to extract title
            prompt = f"""Extract the paper title from the following text from the first page of a research paper:

{first_page_text[:2000]}  # Limit to first 2000 chars to avoid token limits

Return only the title of the paper."""
            
            result = self.model.respond(
                prompt,
                response_format=PaperTitle,
                config={
                    'temperature': 0.1,
                }
            )
            
            return result.parsed['title'].strip()
            
        except Exception as e:
            print(f"Error extracting title from {pdf_path}: {e}")
            # Fallback: use filename without extension
            return Path(pdf_path).stem
    
    def _generate_paper_id(self, pdf_path: str) -> str:
        """
        Generate a deterministic paper ID from file path.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Paper ID in format: user_{hash}
        """
        # Use absolute path for deterministic hashing
        abs_path = os.path.abspath(pdf_path)
        # Generate MD5 hash
        hash_obj = hashlib.md5(abs_path.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:12]  # First 12 chars
        return f"user_{hash_hex}"
    
    def _copy_pdf_to_output(self, source_path: str, paper_id: str, base_folder: str = "output/literature/") -> str:
        """
        Copy PDF to output/literature/{paper_id}/ folder.
        
        Args:
            source_path: Source PDF path
            paper_id: Paper ID (used for folder name)
            base_folder: Base folder for papers
            
        Returns:
            Destination PDF path
        """
        # Sanitize paper ID for filesystem
        safe_id = "".join([c for c in paper_id if c.isalnum() or c in ('-', '_', '.')])
        
        # Create folder for this paper
        paper_folder = os.path.join(base_folder, safe_id)
        os.makedirs(paper_folder, exist_ok=True)
        
        # Copy PDF to paper's folder
        dest_path = os.path.join(paper_folder, f"{safe_id}.pdf")
        
        import shutil
        shutil.copy2(source_path, dest_path)
        
        return dest_path
    
    def load_user_papers(
        self, 
        folder_path: str = "user_files/papers/",
        s2_api: Optional[SemanticScholarAPI] = None
    ) -> List[Paper]:
        """
        Load and process user-provided PDF papers.
        
        Args:
            folder_path: Path to folder containing user PDFs
            s2_api: SemanticScholarAPI instance (optional, will create if not provided)
            
        Returns:
            List of Paper objects
        """
        if s2_api is None:
            s2_api = SemanticScholarAPI()
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"User papers folder not found: {folder_path}")
            return []
        
        # Find all PDF files
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return []
        
        print(f"\nLoading {len(pdf_files)} user-provided PDF(s)...")
        
        papers = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")
            
            try:
                # Extract title using LLM
                title = self._extract_title_from_pdf(str(pdf_path))
                print(f"  Extracted title: {title[:80]}...")
                
                # Always generate a user_* ID for user-provided papers (for folder structure)
                paper_id = self._generate_paper_id(str(pdf_path))
                
                # Search Semantic Scholar with extracted title
                s2_results = s2_api.search_papers(title, max_results=1)
                
                if s2_results:
                    # Use the found Paper object from Semantic Scholar for metadata, but use user_* ID
                    s2_paper = s2_results[0]
                    print(f"  Found in Semantic Scholar: {s2_paper.id} (using user ID: {paper_id})")
                    
                    # Create Paper object with user_* ID but Semantic Scholar metadata
                    paper = Paper(
                        id=paper_id,  # Always use user_* ID for user-provided papers
                        title=s2_paper.title,
                        published=s2_paper.published,
                        authors=s2_paper.authors,
                        summary=s2_paper.summary,
                        pdf_url=None,  # We have local file
                        doi=s2_paper.doi,
                        fields_of_study=s2_paper.fields_of_study,
                        venue=s2_paper.venue,
                        citation_count=s2_paper.citation_count,
                        bibtex=s2_paper.bibtex,
                        is_open_access=s2_paper.is_open_access,
                        user_provided=True,
                        pdf_path=str(pdf_path.relative_to(Path.cwd()))  # Relative path
                    )
                    
                else:
                    # Create minimal Paper object with user_* ID
                    print(f"  Not found in Semantic Scholar, creating minimal paper: {paper_id}")
                    
                    current_year = str(datetime.now().year)
                    
                    paper = Paper(
                        id=paper_id,
                        title=title,
                        published=current_year,
                        authors=[],
                        summary="",
                        pdf_url=None,
                        doi=None,
                        fields_of_study=[],
                        venue=None,
                        citation_count=0,
                        bibtex=None,
                        is_open_access=False,
                        user_provided=True,
                        pdf_path=str(pdf_path.relative_to(Path.cwd()))  # Relative path
                    )
                    
                    # Generate minimal BibTeX entry
                    paper.bibtex = generate_bibtex_entry(paper)
                
                # Copy PDF to output/literature/{paper_id}/ (always uses user_* ID)
                dest_path = self._copy_pdf_to_output(str(pdf_path), paper.id)
                # Update pdf_path to point to copied location
                paper.pdf_path = str(Path(dest_path).relative_to(Path.cwd()))
                
                papers.append(paper)
                print(f"  ✓ Successfully processed\n")
                
            except Exception as e:
                print(f"  ✗ Failed to process {pdf_path.name}: {e}\n")
                import traceback
                traceback.print_exc()
        
        print(f"Loaded {len(papers)} user-provided paper(s)\n")
        return papers

