import os
import re
from typing import List, Optional
from dataclasses import dataclass
import pymupdf4llm
from phases.paper_search.paper import Paper

@dataclass
class MarkdownParseResult:
    pdf_name: str
    markdown_path: str
    markdown_text: str
    pages: list[dict]
    image_dir: str | None
    page_count: int
    pages_fixed_with_marker: list[int]


# Idea: use LLM instead of Marker for math correction and/or table correction
class PDFConverter:
    """Convert PDFs with pymupdf4llm. Can use Marker to fix pages with broken math."""

    def __init__(self, fix_math=False, extract_media=True):
        self.fix_math = fix_math
        self.extract_media = extract_media
        
        # Lazy load Marker models only if needed
        self._marker_converter = None

    def _get_marker_converter(self):
        """Lazy load Marker converter (only loads if math fixing is needed)."""
        # Lazy import marker only when needed
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        if self._marker_converter is None:
            print("Loading Marker models for math correction...")
            model_dict = create_model_dict()
            self._marker_converter = PdfConverter(artifact_dict=model_dict)
        return self._marker_converter

    def _detect_broken_math(self, text: str) -> bool:
        """Detect if a page likely has broken mathematical notation."""
        indicators = [
            r'�',  # Replacement character
            r'_\[.\]_',  # Broken subscripts like _[l]_
            r'\(_\s+\*_',  # Broken function notation
            r'∑\s*$',  # Summation at end of line
            r'∫\s*$',  # Integral at end of line
            r'_\s+_\s+_',  # Multiple isolated underscores
            r'\|\s*\|',  # Broken absolute value bars
            r'\[\]\s*\[\]',  # Multiple empty brackets
        ]
        
        issue_count = sum(1 for pattern in indicators if re.search(pattern, text))
        return issue_count >= 2


    def _extract_page_with_marker(self, pdf_path: str, page_num: int, 
                                base_name: str, image_dir: str | None) -> str:
        """Use Marker to extract a single page with proper math formatting."""

        import pymupdf
        import tempfile
        
        tmp_path = None
        
        try:
            # Create a temporary PDF with just this page
            with pymupdf.open(pdf_path) as doc:
                with pymupdf.open() as single_page_doc:
                    single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    
                    # Save to temp file (don't auto-delete on Windows)
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Save the single page PDF
                    single_page_doc.save(tmp_path)
            
            # Lazy import marker output function
            from marker.output import text_from_rendered
            
            converter = self._get_marker_converter()
            rendered = converter(tmp_path)
            markdown_text, _, images = text_from_rendered(rendered)
            
            # Only save images if extract_media is enabled
            if self.extract_media and image_dir:
                for marker_img_name, img in images.items():
                    new_name = f"{base_name}-marker-p{page_num+1}-{marker_img_name}"
                    new_path = os.path.join(image_dir, new_name)
                    img.save(new_path)
                    markdown_text = markdown_text.replace(
                        f"![]({marker_img_name})",
                        f"![](images/{new_name})"
                    )
            else:
                # Remove image references if not extracting
                markdown_text = re.sub(r'!\[\]\([^)]+\)', '', markdown_text)
            
            return markdown_text
            
        finally:
            # Clean up temp file (with retry on Windows)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass


    def convert_to_markdown(self, file_path: str) -> MarkdownParseResult:
        """Convert PDF to Markdown, using Marker for math-heavy pages.
        
        Expects PDF structure: output/literature/PAPER_ID/PAPER_ID.pdf
        Output goes to: output/literature/PAPER_ID/markdown/
        """

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        pdf_dir = os.path.dirname(file_path)
        
        # Output to output/literature/PAPER_ID/markdown/
        pdf_output_dir = os.path.join(pdf_dir, "markdown")
        image_dir = os.path.join(pdf_output_dir, "images") if self.extract_media else None

        # Create directories
        os.makedirs(pdf_output_dir, exist_ok=True)
        if self.extract_media:
            os.makedirs(image_dir, exist_ok=True)

        markdown_pages = pymupdf4llm.to_markdown(
            file_path,
            detect_bg_color=True,
            ignore_alpha=False,
            hdr_info=None,
            write_images=self.extract_media,
            embed_images=False,
            ignore_images=not self.extract_media,  # True when extract_media=False
            ignore_graphics=not self.extract_media,  # True when extract_media=False
            image_size_limit=0.05,
            dpi=150,
            image_path=image_dir if self.extract_media else "",
            image_format="png",
            force_text=not self.extract_media,  # True when extract_media=False
            margins=0,
            page_chunks=True,
            page_separators=False,
            table_strategy="lines_strict" if self.extract_media else None,
            graphics_limit=5000,
            ignore_code=False,
            extract_words=False,
            show_progress=True,
            use_glyphs=False,
        )

        # Check for broken math and fix with Marker if enabled
        pages_fixed = []
        if self.fix_math:
            print("Checking for pages with broken math...")
            for i, page in enumerate(markdown_pages):
                if self._detect_broken_math(page["text"]):
                    print(f"Page {i+1} has broken math - fixing with Marker...")
                    pages_fixed.append(i + 1)
                    
                    try:
                        corrected_text = self._extract_page_with_marker(file_path, i, base_name, image_dir)
                        page["text"] = corrected_text
                    except Exception as e:
                        print(f"Failed to fix page {i+1}: {e}")

        # Merge all pages
        markdown_text = "\n\n".join(page["text"] for page in markdown_pages)
        
        # Adjust image paths if extracting images
        if self.extract_media:
            markdown_text = markdown_text.replace(
                f"literature/{base_name}/markdown/images/", "images/"
            )

        # Save merged markdown
        merged_path = os.path.join(pdf_output_dir, f"{base_name}.md")
        self._save_markdown(markdown_text, merged_path)

        return MarkdownParseResult(
            pdf_name=base_name,
            markdown_path=merged_path,
            markdown_text=markdown_text,
            pages=markdown_pages,
            image_dir=image_dir,
            page_count=len(markdown_pages),
            pages_fixed_with_marker=pages_fixed
        )

    def _save_markdown(self, markdown_text: str, output_path: str) -> None:
        """Save markdown text to a file with UTF-8 encoding."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
    
    
    def convert_all_papers(self, papers: list[Paper], base_folder: str = "output/literature/") -> list[Paper]:
        """Convert PDFs to markdown and update Paper objects with markdown text.
        
        Handles both:
        - Papers with pdf_url (downloaded papers) → look in output/literature/{id}/
        - Papers with pdf_path (user papers) → use pdf_path directly
        
        Args:
            papers: List of Paper objects to update
            base_folder: Base folder containing paper PDFs
            
        Returns:
            List of Paper objects with markdown_text field populated
        """
        print(f"\nConverting {len(papers)} papers to markdown...")
        
        for i, paper in enumerate(papers, 1):
            pdf_path = None
            
            # Check pdf_path first (user-provided papers)
            if paper.pdf_path:
                # pdf_path is relative to project root
                if os.path.isabs(paper.pdf_path):
                    pdf_path = paper.pdf_path
                else:
                    pdf_path = os.path.join(os.getcwd(), paper.pdf_path)
                    # Normalize path
                    pdf_path = os.path.normpath(pdf_path)
            
            # Fallback to pdf_url location (downloaded papers)
            if not pdf_path or not os.path.exists(pdf_path):
                # Use paper.id (Semantic Scholar ID) - same as PDF downloader
                # Sanitize ID for filesystem (matching PDFDownloader logic)
                safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
                pdf_path = os.path.join(base_folder, safe_id, f"{safe_id}.pdf")
            
            if os.path.exists(pdf_path):
                print(f"[{i}/{len(papers)}] {paper.title[:80]}...")
                try:
                    result = self.convert_to_markdown(pdf_path)
                    paper.markdown_text = result.markdown_text
                    
                    # If this was a user paper using original pdf_path, update it to point to copied location
                    # (if it's not already in output/literature/)
                    if paper.user_provided and paper.pdf_path:
                        current_pdf_path = os.path.normpath(os.path.join(os.getcwd(), paper.pdf_path))
                        if not current_pdf_path.startswith(os.path.normpath(os.path.join(os.getcwd(), base_folder))):
                            # Update pdf_path to point to copied location
                            paper.pdf_path = os.path.relpath(pdf_path, os.getcwd())
                except Exception as e:
                    print(f"FAILED: {e}\n")
                    paper.markdown_text = None
            else:
                print(f"[{i}/{len(papers)}] SKIPPED: PDF not found for {paper.title[:80]}...")
                if paper.pdf_path:
                    print(f"  Expected at: {paper.pdf_path}")
                paper.markdown_text = None
        
        return papers
