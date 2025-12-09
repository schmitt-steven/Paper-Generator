import os
import time
import ssl
import urllib.request as libreq
from typing import List, Tuple
from phases.paper_search.paper import Paper

class PDFDownloader:
    
    @staticmethod
    def download_pdf(pdf_url: str, filename: str):
        """
        Download a single PDF.
        
        Args:
            pdf_url: URL to the PDF
            filename: Path where PDF should be saved
        """
        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Add headers to mimic a browser to avoid some 403s
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        req = libreq.Request(pdf_url, headers=headers)
        
        with libreq.urlopen(req, context=ssl_context) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())

    @staticmethod
    def download_papers_as_pdfs(
        papers: List[Paper], 
        base_folder: str = "output/literature/"
    ) -> Tuple[int, int]:
        """
        Download selected papers as PDFs to specified folder.
        Creates a separate folder for each paper (named by paper ID) containing the PDF.
        Includes 1-second rate limiting.
        Skips papers that already have a PDF downloaded.
        Updates paper.pdf_path after successful download.
        
        Args:
            papers: List of Paper objects to download
            base_folder: Base folder for all papers (will be created if doesn't exist)
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Create base folder if it doesn't exist
        os.makedirs(base_folder, exist_ok=True)
        
        print(f"Downloading {len(papers)} papers to '{base_folder}'...")
        successful = 0
        failed = 0
        skipped = 0
        
        for i, paper in enumerate(papers, 1):
            # Use PDF URL from paper (provided by Semantic Scholar)
            pdf_url = paper.pdf_url
            
            if pdf_url:
                # Sanitize paper ID for filesystem
                safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
                
                # Create folder for this paper
                paper_folder = os.path.join(base_folder, safe_id)
                os.makedirs(paper_folder, exist_ok=True)
                
                # Save PDF in paper's folder
                filename = os.path.join(paper_folder, f"{safe_id}.pdf")
                
                # Skip if PDF already exists
                if os.path.exists(filename):
                    print(f"  [{i}/{len(papers)}] EXISTS: {paper.title[:50]}...")
                    skipped += 1
                    # Update pdf_path even if already exists
                    paper.pdf_path = os.path.relpath(filename, os.getcwd())
                    continue
                
                try:
                    PDFDownloader.download_pdf(pdf_url, filename)
                    print(f"  [{i}/{len(papers)}] {paper.title[:50]}...")
                    successful += 1
                    # Update pdf_path after successful download
                    paper.pdf_path = os.path.relpath(filename, os.getcwd())
                except Exception as e:
                    print(f"  [{i}/{len(papers)}] FAILED: {paper.title[:50]}... ({e})")
                    # Log DOI and title for closed access papers
                    if "403" in str(e) or "Forbidden" in str(e):
                        print(f"      -> Closed access paper")
                        if paper.doi:
                            print(f"      -> DOI: {paper.doi}")
                        print(f"      -> Title: {paper.title}")
                    failed += 1
            else:
                print(f"  [{i}/{len(papers)}] SKIPPED - no PDF URL")
                # Log DOI and title for papers without PDF access
                if paper.doi:
                    print(f"      -> DOI: {paper.doi}")
                print(f"      -> Title: {paper.title}")
                failed += 1
            
            # Add 1-second delay between downloads (except after last one)
            if i < len(papers):
                time.sleep(1)
        
        print(f"Download complete: {successful} downloaded, {skipped} already existed, {failed} failed\n")
        return successful, failed
