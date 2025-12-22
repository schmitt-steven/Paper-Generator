import tkinter as tk
from tkinter import ttk, filedialog
import webbrowser
import threading
import shutil
import re
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional

from ..base_frame import BaseFrame, ProgressPopup, create_gray_button
from phases.paper_search.paper import Paper
from phases.paper_search.user_paper_loader import UserPaperLoader
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper_ranking import PaperRanker
from phases.paper_search.paper_filtering import PaperFilter
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from phases.context_analysis.user_requirements import UserRequirements

from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from utils.pdf_downloader import PDFDownloader
from utils.pdf_converter_pymupdf_marker import PDFConverter
from settings import Settings


HYPOTHESES_FILE = Path("output/hypothesis.md")
PAPERS_FILE = Path("output/papers.json")

class PaperSelectionScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.user_papers: list[Paper] = []
        self.searched_papers: list[Paper] = []
        
        # Widget references
        self.user_paper_widgets: dict[str, ttk.Frame] = {}
        self.searched_paper_widgets: dict[str, ttk.Frame] = {}
        
        # Buttons
        self.upload_btn: ttk.Button
        self.search_btn: ttk.Button
        
        # Count labels
        self.user_count_label: ttk.Label
        self.searched_count_label: ttk.Label
        
        # Paper list containers
        self.user_papers_list: ttk.Frame
        self.searched_papers_list: ttk.Frame
        
        # Loading state
        self.is_uploading = False
        self.is_searching = False
        
        # Track if papers have been loaded
        self._papers_loaded = False
        
        # Track original paper IDs to detect new papers
        self._original_paper_ids: set = set()
        
        # Dynamic button text based on whether output file exists
        next_text = "Continue" if HYPOTHESES_FILE.exists() else "Generate Hypothesis"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Paper Selection",
            next_text=next_text
        )

    def create_content(self):
        self._create_user_papers_section()
        self._create_searched_papers_section()

    def on_show(self):
        """Called when screen is shown - load papers from file if not already loaded."""
        if not self._papers_loaded:
            self._load_papers_from_file()
            self._papers_loaded = True

    def _load_papers_from_file(self):
        """Load papers from papers.json and split into user/searched lists."""
        if not PAPERS_FILE.exists():
            return
        
        try:
            all_papers = LiteratureSearch.load_papers(str(PAPERS_FILE))
            
            # Split into user-provided and searched papers
            self.user_papers = [p for p in all_papers if p.user_provided]
            self.searched_papers = [p for p in all_papers if not p.user_provided]
            
            # Track original IDs to detect new papers later
            self._original_paper_ids = {p.id for p in all_papers}
            
            print(f"[Papers] Loaded {len(self.user_papers)} user papers, {len(self.searched_papers)} searched papers")
            
            # Refresh the UI
            self._refresh_user_papers_list()
            self._refresh_searched_papers_list()
            
        except Exception as e:
            print(f"Error loading papers from {PAPERS_FILE}: {e}")
            import traceback
            traceback.print_exc()

    def _create_section_container(self, parent, title: str, count: int, 
                                   button_text: str, button_command: Callable) -> tuple:
        section_frame = ttk.Frame(parent, style="Card.TFrame", padding=1)
        section_frame.pack(fill="x", pady=10)
        
        # Header row with styled background
        header_frame = ttk.Frame(section_frame, style="CardHeader.TFrame", padding=(10, 6))
        header_frame.pack(fill="x")
        
        left_header = ttk.Frame(header_frame, style="CardHeader.TFrame")
        left_header.pack(side="left")
        
        # Use tk.Label for reliable background color
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        tk.Label(
            left_header, 
            text=title, 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        count_label = tk.Label(
            left_header, 
            text=str(count), 
            font=self.controller.fonts.sub_header_font, 
            fg="#666666",
            bg=header_bg
        )
        count_label.pack(side="left", padx=(10, 0))
        
        style = ttk.Style()
        style.configure("Section.TButton", font=self.controller.fonts.default_font)
        
        action_btn = ttk.Button(header_frame, text=button_text, command=button_command, style="Section.TButton")
        action_btn.pack(side="right")
        
        # Separator
        ttk.Separator(section_frame, orient="horizontal").pack(fill="x")
        
        # Papers list container
        papers_list = ttk.Frame(section_frame, padding=10)
        papers_list.pack(fill="x")
        
        return section_frame, count_label, action_btn, papers_list

    def _create_user_papers_section(self):
        _, self.user_count_label, self.upload_btn, self.user_papers_list = \
            self._create_section_container(
                self.scrollable_frame, "Your Papers", 0, "Upload", self._on_upload_click
            )
        self._show_empty_state(self.user_papers_list, "No papers uploaded yet")

    def _create_searched_papers_section(self):
        _, self.searched_count_label, self.search_btn, self.searched_papers_list = \
            self._create_section_container(
                self.scrollable_frame, "Found Papers", 0, "Auto Search", self._on_auto_search_click
            )
        self._show_empty_state(self.searched_papers_list, "Click 'Auto Search' to find related papers")

    def _show_empty_state(self, container: ttk.Frame, message: str):
        ttk.Label(container, text=message, font=self.controller.fonts.default_font, foreground="gray").pack(pady=20)

    def _create_paper_entry(self, parent: ttk.Frame, paper: Paper, 
                            on_remove: Callable, is_user_paper: bool) -> ttk.Frame:
        entry_frame = ttk.Frame(parent, padding="8")
        entry_frame.pack(fill="x")
        
        content_row = ttk.Frame(entry_frame)
        content_row.pack(fill="x")
        
        # Button container (right-aligned) - Pack FIRST to reserve space
        btn_container = ttk.Frame(content_row)
        btn_container.pack(side="right", padx=(10, 0))

        # Upload button for closed access papers
        if not is_user_paper and not paper.is_open_access:
             # Check if we already have a PDF for this paper
             has_local_pdf = False
             if paper.pdf_path:
                 has_local_pdf = True
             elif paper.id:
                 safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
                 expected_pdf_path = Path("output/literature") / safe_id / f"{safe_id}.pdf"
                 if expected_pdf_path.exists():
                     has_local_pdf = True
             
             if not has_local_pdf:
                 # Use upload icon with theme-aware coloring
                 upload_btn = self.controller.icons.create_icon_label(
                     btn_container,
                     icon_name="upload",
                     command=lambda: self._on_upload_paper_pdf(paper)
                 )
                 upload_btn.pack(side="left", padx=(0, 10))
        
        # X button with theme-aware icon
        x_btn = self.controller.icons.create_icon_label(
            btn_container,
            icon_name="x",
            command=lambda: on_remove(paper.id)
        )
        x_btn.pack(side="left")

        # Content Frame (Title + Metadata) - Pack SECOND to fill remaining space
        content_frame = ttk.Frame(content_row)
        content_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        title_label = ttk.Label(content_frame, text=paper.title, font=self.controller.fonts.default_font)
        title_label.pack(anchor="w", fill="x")
        
        metadata_frame = ttk.Frame(content_frame)
        metadata_frame.pack(anchor="w", pady=(2, 0), fill="x")
        
        # 1. Status Tag (Colored)
        status_text, status_color = self._get_paper_status(paper)
        if status_text:
            status_label = ttk.Label(metadata_frame, text=status_text, font=self.controller.fonts.text_area_font, foreground=status_color)
            status_label.pack(side="left")
            
            # Separator if there is other metadata
            ttk.Label(metadata_frame, text="  \u00B7  ", font=self.controller.fonts.text_area_font, foreground="gray").pack(side="left")

        # 2. Bibliographic Metadata (Gray)
        metadata = self._format_paper_bibliographic_info(paper)
        metadata_label = ttk.Label(metadata_frame, text=metadata, font=self.controller.fonts.text_area_font, foreground="gray")
        metadata_label.pack(side="left")
        
        # Dynamic wraplength
        def update_wraplength(event):
            # Subtract padding/offsets if needed
            width = event.width
            if width > 10: 
                title_label.config(wraplength=width)
                metadata_label.config(wraplength=width)
                
        content_frame.bind("<Configure>", update_wraplength)
        
        for widget in [content_frame, title_label, metadata_label, metadata_frame]:
            widget.bind("<Button-1>", lambda e, p=paper: self._on_paper_click(p, is_user_paper))
            widget.configure(cursor="hand2")
        
        return entry_frame

    def _get_paper_status(self, paper: Paper) -> tuple[Optional[str], Optional[str]]:
        """Return status text and color if applicable."""
        if not paper.is_open_access and not paper.user_provided:
             # Check if we have a PDF for this paper
             has_local_pdf = False
             if paper.pdf_path:
                 has_local_pdf = True
             elif paper.id:
                 safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
                 expected_pdf_path = Path("output/literature") / safe_id / f"{safe_id}.pdf"
                 if expected_pdf_path.exists():
                     has_local_pdf = True
            
             if has_local_pdf:
                 return "PDF Uploaded", "green"
             else:
                 return "Closed Access", "red"
        
        return None, None

    def _format_paper_bibliographic_info(self, paper: Paper) -> str:
        parts = []
        
        if paper.authors:
            first_author = paper.authors[0]
            if ',' in first_author:
                last_name = first_author.split(',')[0].strip()
            else:
                name_parts = first_author.split()
                last_name = name_parts[-1] if name_parts else first_author
            parts.append(f"{last_name} et al." if len(paper.authors) > 1 else last_name)
        
        if paper.published:
            year_match = re.search(r'(\d{4})', paper.published)
            if year_match:
                parts.append(year_match.group(1))
        
        if paper.citation_count is not None:
            parts.append(f"{paper.citation_count:,} citations")
        
        return "  \u00B7  ".join(parts)

    def _on_paper_click(self, paper: Paper, is_user_paper: bool):
        # Check if pdf_path is set and file exists
        if paper.pdf_path:
            pdf_path = Path(paper.pdf_path)
            if not pdf_path.is_absolute():
                pdf_path = Path.cwd() / pdf_path
            if pdf_path.exists():
                webbrowser.open(f"file://{pdf_path.resolve()}")
                return
        
        # Check if downloaded PDF exists at expected location (for auto-searched papers)
        if not is_user_paper and paper.id:
            safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
            expected_pdf_path = Path("output/literature") / safe_id / f"{safe_id}.pdf"
            if expected_pdf_path.exists():
                webbrowser.open(f"file://{expected_pdf_path.resolve()}")
                return
        
        # Fallback to Semantic Scholar URL
        if paper.id:
            webbrowser.open(f"https://www.semanticscholar.org/paper/{paper.id}")
        elif paper.pdf_url:
            webbrowser.open(paper.pdf_url)

    def _on_upload_paper_pdf(self, paper: Paper):
        """Handle uploading a PDF for a specific closed-access paper."""
        file_path = filedialog.askopenfilename(
            title=f"Select PDF for '{paper.title[:30]}...'",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not file_path:
            return

        try:
            # 1. Determine destination path
            # Use safe ID for folder name
            safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
            output_folder = Path("output/literature") / safe_id
            output_folder.mkdir(parents=True, exist_ok=True)
            dest_path = output_folder / f"{safe_id}.pdf"
            
            # 2. Copy file
            shutil.copy2(file_path, dest_path)
            
            # 3. Update paper object
            paper.pdf_path = str(dest_path.resolve().relative_to(Path.cwd()))
            
            # 4. Save and refresh
            self._save_papers()
            self._refresh_searched_papers_list() # Re-render appropriately
            
            print(f"[Papers] Uploaded PDF for: {paper.title[:60]}")
            
        except Exception as e:
            print(f"Error uploading PDF for paper {paper.id}: {e}")
            import traceback
            traceback.print_exc()

    def _on_upload_click(self):
        if self.is_uploading:
            return
        
        file_paths = filedialog.askopenfilenames(
            title="Select PDF Papers",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not file_paths:
            return
        
        self._set_upload_loading(True)
        thread = threading.Thread(target=self._process_uploaded_files, args=(file_paths,))
        thread.daemon = True
        thread.start()

    def _process_uploaded_files(self, file_paths: tuple):
        """Process uploaded PDF files. Simplified: process directly, skip duplicates."""
        try:
            # Get existing paper IDs to skip duplicates
            existing_ids = {p.id for p in self.user_papers}
            
            loader = UserPaperLoader(model_name=Settings.LITERATURE_SEARCH_MODEL)
            new_papers = []
            
            for file_path in file_paths:
                pdf_path = Path(file_path)
                paper_id = f"user_{pdf_path.stem}"
                
                # Skip if already loaded
                if paper_id in existing_ids:
                    print(f"[Papers] Skipping duplicate: {pdf_path.name}")
                    continue
                
                # Process this paper
                paper = loader.load_user_paper(pdf_path)
                if paper:
                    new_papers.append(paper)
                    existing_ids.add(paper.id)
            
            self.after(0, lambda: self._on_upload_complete(new_papers))
            
        except Exception as e:
            print(f"Error processing uploaded files: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self._set_upload_loading(False))

    def _on_upload_complete(self, new_papers: list[Paper]):
        for paper in new_papers:
            self.user_papers.append(paper)
            print(f"[Papers] Added user paper: {paper.title[:60]}")
        
        self._refresh_user_papers_list()
        self._set_upload_loading(False)

    def _set_upload_loading(self, loading: bool):
        self.is_uploading = loading
        if loading:
            self.upload_btn.config(state="disabled", text="Processing")
        else:
            self.upload_btn.config(state="normal", text="Upload")

    def _on_auto_search_click(self):
        if self.is_searching:
            return
        
        self._set_search_loading(True)
        popup = ProgressPopup(self.controller, "Searching Papers")
        
        def task():
            try:
                # Step 1: Search
                self.after(0, lambda: popup.update_status("Building search queries"))
                paper_concept: PaperConcept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                literature_search = LiteratureSearch(model_name=Settings.LITERATURE_SEARCH_MODEL)
                search_queries = literature_search.build_search_queries(paper_concept)
                
                self.after(0, lambda: popup.update_status(f"Searching related papers with {len(search_queries)} queries"))
                papers = literature_search.search_papers(search_queries, max_results_per_query=15)
                
                # Filter out papers already in user papers
                user_paper_ids = {p.id for p in self.user_papers}
                searched_papers = [p for p in papers if p.id not in user_paper_ids]
                
                if not searched_papers:
                    self.after(0, lambda: self._on_search_complete([], popup))
                    return
                
                # Step 2: Rank papers
                self.after(0, lambda: popup.update_status("Ranking papers for relevance"))
                ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
                ranking_context = f"{paper_concept.description}\nOpen Research Questions:\n{paper_concept.open_questions}"
                ranked_papers = ranker.rank_papers(
                    papers=searched_papers,
                    context=ranking_context,
                    weights={'relevance': 0.7, 'citations': 0.2, 'recency': 0.1}
                )
                
                # Step 3: Filter papers for diverse selection
                self.after(0, lambda: popup.update_status("Filtering found papers"))
                filtered_papers = PaperFilter.filter_diverse(
                    ranked_papers,
                    n_cutting_edge=20,
                    n_hidden_gems=15,
                    n_classics=15,
                    n_well_rounded=10
                )
                
                # Step 4: Show results on screen
                self.after(0, lambda: self._on_search_complete(filtered_papers, popup))
                
            except Exception as e:
                print(f"Error during auto search: {e}")
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
                self.after(0, lambda: self._set_search_loading(False))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_search_complete(self, papers: list[Paper], popup: ProgressPopup):
        """Handle search completion - close popup and display papers."""
        popup.close()
        self.searched_papers = papers
        self._refresh_searched_papers_list()
        self._set_search_loading(False)

    def _set_search_loading(self, loading: bool):
        self.is_searching = loading
        if loading:
            self.search_btn.config(state="disabled", text="Searching")
        else:
            self.search_btn.config(state="normal", text="Auto Search")

    def _refresh_user_papers_list(self):
        for widget in self.user_papers_list.winfo_children():
            widget.destroy()
        self.user_paper_widgets.clear()
        
        if not self.user_papers:
            self._show_empty_state(self.user_papers_list, "No papers uploaded yet")
        else:
            for i, paper in enumerate(self.user_papers):
                if i > 0:
                    ttk.Separator(self.user_papers_list, orient="horizontal").pack(fill="x", padx=5)
                entry = self._create_paper_entry(self.user_papers_list, paper, self._remove_user_paper, True)
                self.user_paper_widgets[paper.id] = entry
        
        self._update_user_count()

    def _refresh_searched_papers_list(self):
        for widget in self.searched_papers_list.winfo_children():
            widget.destroy()
        self.searched_paper_widgets.clear()
        
        if not self.searched_papers:
            self._show_empty_state(self.searched_papers_list, "Click 'Auto Search' to find related papers")
        else:
            for i, paper in enumerate(self.searched_papers):
                if i > 0:
                    ttk.Separator(self.searched_papers_list, orient="horizontal").pack(fill="x", padx=5)
                entry = self._create_paper_entry(self.searched_papers_list, paper, self._remove_searched_paper, False)
                self.searched_paper_widgets[paper.id] = entry
        
        self._update_searched_count()

    def _update_user_count(self):
        self.user_count_label.config(text=str(len(self.user_papers)))

    def _update_searched_count(self):
        self.searched_count_label.config(text=str(len(self.searched_papers)))

    def _remove_user_paper(self, paper_id: str):
        removed = next((p for p in self.user_papers if p.id == paper_id), None)
        if removed:
            print(f"[Papers] Removed user paper: {removed.title[:60]}")
            
            # Delete the output folder for this paper
            output_folder = Path("output/literature") / paper_id
            if output_folder.exists():
                shutil.rmtree(output_folder)
        
        self.user_papers = [p for p in self.user_papers if p.id != paper_id]
        self._original_paper_ids.discard(paper_id)
        self._refresh_user_papers_list()
        self._save_papers()

    def _remove_searched_paper(self, paper_id: str):
        removed = next((p for p in self.searched_papers if p.id == paper_id), None)
        if removed:
            print(f"[Papers] Removed searched paper: {removed.title[:60]}")
            
            # Delete the output folder for this paper
            output_folder = Path("output/literature") / paper_id
            if output_folder.exists():
                shutil.rmtree(output_folder)
        
        self.searched_papers = [p for p in self.searched_papers if p.id != paper_id]
        self._original_paper_ids.discard(paper_id)
        self._refresh_searched_papers_list()
        self._save_papers()

    def _save_papers(self):
        """Save current paper selection to papers.json."""
        all_papers = self.user_papers + self.searched_papers
        if all_papers:
            LiteratureSearch.save_papers(all_papers, filename="papers.json", output_dir="output")
        elif PAPERS_FILE.exists():
            PAPERS_FILE.unlink()
            print("[Papers] All papers removed, deleted papers.json")

    def on_next(self):
        """Process new papers if any, then proceed or generate hypotheses."""
        all_papers = self.user_papers + self.searched_papers
        
        # Handle empty case
        if not all_papers:
            if PAPERS_FILE.exists():
                PAPERS_FILE.unlink()
                print("[Papers] All papers removed, deleted papers.json")
            super().on_next()
            return
        
        # Find new papers (not in original loaded set)
        new_papers = [p for p in all_papers if p.id not in self._original_paper_ids]
        
        # Find papers that need processing (no markdown_text but PDF exists)
        papers_needing_conversion = self._find_papers_needing_conversion(all_papers)
        
        if new_papers or papers_needing_conversion:
            # Process new papers and papers needing conversion (deduplicate by ID)
            seen_ids = set()
            papers_to_process = []
            for paper in new_papers + papers_needing_conversion:
                if paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    papers_to_process.append(paper)
            print(f"[Papers] Found {len(papers_to_process)} papers to process ({len(new_papers)} new, {len(papers_needing_conversion)} need conversion)")
            self._process_new_papers(all_papers, papers_to_process)
        elif HYPOTHESES_FILE.exists():
            # No papers to process, hypotheses exist - just continue
            super().on_next()
        else:
            # No papers to process, no hypothesis - generate it
            self._run_hypothesis_generation(all_papers)

    def _find_papers_needing_conversion(self, all_papers: list[Paper]) -> list[Paper]:
        """Find papers that need to be converted to markdown (have PDF but no markdown_text)."""
        papers_needing_conversion = []
        for paper in all_papers:
            # Check if paper doesn't have markdown_text
            has_markdown = getattr(paper, "markdown_text", None) and isinstance(paper.markdown_text, str) and paper.markdown_text.strip()
            if not has_markdown:
                # Check if PDF exists (either pdf_path or in output/literature/)
                pdf_exists = False
                if paper.pdf_path:
                    pdf_path = Path(paper.pdf_path)
                    if pdf_path.is_absolute() and pdf_path.exists():
                        pdf_exists = True
                    elif not pdf_path.is_absolute():
                        full_path = Path.cwd() / pdf_path
                        if full_path.exists():
                            pdf_exists = True
                
                if not pdf_exists:
                    # Check if PDF exists in output/literature/{id}/
                    safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
                    pdf_path = Path("output/literature") / safe_id / f"{safe_id}.pdf"
                    if pdf_path.exists():
                        pdf_exists = True
                
                if pdf_exists:
                    papers_needing_conversion.append(paper)
        
        return papers_needing_conversion

    def _process_new_papers(self, all_papers: list[Paper], new_papers: list[Paper]):
        """Download and convert new papers, save all, then continue or generate hypotheses."""
        popup = ProgressPopup(self.controller, "Processing New Papers")
        
        def task():
            try:
                # Step 1: Download non-user-provided new papers
                to_download = [p for p in new_papers if not p.user_provided]
                if to_download:
                    self.after(0, lambda: popup.update_status(f"Downloading {len(to_download)} PDF(s)"))
                    successful, failed = PDFDownloader.download_papers_as_pdfs(
                        to_download, 
                        base_folder="output/literature/"
                    )
                    print(f"Downloaded {successful} PDF(s), {failed} failed")
                
                # Step 2: Convert all new papers to markdown
                self.after(0, lambda: popup.update_status(f"Converting {len(new_papers)} PDF(s) to markdown"))
                converter = PDFConverter()
                converter.convert_all_papers(new_papers, base_folder="output/literature/")
                
                # Step 3: Save all papers to papers.json
                self.after(0, lambda: popup.update_status("Saving papers"))
                LiteratureSearch.save_papers(all_papers, filename="papers.json", output_dir="output")
                
                # Step 4: Update tracking set
                self._original_paper_ids = {p.id for p in all_papers}
                
                # Step 5: Continue or generate hypotheses
                if HYPOTHESES_FILE.exists():
                    self.after(0, lambda: self._on_processing_complete(popup))
                else:
                    self._run_hypothesis_generation(all_papers, popup)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_processing_complete(self, popup: ProgressPopup):
        """Handle completion when hypotheses already exist."""
        popup.close()
        self.controller.next_screen()
    
    def _run_hypothesis_generation(self, all_papers: list[Paper], popup: Optional[ProgressPopup] = None):
        """Run hypothesis generation from user input only."""
        # Create popup if not provided (called from on_next without processing)
        if popup is None:
            popup = ProgressPopup(self.controller, "Processing...")
        
        def task():
            try:
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Check if user provided hypothesis
                user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                user_provided_hypothesis = bool(user_requirements.hypothesis and user_requirements.hypothesis.strip())
                
                # Step 1: Ensure all papers are converted to markdown
                papers_needing_conversion = self._find_papers_needing_conversion(all_papers)
                if papers_needing_conversion:
                    self.after(0, lambda: popup.update_status(f"Converting {len(papers_needing_conversion)} PDF(s) to markdown"))
                    converter = PDFConverter()
                    converter.convert_all_papers(papers_needing_conversion, base_folder="output/literature/")
                
                # Step 2: Filter by markdown availability
                self.after(0, lambda: popup.update_status("Filtering papers with markdown"))
                papers_with_markdown: list[Paper] = []
                for p in all_papers:
                    if getattr(p, "markdown_text", None) and isinstance(p.markdown_text, str) and p.markdown_text.strip():
                        papers_with_markdown.append(p)
                
                # Update papers.json with filtered results
                LiteratureSearch.save_papers(papers_with_markdown, filename="papers.json", output_dir="output")
                
                # Only generate hypothesis if user provided one in requirements
                if user_provided_hypothesis:
                    self.after(0, lambda: popup.update_status("Creating hypothesis from user input"))
                    hypothesis_builder = HypothesisBuilder(
                        model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
                        paper_concept=paper_concept,
                        top_limitations=[],
                        num_papers_analyzed=0
                    )
                    hypothesis_builder.create_hypothesis_from_user_input(user_requirements)
                # If no user hypothesis, skip generation - user can create manually on hypothesis screen
                

                
                # Success - close popup and go to next screen
                self.after(0, lambda: self._on_generation_success(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_generation_success(self, popup: ProgressPopup):
        """Handle successful generation."""
        popup.close()
        self.controller.next_screen()
