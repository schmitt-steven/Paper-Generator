import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
import threading
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional

from ..base_frame import BaseFrame, ProgressPopup, create_gray_button
from ..icons import HoverColor
from ..info_texts import PAPER_SELECTION_INFO
from ..theme_colors import CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT, MUTED_TEXT
from phases.paper_search.paper import Paper
from phases.paper_search.user_paper_loader import UserPaperLoader
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper_ranking import PaperRanker
from phases.paper_search.paper_filter import PaperFilter
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from phases.context_analysis.user_requirements import UserRequirements

from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from utils.pdf_downloader import PDFDownloader
from utils.pdf_converter import PDFConverter
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
        
        next_text = "Continue" if HYPOTHESES_FILE.exists() else "Generate Hypothesis"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Paper Selection",
            next_text=next_text,
            info_content=PAPER_SELECTION_INFO
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
            
            
            print(f"[Papers] Loaded {len(self.user_papers)} user papers, {len(self.searched_papers)} searched papers")
            
            # Refresh UI
            self._refresh_user_papers_list()
            self._refresh_searched_papers_list()
            
        except Exception as e:
            print(f"Error loading papers from {PAPERS_FILE}: {e}")
            import traceback
            traceback.print_exc()

    def _create_section_container(
        self,
        parent,
        title: str,
        count: int,
        button_text: str,
        button_command: Callable
    ) -> tuple:
        from ..base_frame import CardBorderFrame
        section_frame = CardBorderFrame(parent, padx=1, pady=1)
        section_frame.pack(fill="x", pady=10)
        
        # Header row
        header_frame = ttk.Frame(section_frame, style="CardHeader.TFrame", padding=(10, 6))
        header_frame.pack(fill="x")
        
        left_header = ttk.Frame(header_frame, style="CardHeader.TFrame")
        left_header.pack(side="left")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
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
            fg=MUTED_TEXT,
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
        papers_list = ttk.Frame(section_frame, style="CardContent.TFrame", padding=10)
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
        ttk.Label(container, text=message, font=self.controller.fonts.default_font, foreground="gray", style="CardRow.TLabel").pack(pady=20)

    def _create_paper_entry(self, parent: ttk.Frame, paper: Paper, 
                            on_remove: Callable, is_user_paper: bool) -> ttk.Frame:
        entry_frame = ttk.Frame(parent, style="CardRow.TFrame", padding="8")
        entry_frame.pack(fill="x")
        
        content_row = ttk.Frame(entry_frame, style="CardRow.TFrame")
        content_row.pack(fill="x")
        
        # Button container
        btn_container = ttk.Frame(content_row, style="CardRow.TFrame")
        btn_container.pack(side="right", padx=(10, 0))

        # Upload button for closed access papers (always show to allow overwriting)
        if not is_user_paper and not paper.is_open_access:
            upload_btn = self.controller.icons.create_icon_label(
                btn_container,
                icon_name="upload",
                command=lambda: self._on_upload_paper_pdf(paper),
                hover_color=HoverColor.GREEN,
                base_color=HoverColor.GRAY
            )
            upload_btn.pack(side="left", padx=(0, 10))
        
        # X button
        x_btn = self.controller.icons.create_icon_label(
            btn_container,
            icon_name="x",
            command=lambda: on_remove(paper.id),
            base_color=HoverColor.GRAY
        )
        x_btn.pack(side="left")

        # Content Frame (Title + Metadata)
        content_frame = ttk.Frame(content_row, style="CardRow.TFrame")
        content_frame.pack(side="left", fill="x", expand=True)
        
        title_label = ttk.Label(content_frame, text=paper.title, font=self.controller.fonts.default_font, style="CardRow.TLabel")
        title_label.pack(anchor="w", fill="x")
        
        metadata_frame = ttk.Frame(content_frame, style="CardRow.TFrame")
        metadata_frame.pack(anchor="w", pady=(2, 0), fill="x")
        
        # 1. Status Tag
        status_text, status_color = self._get_paper_status(paper)
        if status_text:
            status_label = ttk.Label(metadata_frame, text=status_text, font=self.controller.fonts.text_area_font, foreground=status_color, style="CardRow.TLabel")
            status_label.pack(side="left")
            
            # Separator if there is other metadata
            ttk.Label(metadata_frame, text="  \u00B7  ", font=self.controller.fonts.text_area_font, foreground="gray", style="CardRow.TLabel").pack(side="left")

        # 2. Bibliographic Metadata
        metadata = self._format_paper_bibliographic_info(paper, is_user_paper)
        metadata_label = ttk.Label(metadata_frame, text=metadata, font=self.controller.fonts.text_area_font, foreground="gray", style="CardRow.TLabel")
        metadata_label.pack(side="left")
        
        def update_wraplength(event):
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
        """Return status text and color for closed-access papers."""
        if not paper.is_open_access and not paper.user_provided:
            if self._check_pdf_exists(paper):
                return "PDF Uploaded", "green"
            return "Closed Access", "red"
        return None, None

    def _format_paper_bibliographic_info(self, paper: Paper, is_user_paper: bool = True) -> str:
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
            if isinstance(paper.published, datetime):
                parts.append(str(paper.published.year))
            else:
                year_match = re.search(r'(\d{4})', str(paper.published))
                if year_match:
                    parts.append(year_match.group(1))
        
        if paper.citation_count is not None:
            parts.append(f"{paper.citation_count:,} citations")
        
        # Show similarity /relevance score for searched papers
        if not is_user_paper and paper.ranking and paper.ranking.relevance_score is not None:
            parts.append(f"Relevance: {paper.ranking.relevance_score:.2f}")
        
        return "  \u00B7  ".join(parts)

    def _on_paper_click(self, paper: Paper, is_user_paper: bool):
        """Open the paper's PDF or Semantic Scholar page."""
        pdf_path = self._get_pdf_path(paper)
        if pdf_path:
            webbrowser.open(f"file://{pdf_path.resolve()}")
        elif paper.id:
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
            
            # 2. Remove existing PDF and markdown files (since we're overwriting)
            if dest_path.exists():
                dest_path.unlink()
                print(f"[Papers] Removed old PDF file: {dest_path}")
            
            md_path = output_folder / f"{safe_id}.md"
            if md_path.exists():
                md_path.unlink()
                print(f"[Papers] Removed old markdown file: {md_path}")
            
            # 3. Clear paper's markdown_text so it will be re-converted
            paper.markdown_text = None
            
            # 4. Copy file
            shutil.copy2(file_path, dest_path)
            
            # 5. Update paper object
            paper.pdf_path = str(dest_path.resolve().relative_to(Path.cwd()))
            
            # 6. Save and refresh
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
        popup = ProgressPopup(self.controller, "Uploading Papers")
        popup.update_status(f"Processing {len(file_paths)} file(s)")
        
        thread = threading.Thread(target=self._process_uploaded_files, args=(file_paths, popup))
        thread.daemon = True
        thread.start()

    def _process_uploaded_files(self, file_paths: tuple, popup: ProgressPopup):
        """Process uploaded PDF files. Simplified: process directly, skip duplicates."""
        try:
            # Get existing paper IDs to skip duplicates
            existing_ids = {p.id for p in self.user_papers}
            
            loader = UserPaperLoader(model_name=Settings.LITERATURE_SEARCH_MODEL)
            new_papers = []
            total = len(file_paths)
            
            for i, file_path in enumerate(file_paths, 1):
                pdf_path = Path(file_path)
                paper_id = f"user_{pdf_path.stem}"
                
                self.after(0, lambda p=pdf_path.name, idx=i: popup.update_status(f"Processing {idx}/{total}: {p}"))
                
                # Skip if already loaded
                if paper_id in existing_ids:
                    print(f"[Papers] Skipping duplicate: {pdf_path.name}")
                    continue
                
                # Process paper
                paper = loader.load_user_paper(pdf_path)
                if paper:
                    new_papers.append(paper)
                    existing_ids.add(paper.id)
            
            self.after(0, lambda: self._on_upload_complete(new_papers, popup))
            
        except Exception as e:
            print(f"Error processing uploaded files: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda err=str(e): popup.show_error(err))
            self.after(0, lambda: self._set_upload_loading(False))

    def _on_upload_complete(self, new_papers: list[Paper], popup: ProgressPopup):
        for paper in new_papers:
            self.user_papers.append(paper)
            print(f"[Papers] Added user paper: {paper.title[:60]}")
        
        self._save_papers()  # Save immediately after upload
        self._refresh_user_papers_list()
        self._set_upload_loading(False)
        popup.close()

    def _set_upload_loading(self, loading: bool):
        self.is_uploading = loading
        if loading:
            self.upload_btn.config(state="disabled", text="Uploading...")
        else:
            self.upload_btn.config(state="normal", text="Upload")

    def _on_auto_search_click(self):
        if self.is_searching:
            return
        
        if self.searched_papers:
            if not messagebox.askyesno(
                "Confirm Auto Search",
                "This will replace any existing found papers with new search results.\n\nDo you want to continue?"
            ):
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
                ranking_context = paper_concept.description
                ranked_papers = ranker.rank_papers(
                    papers=searched_papers,
                    context=ranking_context,
                    weights={'relevance': 0.8, 'citations': 0.1, 'recency': 0.1}
                )
                
                # Save unfiltered papers
                #LiteratureSearch.save_papers(ranked_papers, filename="papers_unfiltered.json", output_dir="output")
                
                # Step 3: Filter papers
                self.after(0, lambda: popup.update_status("Filtering found papers"))
                research_context = f"{paper_concept.description}\n\nOpen Research Questions:\n{paper_concept.open_questions}"
                filtered_papers = PaperFilter.filter_papers(
                    papers=ranked_papers,
                    research_context=research_context,
                    model_name=Settings.LITERATURE_SEARCH_MODEL,
                    target_count=50,
                    min_relevance=0.5
                )
                
                # Step 4: Check arXiv for free PDF versions
                self.after(0, lambda: popup.update_status("Checking arXiv for free PDFs"))
                from utils.open_access_finder import find_open_access_pdfs
                filtered_papers = find_open_access_pdfs(filtered_papers)
                
                # Step 5: Show results on screen
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
        self._save_papers()  # Save immediately after search
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
            # Sort by relevance score (highest first)
            sorted_papers = sorted(
                self.searched_papers,
                key=lambda p: p.ranking.relevance_score if p.ranking and p.ranking.relevance_score else 0,
                reverse=True
            )
            for i, paper in enumerate(sorted_papers):
                if i > 0:
                    ttk.Separator(self.searched_papers_list, orient="horizontal").pack(fill="x", padx=5)
                entry = self._create_paper_entry(self.searched_papers_list, paper, self._remove_searched_paper, False)
                self.searched_paper_widgets[paper.id] = entry
        
        self._update_searched_count()

    def _update_user_count(self):
        self.user_count_label.config(text=str(len(self.user_papers)))

    def _update_searched_count(self):
        total = len(self.searched_papers)
        if total == 0:
            self.searched_count_label.config(text="0")
        else:
            open_count = sum(1 for p in self.searched_papers if p.is_open_access)
            closed_count = total - open_count
            self.searched_count_label.config(text=f"{total} ({open_count} open, {closed_count} closed access)")

    def _remove_paper(self, paper_id: str, is_user_paper: bool):
        """Remove a paper and delete its literature folder."""
        papers = self.user_papers if is_user_paper else self.searched_papers
        removed = next((p for p in papers if p.id == paper_id), None)
        if removed:
            print(f"[Papers] Removed: {removed.title[:60]}")
            output_folder = Path("output/literature") / paper_id
            if output_folder.exists():
                shutil.rmtree(output_folder)
        
        if is_user_paper:
            self.user_papers = [p for p in self.user_papers if p.id != paper_id]
            self._refresh_user_papers_list()
        else:
            self.searched_papers = [p for p in self.searched_papers if p.id != paper_id]
            self._refresh_searched_papers_list()
        self._save_papers()

    def _remove_user_paper(self, paper_id: str):
        self._remove_paper(paper_id, is_user_paper=True)

    def _remove_searched_paper(self, paper_id: str):
        self._remove_paper(paper_id, is_user_paper=False)

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
        
        # Find papers that need processing (download and/or conversion)
        papers_needing_download, papers_needing_conversion = self._find_papers_needing_processing(all_papers)
        
        # Combine papers that need processing (deduplicate by ID)
        seen_ids = set()
        papers_to_process = []
        for paper in papers_needing_download + papers_needing_conversion:
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                papers_to_process.append(paper)
        
        if papers_to_process:
            print(f"[Papers] Processing {len(papers_to_process)} papers: {len(papers_needing_download)} download, {len(papers_needing_conversion)} convert")
            self._process_new_papers(all_papers, papers_to_process, papers_needing_download)
        elif HYPOTHESES_FILE.exists():
            # No papers to process, hypothesis exists -> continue
            super().on_next()
        else:
            # No papers to process, no hypothesis -> generate it
            self._run_hypothesis_generation(all_papers)

    def _find_papers_needing_processing(self, all_papers: list[Paper]) -> tuple[list[Paper], list[Paper]]:
        """
        Find papers that need processing.
        
        Returns:
            Tuple of (papers_needing_download, papers_needing_conversion)
            - papers_needing_download: Open-access papers without a downloaded PDF
            - papers_needing_conversion: Papers with PDF but no markdown_text
        """
        papers_needing_download = []
        papers_needing_conversion = []
        
        for paper in all_papers:
            # Check if paper has markdown_text
            has_markdown = getattr(paper, "markdown_text", None) and isinstance(paper.markdown_text, str) and paper.markdown_text.strip()
            if has_markdown:
                continue  # Paper is fully processed
            
            # Check if PDF exists
            pdf_exists = self._check_pdf_exists(paper)
            
            if pdf_exists:
                # Has PDF but no markdown -> needs conversion
                papers_needing_conversion.append(paper)
            elif paper.is_open_access and paper.pdf_url and not paper.user_provided:
                # Open-access paper without PDF -> needs download (then conversion)
                papers_needing_download.append(paper)
            # Note: Closed-access papers without PDF are skipped (user needs to upload manually)
        
        return papers_needing_download, papers_needing_conversion
    
    def _get_pdf_path(self, paper: Paper) -> Optional[Path]:
        """Return the PDF path if it exists, None otherwise."""
        # Check pdf_path first
        if paper.pdf_path:
            pdf_path = Path(paper.pdf_path)
            if not pdf_path.is_absolute():
                pdf_path = Path.cwd() / pdf_path
            if pdf_path.exists():
                return pdf_path
        
        # Check standard location
        safe_id = "".join([c for c in paper.id if c.isalnum() or c in ('-', '_', '.')])
        standard_path = Path("output/literature") / safe_id / f"{safe_id}.pdf"
        return standard_path if standard_path.exists() else None

    def _check_pdf_exists(self, paper: Paper) -> bool:
        """Check if a PDF file exists for the given paper."""
        return self._get_pdf_path(paper) is not None

    def _process_new_papers(self, all_papers: list[Paper], papers_to_process: list[Paper], papers_needing_download: list[Paper] = None):
        """Download and convert papers, save all, then continue or generate hypotheses."""
        popup = ProgressPopup(self.controller, "Processing Papers")
        
        # Use provided list or empty if not given
        if papers_needing_download is None:
            papers_needing_download = []
        
        def task():
            try:
                # Step 1: Download papers that need downloading
                if papers_needing_download:
                    self.after(0, lambda: popup.update_status(f"Downloading {len(papers_needing_download)} PDF(s)"))
                    successful, failed = PDFDownloader.download_papers_as_pdfs(
                        papers_needing_download, 
                        base_folder="output/literature/"
                    )
                    print(f"Downloaded {successful} PDF(s), {failed} failed")
                
                # Step 2: Convert all papers to process to markdown
                self.after(0, lambda: popup.update_status(f"Converting {len(papers_to_process)} PDF(s) to markdown"))
                converter = PDFConverter()
                converter.convert_all_papers(papers_to_process, base_folder="output/literature/")
                
                # Step 3: Save and continue
                self.after(0, lambda: popup.update_status("Saving papers"))
                LiteratureSearch.save_papers(all_papers, filename="papers.json", output_dir="output")
                
                if HYPOTHESES_FILE.exists():
                    self.after(0, lambda: self._finish_processing(popup))
                else:
                    self._run_hypothesis_generation(all_papers, popup)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _finish_processing(self, popup: ProgressPopup):
        """Close popup and go to next screen."""
        popup.close()
        self.controller.next_screen()
    
    def _run_hypothesis_generation(self, all_papers: list[Paper], popup: Optional[ProgressPopup] = None):
        """Generate hypothesis from user input."""
        if popup is None:
            popup = ProgressPopup(self.controller, "Processing")
        
        def task():
            try:
                self.after(0, lambda: popup.update_status("Loading paper concept"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                
                # Only generate if user provided hypothesis
                if user_requirements.hypothesis and user_requirements.hypothesis.strip():
                    self.after(0, lambda: popup.update_status("Creating hypothesis from user input"))
                    HypothesisBuilder(
                        model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
                        paper_concept=paper_concept,
                        top_limitations=[],
                        num_papers_analyzed=0
                    ).create_hypothesis_from_user_input(user_requirements)
                
                self.after(0, lambda: self._finish_processing(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        threading.Thread(target=task, daemon=True).start()
