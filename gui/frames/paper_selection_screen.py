from tkinter import ttk, filedialog
import webbrowser
import threading
import shutil
import re
from pathlib import Path
from typing import List, Dict, Callable, Any

from ..base_frame import BaseFrame, ProgressPopup, create_gray_button
from phases.paper_search.paper import Paper
from phases.paper_search.user_paper_loader import UserPaperLoader
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper_ranking import PaperRanker
from phases.paper_search.paper_filtering import PaperFilter
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from phases.context_analysis.user_requirements import load_user_requirements
from phases.hypothesis_generation.paper_analysis import PaperAnalyzer
from phases.hypothesis_generation.limitation_analysis import LimitationAnalyzer
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from utils.pdf_downloader import PDFDownloader
from utils.pdf_converter_pymupdf_marker import PDFConverter
from settings import Settings


HYPOTHESES_FILE = Path("output/hypotheses.json")

class PaperSelectionScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.user_papers: List[Paper] = []
        self.searched_papers: List[Paper] = []
        
        # Widget references
        self.user_paper_widgets: Dict[str, ttk.Frame] = {}
        self.searched_paper_widgets: Dict[str, ttk.Frame] = {}
        
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

    def _create_section_container(self, parent, title: str, count: int, 
                                   button_text: str, button_command: Callable) -> tuple:
        section_frame = ttk.Frame(parent, style="Card.TFrame", padding=1)
        section_frame.pack(fill="x", pady=10)
        
        # Header row
        header_frame = ttk.Frame(section_frame, padding=10)
        header_frame.pack(fill="x")
        
        left_header = ttk.Frame(header_frame)
        left_header.pack(side="left")
        
        ttk.Label(left_header, text=title, font=("SF Pro", 14, "bold")).pack(side="left")
        count_label = ttk.Label(left_header, text=str(count), font=("SF Pro", 14), foreground="gray")
        count_label.pack(side="left", padx=(10, 0))
        
        action_btn = ttk.Button(header_frame, text=button_text, command=button_command)
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
        ttk.Label(container, text=message, font=("SF Pro", 14), foreground="gray").pack(pady=20)

    def _create_paper_entry(self, parent: ttk.Frame, paper: Paper, 
                            on_remove: Callable, is_user_paper: bool) -> ttk.Frame:
        entry_frame = ttk.Frame(parent, padding="8")
        entry_frame.pack(fill="x")
        
        content_row = ttk.Frame(entry_frame)
        content_row.pack(fill="x")
        
        content_frame = ttk.Frame(content_row)
        content_frame.pack(side="left", fill="x", expand=True)
        
        title_label = ttk.Label(content_frame, text=paper.title, font=("SF Pro", 14, "bold"), wraplength=500)
        title_label.pack(anchor="w")
        
        metadata = self._format_paper_metadata(paper)
        metadata_label = ttk.Label(content_frame, text=metadata, font=("SF Pro", 12), foreground="gray")
        metadata_label.pack(anchor="w", pady=(2, 0))
        
        trash_btn = create_gray_button(content_row, text="\U0001F5D1", command=lambda: on_remove(paper.id), width=3)
        trash_btn.pack(side="right", padx=(10, 0))
        
        for widget in [content_frame, title_label, metadata_label]:
            widget.bind("<Button-1>", lambda e, p=paper: self._on_paper_click(p, is_user_paper))
            widget.configure(cursor="hand2")
        
        return entry_frame

    def _format_paper_metadata(self, paper: Paper) -> str:
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
        if is_user_paper and paper.pdf_path:
            pdf_path = Path(paper.pdf_path).resolve()
            if pdf_path.exists():
                webbrowser.open(f"file://{pdf_path}")
            else:
                print(f"PDF not found: {pdf_path}")
        elif paper.pdf_url:
            webbrowser.open(paper.pdf_url)
        elif paper.id:
            webbrowser.open(f"https://www.semanticscholar.org/paper/{paper.id}")

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

    def _on_upload_complete(self, new_papers: List[Paper]):
        for paper in new_papers:
            self.user_papers.append(paper)
            print(f"[Papers] Added user paper: {paper.title[:60]}...")
        
        self._refresh_user_papers_list()
        self._set_upload_loading(False)

    def _set_upload_loading(self, loading: bool):
        self.is_uploading = loading
        if loading:
            self.upload_btn.config(state="disabled", text="Processing...")
        else:
            self.upload_btn.config(state="normal", text="Upload")

    def _on_auto_search_click(self):
        if self.is_searching:
            return
        
        self._set_search_loading(True)
        popup = ProgressPopup(self.controller, "Searching Papers...")
        
        def task():
            try:
                # Step 1: Search
                self.after(0, lambda: popup.update_status("Building search queries..."))
                paper_concept: PaperConcept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                literature_search = LiteratureSearch(model_name=Settings.LITERATURE_SEARCH_MODEL)
                search_queries = literature_search.build_search_queries(paper_concept)
                
                self.after(0, lambda: popup.update_status(f"Searching {len(search_queries)} queries..."))
                papers = literature_search.search_papers(search_queries, max_results_per_query=15)
                
                # Filter out papers already in user papers
                user_paper_ids = {p.id for p in self.user_papers}
                searched_papers = [p for p in papers if p.id not in user_paper_ids]
                
                if not searched_papers:
                    self.after(0, lambda: self._on_search_complete([], popup))
                    return
                
                # Step 2: Rank papers
                self.after(0, lambda: popup.update_status("Ranking papers..."))
                ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
                ranking_context = f"{paper_concept.description}\nOpen Research Questions:\n{paper_concept.open_questions}"
                ranked_papers = ranker.rank_papers(
                    papers=searched_papers,
                    context=ranking_context,
                    weights={'relevance': 0.7, 'citations': 0.2, 'recency': 0.1}
                )
                
                # Step 3: Filter papers for diverse selection
                self.after(0, lambda: popup.update_status("Filtering papers for diverse selection..."))
                filtered_papers = PaperFilter.filter_diverse(
                    ranked_papers,
                    n_cutting_edge=20,
                    n_hidden_gems=15,
                    n_classics=15,
                    n_well_rounded=10
                )
                
                # Step 4: Present to user
                self.after(0, lambda: self._on_search_complete(filtered_papers, popup))
                
            except Exception as e:
                print(f"Error during auto search: {e}")
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
                self.after(0, lambda: self._set_search_loading(False))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_search_complete(self, papers: List[Paper], popup: ProgressPopup):
        """Handle search completion - close popup and display papers."""
        popup.close()
        self.searched_papers = papers
        self._refresh_searched_papers_list()
        self._set_search_loading(False)

    def _set_search_loading(self, loading: bool):
        self.is_searching = loading
        if loading:
            self.search_btn.config(state="disabled", text="Searching...")
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
            print(f"[Papers] Removed user paper: {removed.title[:60]}...")
            
            # Delete the output folder for this paper
            output_folder = Path("output/literature") / paper_id
            if output_folder.exists():
                shutil.rmtree(output_folder)
        
        self.user_papers = [p for p in self.user_papers if p.id != paper_id]
        self._refresh_user_papers_list()

    def _remove_searched_paper(self, paper_id: str):
        removed = next((p for p in self.searched_papers if p.id == paper_id), None)
        if removed:
            print(f"[Papers] Removed searched paper: {removed.title[:60]}...")
        
        self.searched_papers = [p for p in self.searched_papers if p.id != paper_id]
        self._refresh_searched_papers_list()

    def on_next(self):
        """Download PDFs, convert to markdown, save papers, and proceed or generate hypotheses."""
        all_papers = self.user_papers + self.searched_papers
        
        if not all_papers:
            super().on_next()
            return
        
        # Check if output exists
        if HYPOTHESES_FILE.exists():
            # If hypotheses already exist, just save papers and continue
            LiteratureSearch.save_papers(all_papers, filename="papers.json", output_dir="output")
            print(f"Saved {len(all_papers)} selected papers")
            super().on_next()
        else:
            # Download, convert, save, then generate
            self._download_and_convert(all_papers)

    def _download_and_convert(self, all_papers: List[Paper]):
        """Download PDFs for searched papers, convert all to markdown, save, then generate hypotheses."""
        popup = ProgressPopup(self.controller, "Processing Papers...")
        
        def task():
            try:
                # Separate user-provided and searched papers
                user_provided_papers = [p for p in all_papers if p.user_provided]
                searched_papers = [p for p in all_papers if not p.user_provided]
                
                # Step 1: Download PDFs for searched papers only
                if searched_papers:
                    self.after(0, lambda: popup.update_status(f"Downloading {len(searched_papers)} PDF(s)..."))
                    successful, failed = PDFDownloader.download_papers_as_pdfs(
                        searched_papers, 
                        base_folder="output/literature/"
                    )
                    print(f"Downloaded {successful} PDF(s), {failed} failed")
                else:
                    print("No searched papers to download")
                
                # Step 2: Convert all papers (both user and searched) from PDF to markdown
                self.after(0, lambda: popup.update_status(f"Converting {len(all_papers)} PDF(s) to markdown..."))
                converter = PDFConverter()
                papers_with_markdown = converter.convert_all_papers(all_papers, base_folder="output/literature/")
                
                # Step 3: Save papers.json with all papers (now with markdown)
                self.after(0, lambda: popup.update_status("Saving papers..."))
                LiteratureSearch.save_papers(papers_with_markdown, filename="papers.json", output_dir="output")
                print(f"Saved {len(papers_with_markdown)} papers with markdown")
                
                # Step 4: Proceed with hypothesis generation (continue in same thread)
                self._run_generation_continuation(papers_with_markdown, popup)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def _run_generation_continuation(self, all_papers: List[Paper], popup: ProgressPopup):
        """Continue hypothesis generation pipeline (called from download/convert thread)."""
        try:
            # Load paper concept
            self.after(0, lambda: popup.update_status("Loading paper concept..."))
            paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
            
            # Check if user provided hypothesis
            user_requirements = load_user_requirements("user_files/user_requirements.md")
            user_provided_hypothesis = bool(user_requirements.hypothesis and user_requirements.hypothesis.strip())
            
            # Papers are already ranked and filtered from the search step
            # Now filter by markdown availability (papers should already have markdown from conversion step)
            self.after(0, lambda: popup.update_status("Filtering papers with markdown..."))
            papers_with_markdown: List[Paper] = []
            for p in all_papers:
                if getattr(p, "markdown_text", None) and isinstance(p.markdown_text, str) and p.markdown_text.strip():
                    papers_with_markdown.append(p)
            
            # Update papers.json with filtered results (papers already ranked from search step)
            LiteratureSearch.save_papers(papers_with_markdown, filename="papers.json", output_dir="output")
            
            findings: list[Any] = []
            top_limitations: list[Any] = []
            
            if not user_provided_hypothesis:
                # Step 3: Extract findings
                self.after(0, lambda: popup.update_status("Extracting findings from papers..."))
                analyzer = PaperAnalyzer(model_name=Settings.PAPER_ANALYSIS_MODEL)
                findings = analyzer.extract_findings(papers_with_markdown)
                
                # Step 4: Analyze limitations
                self.after(0, lambda: popup.update_status("Analyzing limitations..."))
                limitation_analyzer = LimitationAnalyzer.build_from_findings(findings, paper_concept)
                top_limitations = limitation_analyzer.find_top_limitations(n=10)
            
            # Step 5: Generate hypotheses
            self.after(0, lambda: popup.update_status("Generating hypotheses..."))
            hypothesis_builder = HypothesisBuilder(
                model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
                embedding_model_name=Settings.HYPOTHESIS_BUILDER_EMBEDDING_MODEL,
                paper_concept=paper_concept,
                top_limitations=top_limitations,
                num_papers_analyzed=len(findings)
            )
            
            if user_provided_hypothesis:
                hypotheses = hypothesis_builder.create_hypothesis_from_user_input(user_requirements)
            else:
                hypotheses = hypothesis_builder.generate_hypotheses(n_hypotheses=5)
                # Select best hypothesis
                if hypotheses:
                    hypothesis_builder.select_best_hypotheses(hypotheses, max_n=1)
            
            # Success - close popup and proceed
            self.after(0, lambda: self._on_generation_success(popup))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda err=str(e): popup.show_error(err))

    def _on_generation_success(self, popup: ProgressPopup):
        """Handle successful generation."""
        popup.close()
        self.controller.next_screen()
