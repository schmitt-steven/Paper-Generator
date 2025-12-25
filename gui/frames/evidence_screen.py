"""
Evidence Manager Screen - Display, add, and remove evidence chunks by section.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, List, Optional

from ..base_frame import BaseFrame, ProgressPopup, TextBorderFrame, create_scrollable_text_area
from phases.paper_writing.data_models import Evidence, PaperChunk, Section
from phases.paper_writing.evidence_manager import (
    load_evidence, save_evidence, add_evidence, remove_evidence, get_evidence_stats,
    EVIDENCE_FILE
)
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper import Paper


# Sections to display (excluding Abstract and Acknowledgements)
DISPLAY_SECTIONS = [
    Section.INTRODUCTION,
    Section.RELATED_WORK,
    Section.METHODS,
    Section.RESULTS,
    Section.DISCUSSION,
    Section.CONCLUSION,
]


class EvidenceChunkCard(ttk.Frame):
    """A card displaying a single evidence chunk with metadata and remove button."""
    
    def __init__(self, parent, evidence: Evidence, on_remove, controller):
        super().__init__(parent, style="Card.TFrame", padding=1)
        self.evidence = evidence
        self.on_remove = on_remove
        self.controller = controller
        
        self._build_ui()
    
    def _build_ui(self):
        paper = self.evidence.chunk.paper
        
        # Get colors for header
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        
        # Header frame using grid: title+meta on left, button on right
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        header.grid_columnconfigure(0, weight=1)  # Left column expands
        
        # Left side: title and metadata stacked
        left_frame = tk.Frame(header, bg=header_bg)
        left_frame.grid(row=0, column=0, sticky="w")
        
        # Paper title (truncated)
        title_text = paper.title
        if len(title_text) > 60:
            title_text = title_text[:57] + "..."
        
        tk.Label(
            left_frame,
            text=title_text,
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=header_fg,
            anchor="w"
        ).pack(anchor="w")
        
        # Metadata (directly under title)
        authors_text = paper.authors[0] if paper.authors else "Unknown"
        if len(paper.authors) > 1:
            authors_text += " et al."
        
        year_text = paper.published[:4] if paper.published else "N/A"
        citations = paper.citation_count or 0
        citations_text = f"{citations:,}" if citations else "N/A"
        score_text = f"Score: {self.evidence.combined_score:.2f}"
        
        meta_text = f"{authors_text} · {year_text} · {citations_text} citations · {score_text}"
        
        tk.Label(
            left_frame,
            text=meta_text,
            font=self.controller.fonts.small_font if hasattr(self.controller.fonts, 'small_font') else self.controller.fonts.default_font,
            bg=header_bg,
            fg="#888888",
            anchor="w"
        ).pack(anchor="w")
        
        # Remove button on right
        remove_btn = ttk.Button(
            header,
            text="✕",
            width=3,
            command=self._on_remove_click
        )
        remove_btn.grid(row=0, column=1, padx=(5, 0))
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Summary text area (no border)
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True)
        
        text_bg = "#1a1a1a" if self.controller.current_theme == "dark" else "#ffffff"
        text_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        
        text_widget = tk.Text(
            text_frame,
            height=6,
            font=self.controller.fonts.text_area_font,
            wrap="word",
            background=text_bg,
            foreground=text_fg,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=12,  # Inner horizontal padding
            pady=10,  # Inner vertical padding
            yscrollcommand=scrollbar.set
        )
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.insert("1.0", self.evidence.summary)
        text_widget.config(state="disabled")  # Read-only
        
        self.text_widget = text_widget
    
    def _on_remove_click(self):
        """Handle remove button click with confirmation."""
        if messagebox.askyesno("Remove Evidence", "Remove this evidence chunk?"):
            self.on_remove(self.evidence)


class CollapsibleSectionCard(ttk.Frame):
    """A collapsible card for a paper section containing evidence chunks."""
    
    def __init__(self, parent, section: Section, controller, on_chunk_removed):
        super().__init__(parent, style="Card.TFrame", padding=1)
        self.section = section
        self.controller = controller
        self.on_chunk_removed = on_chunk_removed
        self.expanded = False
        self.evidence_list: List[Evidence] = []
        self.chunk_cards: List[EvidenceChunkCard] = []
        
        self._build_ui()
    
    def _build_ui(self):
        # Header frame with toggle
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        header.bind("<Button-1>", lambda e: self.toggle())
        
        # Get colors  
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        
        # Toggle indicator
        self.toggle_label = tk.Label(
            header,
            text="▶",
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.toggle_label.pack(side="left", padx=(0, 10))
        self.toggle_label.bind("<Button-1>", lambda e: self.toggle())
        
        # Section title
        self.title_label = tk.Label(
            header,
            text=f"{self.section.value} (0 chunks)",
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", lambda e: self.toggle())
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame (hidden by default)
        self.content_frame = ttk.Frame(self, padding=10)
        # Don't pack yet - only show when expanded
    
    def toggle(self):
        """Toggle expansion state."""
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_label.config(text="▼")
            self.content_frame.pack(fill="both", expand=True)
        else:
            self.toggle_label.config(text="▶")
            self.content_frame.pack_forget()
    
    def expand(self):
        """Force expand."""
        if not self.expanded:
            self.toggle()
    
    def collapse(self):
        """Force collapse."""
        if self.expanded:
            self.toggle()
    
    def set_evidence(self, evidence_list: List[Evidence]):
        """Set the evidence chunks for this section."""
        self.evidence_list = evidence_list
        self._update_title()
        self._render_chunks()
    
    def _update_title(self):
        """Update the title with chunk count."""
        count = len(self.evidence_list)
        self.title_label.config(text=f"{self.section.value} ({count} chunk{'s' if count != 1 else ''})")
    
    def _render_chunks(self):
        """Render all evidence chunk cards."""
        # Clear existing
        for card in self.chunk_cards:
            card.destroy()
        self.chunk_cards.clear()
        
        # Create new cards
        for evidence in self.evidence_list:
            card = EvidenceChunkCard(
                self.content_frame,
                evidence,
                on_remove=self._handle_remove,
                controller=self.controller
            )
            card.pack(fill="x", pady=(0, 12))
            self.chunk_cards.append(card)
    
    def _handle_remove(self, evidence: Evidence):
        """Handle chunk removal."""
        self.evidence_list = [e for e in self.evidence_list if e is not evidence]
        self._update_title()
        self._render_chunks()
        self.on_chunk_removed(self.section, evidence)
    
    def add_chunk(self, evidence: Evidence):
        """Add a new evidence chunk at the top."""
        self.evidence_list.insert(0, evidence)  # Add at top for visibility
        self._update_title()
        self._render_chunks()


class AddChunkDialog(tk.Toplevel):
    """Dialog for adding a new evidence chunk."""
    
    def __init__(self, parent, papers: List[Paper], sections: List[Section], on_add):
        super().__init__(parent)
        self.parent = parent
        self.papers = papers
        self.sections = sections
        self.on_add = on_add
        
        self.title("Add Evidence Chunk")
        self.transient(parent)
        self.resizable(True, True)
        self.minsize(600, 550)
        self.geometry("750x650")
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 750) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 650) // 2
        self.geometry(f"+{x}+{y}")
        
        self._build_ui()
        
        self.grab_set()
        self.focus_set()
    
    def _build_ui(self):
        main = ttk.Frame(self, padding=20)
        main.pack(fill="both", expand=True)
        main.grid_rowconfigure(5, weight=1)
        main.grid_columnconfigure(0, weight=1)
        
        # Section selection
        ttk.Label(main, text="Target Section(s):", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        
        sections_frame = ttk.Frame(main)
        sections_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        self.section_vars: Dict[Section, tk.BooleanVar] = {}
        for i, section in enumerate(self.sections):
            var = tk.BooleanVar(value=False)
            self.section_vars[section] = var
            cb = ttk.Checkbutton(sections_frame, text=section.value, variable=var)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=5, pady=2)
        
        # Paper selection
        ttk.Label(main, text="Source Paper:", font=("Segoe UI", 10, "bold")).grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )
        
        self.paper_var = tk.StringVar()
        self.paper_combo = ttk.Combobox(main, textvariable=self.paper_var, width=70)
        self.paper_values = [f"[{p.published[:4] if p.published else 'N/A'}] {p.title}" for p in self.papers]
        self.paper_combo['values'] = self.paper_values
        self.paper_combo.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        
        # Bind for filtering
        self.paper_var.trace_add('write', self._filter_papers)
        
        # Text input - styled like other screens
        ttk.Label(main, text="Evidence / Summary / Quote(s):", font=("Segoe UI", 10, "bold")).grid(
            row=4, column=0, sticky="w", pady=(0, 5)
        )
        
        text_frame = ttk.Frame(main)
        text_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 20))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        # Get theme colors
        is_dark = hasattr(self.parent, 'current_theme') and self.parent.current_theme == "dark"
        text_bg = "#1a1a1a" if is_dark else "#ffffff"
        text_fg = "#ffffff" if is_dark else "#1c1c1c"
        border_color = "#2A2A2A" if is_dark else "#cccccc"
        
        # Border container
        border_frame = tk.Frame(text_frame, bg=border_color, padx=1, pady=1)
        border_frame.grid(row=0, column=0, sticky="nsew")
        border_frame.grid_rowconfigure(0, weight=1)
        border_frame.grid_columnconfigure(0, weight=1)
        
        inner_frame = ttk.Frame(border_frame)
        inner_frame.grid(row=0, column=0, sticky="nsew")
        inner_frame.grid_rowconfigure(0, weight=1)
        inner_frame.grid_columnconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(inner_frame, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.text_input = tk.Text(
            inner_frame, 
            height=12, 
            wrap="word", 
            font=("Segoe UI", 10),
            background=text_bg,
            foreground=text_fg,
            insertbackground=text_fg,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=10,
            pady=8,
            yscrollcommand=scrollbar.set
        )
        self.text_input.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.text_input.yview)
        
        # Bottom button bar (Cancel left, Add Chunk right - like other screens)
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=6, column=0, sticky="ew")
        
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left")
        ttk.Button(btn_frame, text="Add Chunk", command=self._submit, style="Accent.TButton").pack(side="right")
    
    def _filter_papers(self, *args):
        """Filter paper dropdown based on typed text."""
        typed = self.paper_var.get().lower()
        if not typed:
            self.paper_combo['values'] = self.paper_values
        else:
            filtered = [p for p in self.paper_values if typed in p.lower()]
            self.paper_combo['values'] = filtered if filtered else self.paper_values
    
    def _submit(self):
        """Validate and submit the new chunk."""
        selected_sections = [s for s, v in self.section_vars.items() if v.get()]
        text = self.text_input.get("1.0", "end-1c").strip()
        
        if not selected_sections:
            messagebox.showwarning("Missing Info", "Select at least one target section.")
            return
        
        # Find selected paper
        paper_idx = None
        selected_paper_str = self.paper_var.get()
        for i, pv in enumerate(self.paper_values):
            if pv == selected_paper_str:
                paper_idx = i
                break
        
        if paper_idx is None:
            messagebox.showwarning("Missing Info", "Select a source paper.")
            return
        
        if not text:
            messagebox.showwarning("Missing Info", "Enter the evidence text.")
            return
        
        self.on_add(selected_sections, self.papers[paper_idx], text)
        self.destroy()


class EvidenceScreen(BaseFrame):
    """Evidence Manager screen for reviewing and editing evidence chunks."""
    
    def __init__(self, parent, controller):
        self.evidence_by_section: Dict[Section, List[Evidence]] = {}
        self.section_cards: Dict[Section, CollapsibleSectionCard] = {}
        self.papers: List[Paper] = []
        self._loaded = False
        
        # Dynamic button text based on whether paper draft exists
        paper_draft_path = Path("output/paper_draft.md")
        next_text = "Continue" if paper_draft_path.exists() else "Generate Paper Draft"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Evidence Manager",
            has_next=True,
            next_text=next_text,
            has_back=True,
            back_text="Back",
            has_regenerate=True,
            regenerate_text="Regather Evidence",
            # header_file_path=Path(EVIDENCE_FILE) if Path(EVIDENCE_FILE).exists() else None,
        )
    
    def create_content(self):
        """Create the initial UI structure with sticky stats bar."""
        # Create sticky bar above the scrollable content
        # Move the canvas to row 1 and put sticky bar at row 0
        self._canvas.grid_configure(row=1)
        self.content_container.grid_rowconfigure(0, weight=0)  # Stats bar doesn't expand
        self.content_container.grid_rowconfigure(1, weight=1)  # Canvas expands
        
        # Sticky stats bar at row 0 (minimal padding)
        self.sticky_frame = ttk.Frame(self.content_container)
        self.sticky_frame.grid(row=0, column=1, sticky="ew", pady=(0, 12))
        
        # Stats label
        self.stats_label = ttk.Label(
            self.sticky_frame,
            text="",
            font=self.controller.fonts.default_font,
            foreground="gray"
        )
        self.stats_label.pack(side="left")
        
        # Buttons
        btn_frame = ttk.Frame(self.sticky_frame)
        btn_frame.pack(side="right")
        
        ttk.Button(
            btn_frame,
            text="+ Add Chunk",
            command=self._open_add_dialog,
            style="Accent.TButton"
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Expand All",
            command=self._expand_all
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Collapse All",
            command=self._collapse_all
        ).pack(side="left")
        
        # Separator line below sticky bar
        ttk.Separator(self.content_container, orient="horizontal").grid(
            row=0, column=1, sticky="sew", pady=(30, 0)
        )
        
        # Remove default padding from scrollable frame for cleaner layout
        self.scrollable_frame.configure(padding=(0, 12, 0, 10))
    
    def on_show(self):
        """Load evidence when screen is shown."""
        if self._loaded:
            return
        
        self._load_data()
        self._build_sections()
        self._loaded = True
    
    def _load_data(self):
        """Load evidence and papers."""
        try:
            self.evidence_by_section = load_evidence()
        except FileNotFoundError:
            self.evidence_by_section = {}
        
        # Load papers for the add dialog
        try:
            self.papers = LiteratureSearch.load_papers("output/papers.json")
        except:
            self.papers = []
    
    def _build_sections(self):
        """Build the section cards (stats bar is already in sticky frame)."""
        # Section cards go in scrollable_frame
        for section in DISPLAY_SECTIONS:
            card = CollapsibleSectionCard(
                self.scrollable_frame,
                section,
                self.controller,
                on_chunk_removed=self._on_chunk_removed
            )
            card.pack(fill="x", pady=(0, 8))
            self.section_cards[section] = card
            
            # Set evidence
            evidence = self.evidence_by_section.get(section, [])
            card.set_evidence(list(evidence))
        
        self._update_stats()
    
    def _update_stats(self):
        """Update the stats bar."""
        total, papers = get_evidence_stats(self.evidence_by_section)
        self.stats_label.config(
            text=f"Total: {total} chunk{'s' if total != 1 else ''} from {papers} paper{'s' if papers != 1 else ''}"
        )
    
    def _expand_all(self):
        """Expand all section cards."""
        for card in self.section_cards.values():
            card.expand()
    
    def _collapse_all(self):
        """Collapse all section cards."""
        for card in self.section_cards.values():
            card.collapse()
    
    def _open_add_dialog(self):
        """Open the add chunk dialog."""
        if not self.papers:
            messagebox.showwarning("No Papers", "No papers are loaded. Please complete prior steps first.")
            return
        
        AddChunkDialog(
            self.controller,
            self.papers,
            DISPLAY_SECTIONS,
            on_add=self._on_add_chunk
        )
    
    def _on_add_chunk(self, sections: List[Section], paper: Paper, text: str):
        """Handle adding a new chunk."""
        import uuid
        
        for section in sections:
            # Create new Evidence object
            chunk = PaperChunk(
                chunk_id=f"user_{uuid.uuid4().hex[:8]}",
                paper=paper,
                chunk_text=text,  # Use summary as chunk text too
                chunk_index=0,
                embedding=[]
            )
            evidence = Evidence(
                chunk=chunk,
                summary=text,
                vector_score=0.0,
                llm_score=1.0,  # User-provided gets high score
                combined_score=1.0,
                source_query="user_added"
            )
            
            add_evidence(self.evidence_by_section, section, evidence)
            self.section_cards[section].add_chunk(evidence)
        
        self._save_evidence()
        self._update_stats()
        messagebox.showinfo("Success", f"Added chunk to {len(sections)} section(s).")
    
    def _on_chunk_removed(self, section: Section, evidence: Evidence):
        """Handle chunk removal."""
        remove_evidence(self.evidence_by_section, section, evidence.chunk.chunk_id)
        self._save_evidence()
        self._update_stats()
    
    def _save_evidence(self):
        """Save current evidence to file."""
        save_evidence(self.evidence_by_section)
    
    def on_next(self):
        """Save evidence and proceed or generate paper draft."""
        self._save_evidence()
        
        # Check if paper draft already exists
        from pathlib import Path
        paper_draft_path = Path("output/paper_draft.md")
        
        if paper_draft_path.exists():
            # Draft exists - just proceed
            super().on_next()
        else:
            # No draft - need to generate it first
            self._run_paper_generation()
    
    def _run_paper_generation(self):
        """Generate paper draft from evidence, then proceed to Paper Draft screen."""
        import threading
        from phases.context_analysis.paper_conception import PaperConception
        from phases.context_analysis.user_requirements import UserRequirements
        from phases.experimentation.experiment_runner import ExperimentRunner
        from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
        
        popup = ProgressPopup(self.controller, "Generating Paper Draft")
        
        def task():
            try:
                # Load context
                self.after(0, lambda: popup.update_status("Loading context..."))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Load experiment result
                experiment_result_file = "output/experiments/experiment_result.json"
                experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
                
                user_requirements = None
                try:
                    user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                except:
                    pass
                
                # Initialize pipeline
                pipeline = PaperWritingPipeline()
                
                # Write Paper using evidence from evidence.json
                def status_update(msg):
                    self.after(0, lambda: popup.update_status(msg))
                
                self.after(0, lambda: popup.update_status("Writing paper sections..."))
                
                pipeline.write_paper_from_evidence(
                    paper_concept=paper_concept,
                    experiment_result=experiment_result,
                    user_requirements=user_requirements,
                    status_callback=status_update
                )
                
                # Success - close popup and proceed to Paper Draft screen
                self.after(0, lambda: self._on_generation_success(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))

        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def _on_generation_success(self, popup: ProgressPopup):
        """Handle successful paper generation."""
        popup.close()
        self.controller.next_screen()
    
    def reload_content(self):
        """Reload the screen content."""
        self._loaded = False
        # Clear section cards
        self.section_cards.clear()
        super().reload_content()
    
    def on_regenerate(self):
        """Regather evidence from papers."""
        if not messagebox.askyesno(
            "Regather Evidence",
            "This will regather all evidence from your papers. Any manually added chunks will be lost.\n\nDo you want to continue?"
        ):
            return
        
        self._run_evidence_gathering()
    
    def _run_evidence_gathering(self):
        """Gather evidence from papers with progress popup."""
        import threading
        from phases.context_analysis.paper_conception import PaperConception
        from phases.context_analysis.user_requirements import UserRequirements
        from phases.experimentation.experiment_runner import ExperimentRunner
        from phases.paper_search.literature_search import LiteratureSearch
        from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
        from phases.paper_writing.evidence_gatherer import EvidenceGatherer
        from phases.paper_writing.evidence_manager import save_evidence
        from phases.paper_writing.data_models import Section
        from settings import Settings
        
        popup = ProgressPopup(self.controller, "Gathering Evidence")
        
        def task():
            try:
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Load experiment result
                self.after(0, lambda: popup.update_status("Loading experiment results"))
                experiment_result_file = "output/experiments/experiment_result.json"
                experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
                
                # Load papers
                self.after(0, lambda: popup.update_status("Loading indexed papers"))
                papers_with_markdown = LiteratureSearch.load_papers("output/papers.json")
                
                # Load user requirements
                user_requirements = None
                try:
                    user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                except:
                    pass
                
                # Initialize pipeline
                pipeline = PaperWritingPipeline()
                
                # Index papers
                self.after(0, lambda: popup.update_status("Indexing papers for evidence search"))
                pipeline.index_papers(papers_with_markdown)
                
                gatherer = EvidenceGatherer(
                    indexed_corpus=pipeline._indexed_corpus or [],
                )
                
                evidence_by_section = {}
                
                for section_type in (
                    Section.METHODS,
                    Section.RESULTS,
                    Section.DISCUSSION,
                    Section.INTRODUCTION,
                    Section.RELATED_WORK,
                    Section.CONCLUSION,
                ):
                    self.after(0, lambda s=section_type: popup.update_status(f"Gathering evidence for {s.value}"))
                    default_queries = pipeline.query_builder.build_default_queries(
                        section_type, paper_concept, experiment_result
                    )
                    
                    evidence, _ = gatherer.gather_evidence(
                        section_type=section_type,
                        context=paper_concept,
                        experiment=experiment_result,
                        default_queries=default_queries,
                        max_iterations=Settings.EVIDENCE_AGENTIC_ITERATIONS,
                        initial_chunks=Settings.EVIDENCE_INITIAL_CHUNKS,
                        filtered_chunks=Settings.EVIDENCE_FILTERED_CHUNKS,
                        user_requirements=user_requirements,
                    )
                    
                    evidence_by_section[section_type] = evidence
                
                # Save evidence
                self.after(0, lambda: popup.update_status("Saving evidence"))
                save_evidence(evidence_by_section)
                
                # Refresh UI
                self.after(0, lambda: self._on_regenerate_success(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def _on_regenerate_success(self, popup: ProgressPopup):
        """Handle successful evidence regeneration."""
        popup.close()
        # Reload the screen to show new evidence
        self.reload_content()
