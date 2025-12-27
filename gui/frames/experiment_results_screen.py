import tkinter as tk
import traceback
from tkinter import ttk
import threading
import os
import subprocess
import platform
from pathlib import Path
from settings import Settings
from ..base_frame import BaseFrame, ProgressPopup, TextBorderFrame, create_scrollable_text_area
from phases.context_analysis.paper_conception import PaperConception
from phases.context_analysis.user_requirements import UserRequirements
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline


PAPER_DRAFT_FILE = Path("output/paper_draft.md")
HYPOTHESES_FILE = "output/hypothesis.md"


class ExperimentResultsScreen(BaseFrame):
    def __init__(self, parent, controller):
        # Dynamic button text based on whether evidence file exists
        from phases.paper_writing.evidence_manager import EVIDENCE_FILE
        next_text = "Continue" if Path(EVIDENCE_FILE).exists() else "Gather Evidence"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Experiment Results",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate",
            header_file_path=Path("output/experiments/experiment_result.json")
        )

    def create_content(self):
        """Create the experiment results display."""
        self.results_container = ttk.Frame(self.scrollable_frame)
        self.results_container.pack(fill="x", expand=True)

    def _load_and_display_results(self):
        """Load and display experiment results."""
        if not Path(HYPOTHESES_FILE).exists():
            self._show_error(f"Hypotheses file not found: {HYPOTHESES_FILE}")
            return
        
        try:
            # Get selected hypothesis
            # Get selected hypothesis
            selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
            
            if selected_hypothesis is None:
                self._show_error("No hypothesis found")
                return
            
            # Load experiment result
            experiment_result_file = Path("output/experiments/experiment_result.json")
            if not experiment_result_file.exists():
                self._show_error(f"Experiment result not found: {experiment_result_file}")
                return
            
            experiment_result = ExperimentRunner.load_experiment_result(str(experiment_result_file))
            
            # Display results
            self._create_results_display(experiment_result)
            
        except Exception as e:     
            # Print full error to console for debugging
            print(f"\n[ERROR] Failed to load experiment results:")
            traceback.print_exc()
            # Show simplified error in GUI
            self._show_error(f"Error loading experiment results: {e}")

    def _show_error(self, message: str):
        """Display an error message."""
        # Also print to console
        print(f"{message}")
        
        error_frame = ttk.Frame(self.scrollable_frame, padding="20")
        error_frame.pack(fill="x", pady=20)
        
        ttk.Label(
            error_frame,
            text=message,
            font=self.controller.fonts.default_font,
            foreground="red",
            wraplength=500
        ).pack()

    def _create_results_display(self, experiment_result):
        """Create the results display."""
        
        # Store experiment result for later access (e.g., saving captions)
        self.experiment_result = experiment_result
        self.caption_editors = {}  # Map plot filename to text widget
        
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()

        # Hypothesis description
        # Hypothesis description
        hyp_container = self.create_card_frame(self.results_container, "Hypothesis")


        hyp_label = ttk.Label(
            hyp_container,
            text=experiment_result.hypothesis.description,
            font=self.controller.fonts.text_area_font,
            justify="left"
        )
        hyp_label.pack(anchor="w", fill="x", padx=(5, 0))
        
        def update_hyp_wrap(event):
            hyp_label.config(wraplength=event.width - 20)
        hyp_container.bind("<Configure>", update_hyp_wrap)

        # Verdict section
        # Verdict section
        verdict_container = self.create_card_frame(self.results_container, "Verdict")

        
        verdict = experiment_result.hypothesis_evaluation.verdict
        verdict_color = "green" if verdict.lower() == "proven" else ("red" if verdict.lower() == "disproven" else "orange")
        ttk.Label(
            verdict_container,
            text=verdict.upper(),
            font=self.controller.fonts.sub_header_font,
            foreground=verdict_color
        ).pack(anchor="w", padx=(5, 0))
        
        # Reasoning
        reasoning_label = ttk.Label(
            verdict_container,
            text=experiment_result.hypothesis_evaluation.reasoning,
            font=self.controller.fonts.text_area_font,
            justify="left"
        )
        reasoning_label.pack(anchor="w", pady=(5, 0), fill="x", padx=(5, 0))
        
        def update_reasoning_wrap(event):
             reasoning_label.config(wraplength=event.width - 20)
        verdict_container.bind("<Configure>", update_reasoning_wrap)


        # --- Plots Section ---
        self._create_plots_section()

        # --- Experiment Code Section ---
        self._create_code_section()

    def _create_plots_section(self):
        """Create section to display generated plots with editable captions."""
        from PIL import Image, ImageTk
        
        # Use plots from experiment result (contains filename and caption)
        if not hasattr(self, 'experiment_result') or not self.experiment_result.plots:
            return
        
        # Store figure containers for potential removal
        self.figure_containers = {}
        
        for idx, plot in enumerate(self.experiment_result.plots, start=1):
            plot_path = Path(plot.filename)
            
            # Ensure the path exists (could be relative or absolute)
            if not plot_path.exists():
                # Try relative to project root
                plot_path = Path("output/experiments/plots") / plot_path.name
                if not plot_path.exists():
                    print(f"Plot file not found: {plot.filename}")
                    continue
            
            try:
                # --- Create Figure Card ---
                figure_card = self._create_figure_card(
                    self.results_container,
                    f"Figure {idx}",
                    plot.filename
                )
                self.figure_containers[plot.filename] = figure_card
                
                # Get the content frame from the card
                content_frame = figure_card.winfo_children()[-1]  # Last child is content frame
                
                # --- Plot Image ---
                pil_img = Image.open(plot_path)
                # Max width 600, keep aspect ratio
                width = 600
                w_percent = (width / float(pil_img.size[0]))
                h_size = int((float(pil_img.size[1]) * float(w_percent)))
                pil_img = pil_img.resize((width, h_size), Image.Resampling.LANCZOS)
                
                tk_img = ImageTk.PhotoImage(pil_img)
                
                img_label = ttk.Label(content_frame, image=tk_img)
                img_label.image = tk_img  # Keep reference!
                img_label.pack(pady=(10, 10), padx=5)
                
                # --- Caption Section ---
                # Editable caption using consistent styling
                caption_container, caption_text = create_scrollable_text_area(
                    content_frame, 
                    height=6,
                    font=self.controller.fonts.text_area_font
                )
                caption_container.pack(fill="x", padx=5, pady=(0, 5))
                caption_text.insert("1.0", plot.caption or "")
                
                # Store reference for saving later
                self.caption_editors[plot.filename] = caption_text
                
            except Exception as e:
                print(f"Error loading plot {plot_path}: {e}")
    
    def _create_figure_card(self, parent, title: str, plot_filename: str):
        """Create a card frame for a figure with a delete button in the header."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=1)
        card.pack(fill="x", padx=0, pady=10)
        
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        # Title on the left
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        tk.Label(
            header, 
            text=title, 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Delete button on right (minimalistic with hover effect)
        delete_color = "#666666" if self.controller.current_theme == "dark" else "#888888"
        hover_color = "#ff6b6b" if self.controller.current_theme == "dark" else "#e05555"
        
        delete_btn = tk.Label(
            header,
            text="✕",
            font=("Segoe UI", 14),
            bg=header_bg,
            fg=delete_color,
            cursor="hand2"
        )
        delete_btn.pack(side="right", padx=(0, 5))
        
        # Hover effects
        delete_btn.bind("<Enter>", lambda e: delete_btn.config(fg=hover_color))
        delete_btn.bind("<Leave>", lambda e: delete_btn.config(fg=delete_color))
        delete_btn.bind("<Button-1>", lambda e, fn=plot_filename, t=title: self._delete_figure(fn, t))
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        content = ttk.Frame(card, padding=10)
        content.pack(fill="x")
        
        return card
    
    def _delete_figure(self, plot_filename: str, title: str):
        """Delete a figure after confirmation."""
        confirm = tk.messagebox.askyesno(
            "Delete Figure",
            f"Are you sure you want to delete {title}?"
        )
        
        if confirm:
            # Remove from experiment result
            self.experiment_result.plots = [
                p for p in self.experiment_result.plots if p.filename != plot_filename
            ]
            
            # Remove from caption editors
            if plot_filename in self.caption_editors:
                del self.caption_editors[plot_filename]
            
            # Remove the UI card
            if plot_filename in self.figure_containers:
                self.figure_containers[plot_filename].destroy()
                del self.figure_containers[plot_filename]
            
            # Save the updated experiment result
            self._save_experiment_result()
            print(f"Deleted {title}")

    def _create_code_section(self):
        """Create section to view/edit experiment code (styled as a card)."""
        
        # === Card container ===
        card = ttk.Frame(self.results_container, style="Card.TFrame", padding=1)
        card.pack(fill="both", expand=True, pady=10)
        
        # === Header with title and buttons ===
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        # Title on left
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        tk.Label(
            header, 
            text="Experiment Code", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Buttons on right
        btn_frame = ttk.Frame(header, style="CardHeader.TFrame")
        btn_frame.pack(side="right")
        
        ttk.Button(
            btn_frame, 
            text="Execute", 
            command=self._execute_code
        ).pack(side="right", padx=(5, 0))
        
        ttk.Button(
            btn_frame, 
            text="Open in Editor", 
            command=self._open_code_in_editor
        ).pack(side="right")
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # === Content: Code editor ===
        content = ttk.Frame(card, padding=0)
        content.pack(fill="both", expand=True)

        # Container for text + scrollbars (no border)
        editor_container = ttk.Frame(content)
        editor_container.pack(fill="both", expand=True)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(editor_container, orient="vertical")
        h_scroll = ttk.Scrollbar(editor_container, orient="horizontal")
        
        self.code_editor = tk.Text(
            editor_container, 
            height=30,
            font=self.controller.fonts.code_font,
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            wrap="none",
            padx=10,
            pady=5,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        
        v_scroll.config(command=self.code_editor.yview)
        h_scroll.config(command=self.code_editor.xview)
        
        # Layout
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.code_editor.pack(side="left", fill="both", expand=True)
        
        # Load code
        code_path = Path("output/experiments/experiment.py")
        if not code_path.exists():
            if Path("rbql_vs_q_gemini.py").exists():
                code_path = Path("rbql_vs_q_gemini.py")
        
        if code_path.exists():
            try:
                with open(code_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                self.code_editor.insert("1.0", code_content)
                self.current_code_path = code_path
            except Exception as e:
                self.code_editor.insert("1.0", f"# Error loading code: {e}")
        else:
            self.code_editor.insert("1.0", "# No experiment code found.")

    def _open_code_in_editor(self):
        """Open the experiment code file in the system's default editor."""
        if not hasattr(self, 'current_code_path') or not self.current_code_path:
            tk.messagebox.showwarning("No Code", "No experiment code file found.")
            return
        
        path = str(self.current_code_path.absolute())
        print(f"Opening {path} in editor...")
        
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':
                subprocess.call(('open', path))
            else:
                subprocess.call(('xdg-open', path))
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not open file: {e}")

    def _execute_code(self):
        """Execute the experiment code (save and run)."""
        if not tk.messagebox.askyesno(
            "Execute Code",
            "Do you want to execute the experiment code?\n\n"
            "This will save and run the code, then refresh the results."
        ):
            return
        
        # First save any edits
        self.save_code()
        # Then run the code
        self._rerun_experiment_code()

    def save_code(self):
        """Save the code from editor to file."""
        if hasattr(self, 'current_code_path') and self.current_code_path:
            try:
                content = self.code_editor.get("1.0", "end-1c")
                with open(self.current_code_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Saved code to {self.current_code_path}")
            except Exception as e:
                print(f"Error saving code: {e}")
    
    def _save_captions(self):
        """Save captions from editors back to experiment result and file."""
        if not hasattr(self, 'experiment_result') or not hasattr(self, 'caption_editors'):
            return
        
        try:
            # Update caption values from text editors
            for plot in self.experiment_result.plots:
                if plot.filename in self.caption_editors:
                    editor = self.caption_editors[plot.filename]
                    plot.caption = editor.get("1.0", "end-1c").strip()
            
            # Save updated experiment result to JSON
            self._save_experiment_result()
            print("Saved plot captions")
        except Exception as e:
            print(f"Error saving captions: {e}")
    
    def _save_experiment_result(self):
        """Save the experiment result back to JSON file."""
        if not hasattr(self, 'experiment_result'):
            return
        
        try:
            # Save using ExperimentRunner
            runner = ExperimentRunner()
            runner.save_experiment_result(self.experiment_result)
        except Exception as e:
            print(f"Error saving experiment result: {e}")

    def on_next(self):
        """Proceed or gather evidence."""
        # Save code and captions first
        self.save_code()
        self._save_captions()
        
        # Check if evidence already gathered
        from phases.paper_writing.evidence_manager import EVIDENCE_FILE
        if Path(EVIDENCE_FILE).exists():
            super().on_next()
        else:
            self._run_generation()

    def on_regenerate(self):
        """Regenerate the experiment code in-place."""
        if not tk.messagebox.askyesno(
            "Regenerate Experiment",
            "Do you want to regenerate the experiment?\n\n"
            "This will re-implement the experiment code based on the plan and run it."
        ):
            return
        
        self._save_captions()
        
        popup = ProgressPopup(self.controller, "Regenerating Experiment")
        
        def task():
            try:
                self.after(0, lambda: popup.update_status("Loading resources..."))
                
                selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                runner = ExperimentRunner()
                
                self.after(0, lambda: popup.update_status("Generating new code..."))
                result = runner.run_experiment(
                    hypothesis=selected_hypothesis, 
                    paper_concept=paper_concept,
                    load_existing_plan=True,
                    load_existing_code=False  # Generate NEW code from plan
                )
                
                # Reload screen results
                self.after(0, lambda: self._on_rerun_success(popup))
                
            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
                
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _rerun_experiment_code(self):
        """Re-run the experiment using the current (potentially edited) code."""
        self.save_code()
        self._save_captions()
        
        # Use existing loading logic logic but trigger re-run
        popup = ProgressPopup(self.controller, "Re-running Experiment")
        
        def task():
            try:
                self.after(0, lambda: popup.update_status("Loading resources..."))
                
                # Load necessary objects
                # Load necessary objects
                selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Check specifics for manual file override (hacky but needed for this specific task context)
                # If we are viewing rbql_vs_q_gemini.py results, we might not have a formal 'experiment result' structure 
                # fully aligned if it was run manually. But let's assume standard flow for the runner.
                
                runner = ExperimentRunner()
                
                self.after(0, lambda: popup.update_status("Executing code..."))
                result = runner.run_experiment(
                    hypothesis=selected_hypothesis, 
                    paper_concept=paper_concept,
                    load_existing_plan=True, 
                    load_existing_code=True # CRITICAL FLAG
                )
                
                # Reload screen results
                self.after(0, lambda: self._on_rerun_success(popup))
                
            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
                
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_rerun_success(self, popup):
        popup.close()
        # Force reload of results
        self._load_and_display_results()

    def _run_generation(self):
        """Gather evidence for paper writing with progress popup."""
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
                
                # Load papers with markdown
                self.after(0, lambda: popup.update_status("Loading indexed papers"))
                papers_with_markdown = LiteratureSearch.load_papers("output/papers.json")
                
                # Load user requirements
                user_requirements = None
                try:
                    user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                except:
                    pass
                
                # Initialize pipeline and gather evidence
                # This only gathers evidence and saves to evidence.json
                # Paper writing happens later on Paper Draft screen
                paper_writing_pipeline = PaperWritingPipeline()
                
                def status_update(msg):
                    self.after(0, lambda: popup.update_status(msg))
                
                # Index papers first
                self.after(0, lambda: popup.update_status("Indexing papers for evidence search"))
                paper_writing_pipeline.index_papers(papers_with_markdown)
                
                # Gather evidence for each section
                from phases.paper_writing.evidence_gatherer import EvidenceGatherer
                from phases.paper_writing.evidence_manager import save_evidence
                from phases.paper_writing.data_models import Section
                from settings import Settings
                
                gatherer = EvidenceGatherer(
                    indexed_corpus=paper_writing_pipeline._indexed_corpus or [],
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
                    default_queries = paper_writing_pipeline.query_builder.build_default_queries(
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
                
                # Save evidence for Evidence Manager screen
                self.after(0, lambda: popup.update_status("Saving evidence"))
                save_evidence(evidence_by_section)
                
                # Success - proceed to Evidence Screen (not Paper Draft)
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
    
    def on_show(self):
        """Called when screen is shown - load results if not already loaded."""
        # Only load if we haven't loaded yet
        if not hasattr(self, '_results_loaded') or not self._results_loaded:
            self._load_and_display_results()
            self._results_loaded = True

