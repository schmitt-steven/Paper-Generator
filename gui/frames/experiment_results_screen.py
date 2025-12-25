import tkinter as tk
import traceback
from tkinter import ttk
import threading
from pathlib import Path
from settings import Settings
from ..base_frame import BaseFrame, ProgressPopup, TextBorderFrame
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
        """Create section to display generated plots."""
        import os
        from PIL import Image, ImageTk
        
        plots_dir = Path("output/experiments/plots")
        search_dir = Path("output/experiments/plots")
        plot_files = []
        if search_dir.exists():
            plot_files.extend(list(search_dir.glob("*.png")))
            plot_files.extend(list(search_dir.glob("*.jpg")))
        
        if not plot_files:
            return

        plots_container = self.create_card_frame(self.results_container, "Generated Plots")


        for plot_path in plot_files:
            try:
                # Open and resize image
                pil_img = Image.open(plot_path)
                # Max width 600, keep aspect ratio
                width = 600
                w_percent = (width / float(pil_img.size[0]))
                h_size = int((float(pil_img.size[1]) * float(w_percent)))
                pil_img = pil_img.resize((width, h_size), Image.Resampling.LANCZOS)
                
                tk_img = ImageTk.PhotoImage(pil_img)
                
                img_label = ttk.Label(plots_container, image=tk_img)
                img_label.image = tk_img # Keep reference!
                img_label.pack(pady=5, padx=(5, 0))
                
                name_label = ttk.Label(plots_container, text=plot_path.name, font=self.controller.fonts.small_font)
                name_label.pack(pady=(0, 15), padx=(5, 0))
            except Exception as e:
                print(f"Error loading plot {plot_path}: {e}")

    def _create_code_section(self):
        """Create section to view/edit experiment code."""
        
        section_container = ttk.Frame(self.results_container, padding=(0, 0, 0, 15))
        section_container.pack(fill="both", expand=True)

        ttk.Label(
            section_container, 
            text="Experiment Code", 
            font=self.controller.fonts.sub_header_font
        ).pack(anchor="w", pady=(0, 10))
        
        # Container with border (simulated)
        # We use TextBorderFrame corresponding to our theming strategy
        border_container = TextBorderFrame(section_container, padx=1, pady=1)
        border_container.pack(fill="both", expand=True)

        # Container for text + scrollbars
        editor_container = ttk.Frame(border_container)
        editor_container.pack(fill="both", expand=True)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(editor_container, orient="vertical")
        h_scroll = ttk.Scrollbar(editor_container, orient="horizontal")
        
        self.code_editor = tk.Text(
            editor_container, 
            height=35,  # Fixed height (increased)
            font=self.controller.fonts.code_font,
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            wrap="none",
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        
        v_scroll.config(command=self.code_editor.yview)
        h_scroll.config(command=self.code_editor.xview)
        
        # Grid layout for scrollbars
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.code_editor.pack(side="left", fill="both", expand=True)
        
        # Load code
        code_path = Path("output/experiments/experiment.py")
        if not code_path.exists():
            # Fallback for the custom script if it exists there
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
             
        # Bind mousewheel for standard vertical scrolling
        def on_text_mousewheel(event):
            # Allow default scrolling behavior since we are scrolling inside the fixed area now
            # Tkinter Text widget handles this usually, but sometimes needs help on Linux/Windows mix.
            # For now, let's remove the 'return' block that prevented it.
            pass
             
        # self.code_editor.bind("<MouseWheel>", on_text_mousewheel) # Rely on default bindings

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

    def on_next(self):
        """Proceed or gather evidence."""
        # Save code first
        self.save_code()
        
        # Check if evidence already gathered
        from phases.paper_writing.evidence_manager import EVIDENCE_FILE
        if Path(EVIDENCE_FILE).exists():
            super().on_next()
        else:
            self._run_generation()

    def on_regenerate(self):
        """Show popup to choose regeneration method."""
        # Use native dialog to choose action
        # Yes = Restart from experiment plan
        # No = Run current code
        # Cancel = Do nothing
        choice = tk.messagebox.askyesnocancel(
            "Regenerate Experiment",
            "Do you want to generate new experiment results?\n\n"
            "Click 'Yes' to implement the experiment plan again, "
            "click 'No' to only execute the current experiment code."
        )
        
        if choice is True:
            # Yes -> Restart plan
            self.controller.show_frame("ExperimentationPlanScreen")
        elif choice is False:
            # No -> Run current code
            self._rerun_experiment_code()
        # None -> Cancel (do nothing)

    def _rerun_experiment_code(self):
        """Re-run the experiment using the current (potentially edited) code."""
        self.save_code()
        
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

