import tkinter as tk
import traceback
from tkinter import ttk
import threading
from pathlib import Path
from settings import Settings
from ..base_frame import BaseFrame, ProgressPopup
from phases.context_analysis.paper_conception import PaperConception
from phases.context_analysis.user_requirements import UserRequirements
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline


PAPER_DRAFT_FILE = Path("output/paper_draft.md")
HYPOTHESES_FILE = "output/hypotheses.json"


class ExperimentResultsScreen(BaseFrame):
    def __init__(self, parent, controller):
        # Dynamic button text based on whether output file exists
        next_text = "Continue" if PAPER_DRAFT_FILE.exists() else "Generate Paper Draft"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Experiment Results",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate"
        )

    def create_content(self):
        """Create the experiment results display."""
        self._create_info_section()
        self.results_container = ttk.Frame(self.scrollable_frame)
        self.results_container.pack(fill="x", expand=True)

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Review the experiment results below.\n"
            "These results show how the hypothesis was tested and the outcome.\n"
            "The paper draft will be generated based on these results."
        )

        label = ttk.Label(
            explanation_frame,
            text=explanation_text,
            font=("SF Pro", 14),
            foreground="gray",
            justify="left"
        )
        label.pack(anchor="w", fill="x")

        def set_wraplength(event):
            label.config(wraplength=event.width - 10)
        label.bind("<Configure>", set_wraplength)

    def _load_and_display_results(self):
        """Load and display experiment results."""
        if not Path(HYPOTHESES_FILE).exists():
            self._show_error(f"Hypotheses file not found: {HYPOTHESES_FILE}")
            return
        
        try:
            # Get selected hypothesis
            hypotheses = HypothesisBuilder.load_hypotheses(HYPOTHESES_FILE)
            selected_hypothesis = None
            for hyp in hypotheses:
                if hyp.selected_for_experimentation:
                    selected_hypothesis = hyp
                    break
            if selected_hypothesis is None and hypotheses:
                selected_hypothesis = hypotheses[0]
            
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
            font=("SF Pro", 14),
            foreground="red",
            wraplength=500
        ).pack()

    def _create_results_display(self, experiment_result):
        """Create the results display."""
        
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()

        # Hypothesis description
        hyp_frame = ttk.LabelFrame(self.results_container, text="Hypothesis", padding="10")
        hyp_frame.pack(fill="x", pady=10)
        
        ttk.Label(
            hyp_frame,
            text=experiment_result.hypothesis.description,
            font=("SF Pro", 14),
            wraplength=600,
            justify="left"
        ).pack(anchor="w")

        # Verdict section
        verdict_frame = ttk.LabelFrame(self.results_container, text="Verdict", padding="10")
        verdict_frame.pack(fill="x", pady=10)
        verdict = experiment_result.hypothesis_evaluation.verdict
        verdict_color = "green" if verdict.lower() == "proven" else ("red" if verdict.lower() == "disproven" else "orange")
        ttk.Label(
            verdict_frame,
            text=verdict.upper(),
            font=("SF Pro", 18, "bold"),
            foreground=verdict_color
        ).pack(anchor="w")
        
        # Reasoning
        reasoning_label = ttk.Label(
            verdict_frame,
            text=experiment_result.hypothesis_evaluation.reasoning,
            font=("SF Pro", 14),
            wraplength=600,
            justify="left"
        )
        reasoning_label.pack(anchor="w", pady=(10, 0))

        # --- Plots Section ---
        self._create_plots_section()

        # --- Experiment Code Section ---
        self._create_code_section()

    def _create_plots_section(self):
        """Create section to display generated plots."""
        import os
        from PIL import Image, ImageTk
        
        plots_dir = Path("output/experiments/plots")
        # Also check root output/experiments for direct file saves if not in subfolder
        # But experiment runner specifically puts them in output/experiments/plots usually, OR in base dir.
        # Let's check both for safety or just the base dir if that's where they are.
        # Based on previous tool outputs, plots are in output/experiments/comparison_plot.png etc.
        # Actually in this specific comparison script case, they are in the CWD (Paper-Generator).
        # But ideally the runner moves them? 
        # The runner executes in base_output_dir. Let's look in output/experiments first.
        
        search_dir = Path("output/experiments/plots")
        plot_files = []
        if search_dir.exists():
            plot_files.extend(list(search_dir.glob("*.png")))
            plot_files.extend(list(search_dir.glob("*.jpg")))
        
        if not plot_files:
            return

        plots_frame = ttk.LabelFrame(self.results_container, text="Generated Plots", padding="10")
        plots_frame.pack(fill="x", pady=10)

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
                
                img_label = ttk.Label(plots_frame, image=tk_img)
                img_label.image = tk_img # Keep reference!
                img_label.pack(pady=5)
                
                name_label = ttk.Label(plots_frame, text=plot_path.name, font=("SF Pro", 10))
                name_label.pack(pady=(0, 15))
            except Exception as e:
                print(f"Error loading plot {plot_path}: {e}")

    def _create_code_section(self):
        """Create section to view/edit experiment code."""
        from tkinter.scrolledtext import ScrolledText
        
        code_frame = ttk.LabelFrame(self.results_container, text="Experiment Code", padding="10")
        code_frame.pack(fill="both", expand=True, pady=10)
        
        self.code_editor = ScrolledText(code_frame, height=20, font=("Menlo", 12))
        self.code_editor.pack(fill="both", expand=True)
        
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
        """Proceed or generate paper draft."""
        # Save code first
        self.save_code()
        
        if PAPER_DRAFT_FILE.exists():
            super().on_next()
        else:
            self._run_generation()

    def on_regenerate(self):
        """Show popup to choose regeneration method."""
        # Create a custom popup window
        popup = tk.Toplevel(self)
        popup.title("Regenerate Experiment")
        popup.geometry("500x300")
        
        # Center popup
        x = self.controller.winfo_x() + (self.controller.winfo_width() // 2) - 250
        y = self.controller.winfo_y() + (self.controller.winfo_height() // 2) - 150
        popup.geometry(f"+{x}+{y}")
        
        ttk.Label(popup, text="Choose how to regenerate the experiment.", font=("SF Pro", 14)).pack(pady=30)
        
        def run_new_plan():
            popup.destroy()
            self.controller.show_frame("ExperimentationPlanScreen")
            
        def rerun_code():
            popup.destroy()
            self._rerun_experiment_code()
            
        # Buttons with fixed width or internal padding, not filling X
        ttk.Button(popup, text="Restart from experiment plan", command=run_new_plan).pack(pady=10, ipadx=20)
        ttk.Button(popup, text="Run current code", command=rerun_code).pack(pady=10, ipadx=20)
        ttk.Button(popup, text="Cancel", command=popup.destroy).pack(pady=30, ipadx=20)

    def _rerun_experiment_code(self):
        """Re-run the experiment using the current (potentially edited) code."""
        self.save_code()
        
        # Use existing loading logic logic but trigger re-run
        popup = ProgressPopup(self.controller, "Re-running Experiment")
        
        def task():
            try:
                self.after(0, lambda: popup.update_status("Loading resources..."))
                
                # Load necessary objects
                hypotheses = HypothesisBuilder.load_hypotheses(HYPOTHESES_FILE)
                selected_hypothesis = next((h for h in hypotheses if h.selected_for_experimentation), hypotheses[0] if hypotheses else None)
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
        """Run paper draft generation with progress popup."""
        popup = ProgressPopup(self.controller, "Generating Paper Draft")
        
        def task():
            try:
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Load hypothesis
                self.after(0, lambda: popup.update_status("Loading hypothesis"))
                hypotheses = HypothesisBuilder.load_hypotheses(HYPOTHESES_FILE)
                selected_hypothesis = None
                for hyp in hypotheses:
                    if hyp.selected_for_experimentation:
                        selected_hypothesis = hyp
                        break
                if selected_hypothesis is None and hypotheses:
                    selected_hypothesis = hypotheses[0]
                
                if selected_hypothesis is None:
                    raise ValueError("No hypothesis found")
                
                # Load experiment result (simplified filename without hypothesis ID)
                self.after(0, lambda: popup.update_status("Loading experiment results"))
                experiment_result_file = "output/experiments/experiment_result.json"
                experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
                
                # Load papers with markdown
                self.after(0, lambda: popup.update_status("Loading indexed papers"))
                papers_with_markdown = LiteratureSearch.load_papers("output/papers.json")
                
                # Load user requirements
                self.after(0, lambda: popup.update_status("Loading user requirements"))
                user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                
                # Write paper
                self.after(0, lambda: popup.update_status("Writing paper sections"))
                paper_writing_pipeline = PaperWritingPipeline()
                def status_update(msg):
                    self.after(0, lambda: popup.update_status(msg))

                paper_writing_pipeline.write_paper(
                    paper_concept=paper_concept,
                    experiment_result=experiment_result,
                    papers=papers_with_markdown,
                    user_requirements=user_requirements,
                    status_callback=status_update
                )
                
                # Success - close popup and proceed
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

