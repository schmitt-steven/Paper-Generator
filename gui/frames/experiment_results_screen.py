import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

from ..base_frame import BaseFrame, ProgressPopup
from phases.context_analysis.paper_conception import PaperConception
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
            self._show_error(f"Error loading experiment results: {e}")

    def _show_error(self, message: str):
        """Display an error message."""
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

        # Hypothesis description
        hyp_frame = ttk.LabelFrame(self.scrollable_frame, text="Hypothesis", padding="10")
        hyp_frame.pack(fill="x", pady=10)
        
        ttk.Label(
            hyp_frame,
            text=experiment_result.hypothesis.description,
            font=("SF Pro", 14),
            wraplength=600,
            justify="left"
        ).pack(anchor="w")

        # Verdict section
        verdict_frame = ttk.LabelFrame(self.scrollable_frame, text="Verdict", padding="10")
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

    def on_next(self):
        """Proceed or generate paper draft."""
        if PAPER_DRAFT_FILE.exists():
            super().on_next()
        else:
            self._run_generation()

    def _run_generation(self):
        """Run paper draft generation with progress popup."""
        popup = ProgressPopup(self.controller, "Generating Paper Draft...")
        
        def task():
            try:
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept..."))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Load hypothesis
                self.after(0, lambda: popup.update_status("Loading hypothesis..."))
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
                self.after(0, lambda: popup.update_status("Loading experiment results..."))
                experiment_result_file = "output/experiments/experiment_result.json"
                experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
                
                # Load papers with markdown
                self.after(0, lambda: popup.update_status("Loading indexed papers..."))
                papers_with_markdown = LiteratureSearch.load_papers("output/papers.json")
                
                # Write paper
                self.after(0, lambda: popup.update_status("Writing paper sections..."))
                paper_writing_pipeline = PaperWritingPipeline()
                paper_writing_pipeline.write_paper(
                    paper_concept=paper_concept,
                    experiment_result=experiment_result,
                    papers=papers_with_markdown,
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

