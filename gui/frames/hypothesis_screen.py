import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path
from typing import List, Optional

from ..base_frame import BaseFrame, ProgressPopup, create_styled_text
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.hypothesis_generation.hypothesis_models import Hypothesis
from phases.context_analysis.paper_conception import PaperConception
from phases.experimentation.experiment_runner import ExperimentRunner


HYPOTHESES_FILE = "output/hypotheses.json"

# Output file to check for dynamic button text
EXPERIMENTAL_PLAN_FILE = Path("output/experiments/experimental_plan.md")


class HypothesisScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.hypotheses: List[Hypothesis] = []
        self.current_hypothesis: Optional[Hypothesis] = None
        self.current_hypothesis_index: int = 0
        
        # Text widgets for each field
        self.description_text: tk.Text
        self.rationale_text: tk.Text
        self.methods_text: tk.Text
        self.expected_improvement_text: tk.Text
        self.baseline_text: tk.Text
        
        # Dynamic button text based on whether output file exists
        next_text = "Continue" if EXPERIMENTAL_PLAN_FILE.exists() else "Generate Experiment Plan"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Hypothesis",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate"
        )

    def create_content(self):
        # Info text
        self._create_info_section()
        
        # Don't load hypothesis yet - wait until screen is shown
        # This prevents loading on app startup

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Review and edit the hypothesis below.\n"
            "This hypothesis was generated based on the paper concept and literature analysis.\n"
            "It will be used as the basis for the experimentation phase."
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

    def _load_hypothesis(self):
        """Load hypotheses from file and display the selected one."""
        # Only load if file exists
        if not Path(HYPOTHESES_FILE).exists():
            self._show_error(f"No hypotheses found: {HYPOTHESES_FILE}\n\nPlease complete the previous steps first.")
            return
        
        try:
            self.hypotheses = HypothesisBuilder.load_hypotheses(HYPOTHESES_FILE)
        except Exception as e:
            self._show_error(f"Error loading hypotheses: {e}")
            return
        
        if not self.hypotheses:
            self._show_error(f"No hypotheses found in {HYPOTHESES_FILE}\n\nPlease complete the previous steps first.")
            return
        
        # Find the selected hypothesis, or use the first one
        self.current_hypothesis = None
        self.current_hypothesis_index = 0
        
        for i, hyp in enumerate(self.hypotheses):
            if hyp.selected_for_experimentation:
                self.current_hypothesis = hyp
                self.current_hypothesis_index = i
                break
        
        if self.current_hypothesis is None:
            self.current_hypothesis = self.hypotheses[0]
            self.current_hypothesis_index = 0
        
        # Create the editable sections
        self._create_hypothesis_fields()
    
    def on_show(self):
        """Called when screen is shown - load hypothesis if not already loaded."""
        # Only load if we haven't loaded yet and file exists
        if not hasattr(self, 'current_hypothesis') or self.current_hypothesis is None:
            if Path(HYPOTHESES_FILE).exists():
                self._load_hypothesis()

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

    def _create_hypothesis_fields(self):
        """Create editable fields for the hypothesis."""
        if self.current_hypothesis is None:
            return
        hyp = self.current_hypothesis
        
        self.description_text = self._create_section(
            "Description", 
            hyp.description, 
            height=4
        )
        
        self.rationale_text = self._create_section(
            "Rationale", 
            hyp.rationale, 
            height=4
        )
        
        self.methods_text = self._create_section(
            "Methods", 
            hyp.method_combination, 
            height=3
        )
        
        self.expected_improvement_text = self._create_section(
            "Expected Improvement", 
            hyp.expected_improvement, 
            height=3
        )
        
        self.baseline_text = self._create_section(
            "Baseline to Beat", 
            hyp.baseline_to_beat or "", 
            height=2
        )

    def _create_section(self, title: str, content: str, height: int = 4) -> tk.Text:
        """Create a labeled section with an editable text area."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding="10")
        frame.pack(fill="x", pady=10)
        
        text_widget = create_styled_text(frame, height=height)
        text_widget.pack(fill="x", expand=True)
        text_widget.insert("1.0", content)
        
        return text_widget

    def _save_hypothesis(self) -> Hypothesis | None:
        """Save the edited hypothesis and return it."""
        if self.current_hypothesis is None:
            return None
        
        # Get content from text widgets
        description = self.description_text.get("1.0", "end-1c").strip()
        rationale = self.rationale_text.get("1.0", "end-1c").strip()
        method_combination = self.methods_text.get("1.0", "end-1c").strip()
        expected_improvement = self.expected_improvement_text.get("1.0", "end-1c").strip()
        baseline_to_beat = self.baseline_text.get("1.0", "end-1c").strip() or None
        
        # Update the hypothesis object
        updated_hypothesis = Hypothesis(
            id=self.current_hypothesis.id,
            description=description,
            rationale=rationale,
            method_combination=method_combination,
            expected_improvement=expected_improvement,
            baseline_to_beat=baseline_to_beat,
            selected_for_experimentation=self.current_hypothesis.selected_for_experimentation
        )
        
        # Replace in the list
        self.hypotheses[self.current_hypothesis_index] = updated_hypothesis
        
        # Save all hypotheses back to file
        try:
            HypothesisBuilder.save_hypotheses(self.hypotheses, HYPOTHESES_FILE)
            print(f"[Hypothesis] Saved changes to {HYPOTHESES_FILE}")
        except Exception as e:
            print(f"[Hypothesis] Failed to save: {e}")
        
        return updated_hypothesis

    def on_next(self):
        """Save the edited hypothesis and proceed or generate experiment plan."""
        if self.current_hypothesis is None:
            super().on_next()
            return
        
        # Always save first
        updated_hypothesis = self._save_hypothesis()
        if updated_hypothesis is None:
            super().on_next()
            return
        
        # Check if output exists
        if EXPERIMENTAL_PLAN_FILE.exists():
            super().on_next()
        else:
            self._run_generation(updated_hypothesis)

    def _run_generation(self, hypothesis: Hypothesis):
        """Run experiment plan generation with progress popup."""
        popup = ProgressPopup(self.controller, "Generating Experiment Plan...")
        
        def task():
            try:
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept..."))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Generate experimental plan
                self.after(0, lambda: popup.update_status("Generating experimental plan..."))
                experiment_runner = ExperimentRunner()
                experimental_plan = experiment_runner._generate_experimental_plan(hypothesis, paper_concept)
                experiment_runner.save_experimental_plan(experimental_plan)
                
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
