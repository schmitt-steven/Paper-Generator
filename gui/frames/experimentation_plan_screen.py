import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

from ..base_frame import BaseFrame, ProgressPopup, create_scrollable_text_area
from utils.file_utils import load_markdown, save_markdown
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.context_analysis.paper_conception import PaperConception
from phases.experimentation.experiment_runner import ExperimentRunner


EXPERIMENTS_DIR = "output/experiments"
EXPERIMENT_PLAN_FILE = "experiment_plan.md"
HYPOTHESES_FILE = "output/hypotheses.json"

class ExperimentationPlanScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.plan_text: tk.Text
        
        # Check if experiment result exists to set button text
        experiment_result_file = Path("output/experiments/experiment_result.json")
        next_text = "Continue" if experiment_result_file.exists() else "Run Experiment"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Experiment Plan",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate",
            header_file_path=Path(EXPERIMENTS_DIR) / EXPERIMENT_PLAN_FILE
        )

    def create_content(self):
        # Info text
        self._create_info_section()
        
        # Don't load plan yet - wait until screen is shown
        # This prevents loading on app startup

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Review and edit the experiment plan below.\n"
            "This plan was generated based on your hypothesis and paper concept.\n"
            "It describes how the experiments will be conducted to test the hypothesis."
        )

        label = ttk.Label(
            explanation_frame,
            text=explanation_text,
            font=self.controller.fonts.default_font,
            foreground="gray",
            justify="left"
        )
        label.pack(anchor="w", fill="x")

        def set_wraplength(event):
            label.config(wraplength=event.width - 10)
        label.bind("<Configure>", set_wraplength)

    def _load_plan(self):
        """Load experiment plan from file and display it."""
        try:
            plan_content = load_markdown(EXPERIMENT_PLAN_FILE, EXPERIMENTS_DIR)
        except FileNotFoundError:
            self._show_error(
                f"Experiment plan not found: {EXPERIMENTS_DIR}/{EXPERIMENT_PLAN_FILE}\n\n"
                "Please complete the previous steps first."
            )
            return
        except Exception as e:
            self._show_error(f"Error loading experiment plan: {e}")
            return
        
        # Create the editable text area
        self._create_plan_section(plan_content)

    def _show_error(self, message: str):
        """Display an error message."""
        error_frame = ttk.Frame(self.scrollable_frame, padding="20")
        error_frame.pack(fill="x", pady=20)
        
        ttk.Label(
            error_frame,
            text=message,
            font=self.controller.fonts.default_font,
            foreground="red",
            wraplength=500
        ).pack()

    def _create_plan_section(self, content: str):
        """Create a labeled section with an editable text area for the plan."""
        # Container
        section_container = ttk.Frame(self.scrollable_frame, padding=(0, 0, 0, 15))
        section_container.pack(fill="both", expand=True)

        # Header
        ttk.Label(
            section_container, 
            text="Description", 
            font=self.controller.fonts.sub_header_font
        ).pack(anchor="w", pady=(0, 10))
        
        container, self.plan_text = create_scrollable_text_area(section_container, height=25)
        container.pack(fill="both", expand=True, padx=(15, 0))
        self.plan_text.insert("1.0", content)

    def _save_plan(self):
        """Save the edited plan."""
        if not hasattr(self, 'plan_text'):
            return
        
        # Get content from text widget
        plan_content = self.plan_text.get("1.0", "end-1c")
        
        try:
            save_markdown(plan_content, EXPERIMENT_PLAN_FILE, EXPERIMENTS_DIR)
            print(f"[ExperimentationPlan] Saved changes to {EXPERIMENTS_DIR}/{EXPERIMENT_PLAN_FILE}")
        except Exception as e:
            print(f"[ExperimentationPlan] Failed to save: {e}")

    def on_next(self):
        """Save the edited plan and proceed or run experiments."""
        # Always save first
        self._save_plan()
        
        # Check if output exists (simplified filename without hypothesis ID)
        experiment_result_file = Path("output/experiments/experiment_result.json")
        if experiment_result_file.exists():
            super().on_next()
        else:
            self._run_generation()

    def _run_generation(self):
        """Run experiment with progress popup."""
        popup = ProgressPopup(self.controller, "Running Experiments")
        
        def task():
            try:
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
                
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Run experiment
                self.after(0, lambda: popup.update_status("Running experiment"))
                experiment_runner = ExperimentRunner()
                result = experiment_runner.run_experiment(
                    selected_hypothesis,
                    paper_concept,
                    load_existing_plan=True,  # Use the plan we just saved
                    load_existing_code=False  # Generate new code
                )
                
                # Continue to next screen regardless of verdict (disproven/inconclusive is still a valid result)
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
        """Called when screen is shown - load plan if not already loaded."""
        # Only load if we haven't loaded yet
        if not hasattr(self, 'plan_text'):
            plan_path = Path(EXPERIMENTS_DIR) / EXPERIMENT_PLAN_FILE
            if plan_path.exists():
                self._load_plan()
