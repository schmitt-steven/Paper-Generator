import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

from ..base_frame import BaseFrame, ProgressPopup, create_scrollable_text_area
from utils.file_utils import load_markdown, save_markdown
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.context_analysis.paper_conception import PaperConception
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.context_analysis.user_requirements import UserRequirements
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from settings import Settings


EXPERIMENTS_DIR = "output/experiments"
EXPERIMENT_PLAN_FILE = "experiment_plan.md"
HYPOTHESES_FILE = "output/hypothesis.md"

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
        pass

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
        container.pack(fill="both", expand=True)
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
                # Load hypothesis
                self.after(0, lambda: popup.update_status("Loading hypothesis"))
                selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)

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

    def on_regenerate(self):
        """Regenerate the experiment plan from scratch."""
        if not tk.messagebox.askyesno("Confirm Regeneration", 
                                      "This will create a completely new experiment plan based on your hypothesis and code, overwriting the current one.\n\nDo you want to continue?"):
            return

        popup = ProgressPopup(self.controller, "Regenerating Experiment Plan")
        
        def task():
            try:
                # 1. Load context
                self.after(0, lambda: popup.update_status("Loading context..."))
                selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
                if selected_hypothesis is None:
                    raise ValueError("No hypothesis found")
                
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                user_requirements = None
                try:
                    user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                except:
                    pass
                
                user_code = None
                try:
                    code_analyzer = CodeAnalyzer(model_name=Settings.CODE_ANALYSIS_MODEL)
                    user_code = code_analyzer.load_code_files("user_files")
                except:
                    pass
                
                # 2. Generate Plan
                self.after(0, lambda: popup.update_status("Generating new plan..."))
                experiment_runner = ExperimentRunner()
                
                experiment_plan = experiment_runner._generate_experiment_plan(
                    selected_hypothesis, 
                    paper_concept,
                    user_requirements=user_requirements,
                    user_code=user_code
                )
                
                # 3. Save Plan
                experiment_runner.save_experiment_plan(experiment_plan)
                
                # 4. Reload UI
                self.after(0, lambda: self._on_regeneration_complete(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_regeneration_complete(self, popup: ProgressPopup):
        """Handle regeneration completion."""
        popup.close()
        
        # Refresh text widget
        try:
             content = load_markdown(EXPERIMENT_PLAN_FILE, EXPERIMENTS_DIR)
             self.plan_text.delete("1.0", "end")
             self.plan_text.insert("1.0", content)
        except Exception as e:
             print(f"Error refreshing plan text: {e}")
