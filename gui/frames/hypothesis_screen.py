import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path
from typing import List, Optional

from ..base_frame import BaseFrame, ProgressPopup, create_text_area
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.hypothesis_generation.hypothesis_models import Hypothesis
from phases.context_analysis.paper_conception import PaperConception
from phases.experimentation.experiment_runner import ExperimentRunner


HYPOTHESES_FILE = "output/hypotheses.json"

# Output file to check for dynamic button text
EXPERIMENT_PLAN_FILE = Path("output/experiments/experiment_plan.md")


class HypothesisScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.hypotheses: list[Hypothesis] = []
        self.current_hypothesis: Optional[Hypothesis] = None
        self.current_hypothesis_index: int = 0
        
        # Text widgets for each field
        self.description_text: tk.Text
        self.rationale_text: tk.Text
        self.success_criteria_text: tk.Text
        
        # Dynamic button text based on whether output file exists
        next_text = "Continue" if EXPERIMENT_PLAN_FILE.exists() else "Generate Experiment Plan"
        
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
            "Review the hypothesis below.\n"
            "You can modify the generated hypothesis or create a completely new one.\n"
            "It will be used as the basis for experimentation."
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

    def _load_hypothesis(self):
        """Load hypotheses from file or create empty one for manual entry."""
        # Try to load from file if it exists
        if Path(HYPOTHESES_FILE).exists():
            try:
                self.hypotheses = HypothesisBuilder.load_hypotheses(HYPOTHESES_FILE)
                if self.hypotheses:
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
                    return
            except Exception as e:
                print(f"Error loading hypotheses: {e}")
        
        # No file or empty file - create empty hypothesis for manual entry
        self._create_empty_hypothesis()
    
    def _create_empty_hypothesis(self):
        """Create an empty hypothesis for manual entry."""
        empty_hypothesis = Hypothesis(
            id="hyp_manual_001",
            description="",
            rationale="",
            success_criteria="",
            selected_for_experimentation=True
        )
        
        self.hypotheses = [empty_hypothesis]
        self.current_hypothesis = empty_hypothesis
        self.current_hypothesis_index = 0
        
        # Create the editable sections
        self._create_hypothesis_fields()
    
    def on_show(self):
        """Called when screen is shown - load hypothesis if not already loaded."""
        # Only load if we haven't loaded yet
        if not hasattr(self, 'current_hypothesis') or self.current_hypothesis is None:
            self._load_hypothesis()

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

    def _create_hypothesis_fields(self):
        """Create editable fields for the hypothesis."""
        if self.current_hypothesis is None:
            return
        hyp = self.current_hypothesis
        
        self.description_text = self._create_section(
            "Description", 
            hyp.description, 
            height=8
        )
        
        self.rationale_text = self._create_section(
            "Rationale", 
            hyp.rationale, 
            height=8
        )
        
        self.success_criteria_text = self._create_section(
            "Success Criteria", 
            hyp.success_criteria, 
            height=8
        )

    def _create_section(self, title: str, content: str, height: int = 4) -> tk.Text:
        """Create a labeled section with an editable text area."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding="10")
        frame.pack(fill="x", pady=10)
        
        text_widget = create_text_area(frame, height=height)
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
        success_criteria = self.success_criteria_text.get("1.0", "end-1c").strip()
        
        # Update the hypothesis object
        updated_hypothesis = Hypothesis(
            id=self.current_hypothesis.id,
            description=description,
            rationale=rationale,
            success_criteria=success_criteria,
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
        if EXPERIMENT_PLAN_FILE.exists():
            super().on_next()
        else:
            self._run_generation(updated_hypothesis)

    def _run_generation(self, hypothesis: Hypothesis):
        """Run experiment plan generation with progress popup."""
        popup = ProgressPopup(self.controller, "Generating Experiment Plan")
        
        def task():
            try:
                # Load paper concept
                self.after(0, lambda: popup.update_status("Loading paper concept"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Load user requirements
                self.after(0, lambda: popup.update_status("Loading user requirements"))
                from phases.context_analysis.user_requirements import UserRequirements
                user_requirements = None
                try:
                    user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                except FileNotFoundError:
                    print("User requirements file not found, proceeding without it")
                except Exception as e:
                    print(f"Warning: Failed to load user requirements: {e}")
                
                # Load raw code files (needed for experiment plan - paper concept already has snippets)
                self.after(0, lambda: popup.update_status("Loading code files"))
                from phases.context_analysis.user_code_analysis import CodeAnalyzer
                from settings import Settings
                user_code = None
                try:
                    code_analyzer = CodeAnalyzer(model_name=Settings.CODE_ANALYSIS_MODEL)
                    user_code = code_analyzer.load_code_files("user_files")
                except Exception as e:
                    print(f"Warning: Failed to load code files: {e}")
                    user_code = None
                
                # Generate experiment plan
                self.after(0, lambda: popup.update_status("Generating experiment plan"))
                experiment_runner = ExperimentRunner()
                experiment_plan = experiment_runner._generate_experiment_plan(
                    hypothesis, 
                    paper_concept,
                    user_requirements=user_requirements,
                    user_code=user_code
                )
                experiment_runner.save_experiment_plan(experiment_plan)
                
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
