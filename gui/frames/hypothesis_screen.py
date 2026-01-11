import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path
from typing import List, Optional

from ..base_frame import BaseFrame, ProgressPopup, create_scrollable_text_area
from ..info_texts import HYPOTHESIS_INFO
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.hypothesis_generation.hypothesis_builder import Hypothesis
from phases.context_analysis.paper_conception import PaperConception
from phases.context_analysis.user_requirements import UserRequirements
from phases.experimentation.experiment_runner import ExperimentRunner
from settings import Settings


HYPOTHESIS_FILE = "output/hypothesis.md"

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
            regenerate_text="Regenerate",
            header_file_path=HYPOTHESIS_FILE,
            info_content=HYPOTHESIS_INFO
        )

    def create_content(self):
       pass

    def _load_hypothesis(self):
        """Load hypothesis from file or create empty one for manual entry."""
        if Path(HYPOTHESIS_FILE).exists():
            try:
                self.current_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESIS_FILE)
                if self.current_hypothesis:
                    # Create the editable sections
                    self._create_hypothesis_fields()
                    return
            except Exception as e:
                print(f"Error loading hypothesis: {e}")
        
        # No file or load failed, create empty hypothesis
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
        
        self.current_hypothesis = empty_hypothesis
        
        # Create the editable sections
        self._create_hypothesis_fields()
    
    def on_show(self):
        """Called when screen is shown - load hypothesis if not already loaded."""
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
            height=12
        )
        
        self.rationale_text = self._create_section(
            "Rationale", 
            hyp.rationale, 
            height=12
        )
        
        self.success_criteria_text = self._create_section(
            "Success Criteria", 
            hyp.success_criteria, 
            height=12
        )

    def _create_section(self, title: str, content: str, height: int = 4) -> tk.Text:
        """Create a labeled section with an editable text area inside a card."""
        from ..base_frame import CardBorderFrame
        from ..theme_colors import CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT
        
        # Card Container
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)
        
        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        tk.Label(
            header, 
            text=title, 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # Content
        content_frame = ttk.Frame(card, style="CardContent.TFrame", padding=0)
        content_frame.pack(fill="both", expand=True)
        
        # Text area container
        inner = ttk.Frame(content_frame, style="CardRow.TFrame")
        inner.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(inner, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        
        text_widget = tk.Text(
            inner,
            height=height,
            wrap="word",
            font=self.controller.fonts.text_area_font,
            padx=10,
            pady=10,
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            yscrollcommand=scrollbar.set
        )
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.insert("1.0", content)
        
        self.controller.apply_theme_colors(text_widget)
        
        return text_widget

    def _save_hypothesis(self) -> Hypothesis | None:
        """Save the edited hypothesis and return it."""
        if self.current_hypothesis is None:
            return None
        
        # Get content from text widgets
        description = self.description_text.get("1.0", "end-1c").strip()
        rationale = self.rationale_text.get("1.0", "end-1c").strip()
        success_criteria = self.success_criteria_text.get("1.0", "end-1c").strip()
        
        # Update hypothesis object
        updated_hypothesis = Hypothesis(
            id=self.current_hypothesis.id,
            description=description,
            rationale=rationale,
            success_criteria=success_criteria,
            selected_for_experimentation=self.current_hypothesis.selected_for_experimentation
        )
        
        # Save back to file
        try:
            HypothesisBuilder.save_hypothesis(updated_hypothesis, HYPOTHESIS_FILE)
            self.current_hypothesis = updated_hypothesis # Update current
            print(f"[Hypothesis] Saved changes to {HYPOTHESIS_FILE}")
        except Exception as e:
            print(f"[Hypothesis] Failed to save: {e}")
        
        return updated_hypothesis

    def on_next(self):
        """Save the edited hypothesis and proceed or generate experiment plan."""
        if self.current_hypothesis is None:
            super().on_next()
            return
        
        # Save first
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
                
                # Close popup and continue
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

    def on_regenerate(self):
        """Regenerate the hypothesis from scratch."""
        if not tk.messagebox.askyesno("Confirm Regeneration", 
                                      "This will completely overwrite the current hypothesis based on your paper concept and requirements.\n\nDo you want to continue?"):
            return

        popup = ProgressPopup(self.controller, "Regenerating Hypothesis")
        
        def task():
            try:
                # 1. Load resources
                self.after(0, lambda: popup.update_status("Loading resources"))
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                
                # 2. Initialize Builder
                self.after(0, lambda: popup.update_status("Initializing builder"))
                builder = HypothesisBuilder(
                    model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
                    paper_concept=paper_concept,
                    top_limitations=[],
                    num_papers_analyzed=0
                )
                
                # 3. Generate Hypothesis
                self.after(0, lambda: popup.update_status("Generating hypothesis"))
                # This automatically saves to file
                builder.create_hypothesis_from_user_input(user_requirements)
                
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
        # Force reload from file
        self._load_hypothesis()
        
        # Refresh widgets
        if self.current_hypothesis:
             if hasattr(self, 'description_text'):
                 self.description_text.delete("1.0", "end")
                 self.description_text.insert("1.0", self.current_hypothesis.description)
                 
                 self.rationale_text.delete("1.0", "end")
                 self.rationale_text.insert("1.0", self.current_hypothesis.rationale)
                 
                 self.success_criteria_text.delete("1.0", "end")
                 self.success_criteria_text.insert("1.0", self.current_hypothesis.success_criteria)
             else:
                 self._create_hypothesis_fields()
