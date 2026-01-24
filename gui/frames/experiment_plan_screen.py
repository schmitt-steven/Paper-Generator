import tkinter as tk
from tkinter import ttk, messagebox
import threading
import re
from pathlib import Path
from typing import Dict

from ..base_frame import BaseFrame, ProgressPopup, CardBorderFrame
from ..info_texts import EXPERIMENT_PLAN_INFO
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
)
from utils.file_utils import load_markdown
from ..markdown_view import MarkdownView
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.context_analysis.paper_conception import PaperConception
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.context_analysis.user_requirements import UserRequirements
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from settings import Settings


EXPERIMENTS_DIR = "output/experiments"
EXPERIMENT_PLAN_FILE = "experiment_plan.md"
HYPOTHESES_FILE = "output/hypothesis.md"


class CollapsiblePlanCard(CardBorderFrame):
    """A collapsible card for experiment plan sections (read-only)."""
    
    def __init__(self, parent, section_name: str, content: str, controller, start_expanded: bool = False):
        super().__init__(parent, padx=1, pady=1)
        self.section_name = section_name
        self.content = content
        self.controller = controller
        self.start_expanded = start_expanded
        self.expanded = False
        
        self._build_ui()
        
        if start_expanded:
            self.expand()
    
    def _build_ui(self):
        # Header frame
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        # Left side: toggle + title (clickable)
        left_frame = tk.Frame(header, bg=header_bg)
        left_frame.pack(side="left", fill="x", expand=True)
        left_frame.bind("<Button-1>", lambda e: self.toggle())
        
        # Toggle indicator
        self.toggle_label = tk.Label(
            left_frame,
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
            left_frame,
            text=self.section_name,
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", lambda e: self.toggle())
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame (hidden by default)
        self.content_frame = ttk.Frame(self, style="CardContent.TFrame", padding=0)
        
        # Text widget
        text_bg = TEXT_BG_DARK_ALT if self.controller.current_theme == "dark" else TEXT_BG_LIGHT_ALT
        text_fg = TEXT_FG_DARK if self.controller.current_theme == "dark" else TEXT_FG_LIGHT
        
        self.text_widget = MarkdownView(
            self.content_frame,
            font_manager=self.controller.fonts,
            theme_mode=self.controller.current_theme,
            height=100,
            padx=12,
            pady=10
        )
        
        self.text_widget.pack(side="left", fill="both", expand=True)
        
        self.text_widget.set_markdown(self.content)
        
        # After insert, calculate actual display lines and resize
        def adjust_height():
            self.text_widget.update_idletasks()
            try:
                display_lines = self.text_widget.count("1.0", "end", "displaylines")
                if display_lines:
                    actual_lines = display_lines[0] if isinstance(display_lines, tuple) else display_lines
                    height = min(actual_lines + 1, 50)
                    self.text_widget.config(height=height)
            except:
                pass
        
        self.text_widget.after(10, adjust_height)
    
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


class ExperimentPlanScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.cards: list[CollapsiblePlanCard] = []
        
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
            header_file_path=Path(EXPERIMENTS_DIR) / EXPERIMENT_PLAN_FILE,
            info_content=EXPERIMENT_PLAN_INFO
        )
        
        # Bind resize event to adjust card height
        self._canvas.bind("<Configure>", self._update_card_height, add="+")

    def _update_card_height(self, event=None):
        """Dynamically adjust card height to fill available space."""
        if not self.cards:
            return
            
        # Get canvas height
        canvas_height = self._canvas.winfo_height()
        
        # Determine target height (canvas height - margins - card header)
        # Margins: BaseFrame padding ~20, Card margin ~20, Card header ~40, plus safe zone
        target_height = canvas_height - 140 
        
        # Minimum safe height
        if target_height < 400:
            target_height = 400
            
        # Update the card
        for card in self.cards:
            if hasattr(card, 'text_widget'):
                card.text_widget.configure(height=target_height)

    def create_content(self):
        pass

    def _load_plan(self):
        """Load experiment plan from file and display it as cards."""
        try:
            plan_content = load_markdown(EXPERIMENT_PLAN_FILE, EXPERIMENTS_DIR)
        except FileNotFoundError:
            self.show_error_message(
                "Experiment Plan Error",
                f"Experiment plan not found: {EXPERIMENTS_DIR}/{EXPERIMENT_PLAN_FILE}\n\n"
                "Please complete the previous steps first."
            )
            return
        except Exception as e:
            self.show_error_message("Error", f"Error loading experiment plan: {e}")
            return
        
        # Clear existing
        for card in self.cards:
            card.destroy()
        self.cards.clear()
        
        # Create single card with full content
        card = CollapsiblePlanCard(
            self.scrollable_frame,
            "Experiment Plan",
            plan_content.strip(),
            self.controller,
            start_expanded=True
        )
        card.pack(fill="both", expand=True, pady=10)
        self.cards.append(card)
        
        # Force initial resize
        self.after(100, self._update_card_height)

    def on_next(self):
        """Proceed or run experiments (no saving)."""
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
                    load_existing_plan=True,  # Use existing plan file (we trust it's there)
                    load_existing_code=False  # Generate new code
                )
                
                # Continue to next screen
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
        messagebox.showinfo("Success", "Experiment successfully completed.")
        self.controller.next_screen()
    
    def on_show(self):
        """Called when screen is shown - load plan if not already loaded."""
        if not self.cards: # Reload if no cards shown
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
                self.after(0, lambda: popup.update_status("Generating new plan"))
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
        # Force reload from file
        self._load_plan()
        
        # Re-apply theme colors
        self.controller.apply_theme_colors(self)
        
        messagebox.showinfo("Success", "Experiment plan successfully regenerated.")
