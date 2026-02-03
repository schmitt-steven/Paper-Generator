import tkinter as tk
from tkinter import ttk, messagebox
import threading
from pathlib import Path
from typing import Optional

from ..base_frame import BaseFrame, ProgressPopup, CardBorderFrame
from ..markdown_view import MarkdownView
from ..info_texts import HYPOTHESIS_INFO
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
)
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder, Hypothesis
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from phases.context_analysis.paper_specification import PaperSpecification
from phases.experimentation.experiment_runner import ExperimentRunner
from settings import Settings


HYPOTHESIS_FILE = "output/hypothesis.md"
EXPERIMENT_PLAN_FILE = Path("output/experiments/experiment_plan.md")


class CollapsibleHypothesisCard(CardBorderFrame):
    """A collapsible card for hypothesis sections (read-only)."""
    
    def __init__(self, parent, section_name: str, content: str, controller, start_expanded: bool = False, height: int = 250):
        super().__init__(parent, padx=1, pady=1)
        self.section_name = section_name
        self.content = content
        self.controller = controller
        self.height = height
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
        
        # Text widget to display content
        self.text_widget = MarkdownView(
            self.content_frame,
            font_manager=self.controller.fonts,
            theme_mode=self.controller.current_theme,
            padx=12,
            pady=10,
            height=self.height
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        self.text_widget.set_markdown(self.content)
    
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


class HypothesisScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.current_hypothesis: Optional[Hypothesis] = None
        self.cards: list[CollapsibleHypothesisCard] = []
        
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
        """Load hypothesis from file or create empty one."""
        if Path(HYPOTHESIS_FILE).exists():
            try:
                self.current_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESIS_FILE)
                if self.current_hypothesis:
                    self._create_hypothesis_cards()
                    return
            except Exception as e:
                print(f"Error loading hypothesis: {e}")
        
        # No file or load failed
        self.show_error_message("Hypothesis Error", "No hypothesis found. Please complete previous steps or try to regenerate it.")
    
    def on_show(self):
        """Called when screen is shown - load hypothesis if not already loaded."""
        if not hasattr(self, 'current_hypothesis') or self.current_hypothesis is None:
            self._load_hypothesis()
            
        # Update next button text
        next_text = "Continue" if EXPERIMENT_PLAN_FILE.exists() else "Generate Experiment Plan"
        self.set_next_text(next_text)

    def _create_hypothesis_cards(self):
        """Create collapsible cards for the hypothesis fields."""
        if self.current_hypothesis is None:
            return
            
        hyp = self.current_hypothesis
        
        # Clear existing cards
        for card in self.cards:
            card.destroy()
        self.cards.clear()
        
        sections = [
            ("Description", hyp.description),
            ("Rationale", hyp.rationale),
            ("Success Criteria", hyp.success_criteria)
        ]
        
        for i, (title, content) in enumerate(sections):
            card = CollapsibleHypothesisCard(
                self.scrollable_frame,
                title,
                content,
                self.controller,
                start_expanded=True  # All expanded by default
            )
            card.pack(fill="x", pady=10)
            self.cards.append(card)

    def on_next(self):
        """Proceed to next screen or generate experiment plan."""
        if self.current_hypothesis is None:
            super().on_next()
            return
        
        # Check if output exists
        if EXPERIMENT_PLAN_FILE.exists():
            super().on_next()
        else:
            self._run_generation(self.current_hypothesis)

    def _run_generation(self, hypothesis: Hypothesis):
        """Run experiment plan generation with progress popup."""
        popup = ProgressPopup(self.controller, "Generating Experiment Plan")
        
        def task():
            try:
                def status_callback(msg):
                    self.after(0, lambda m=msg: popup.update_status(m))
                
                ExperimentRunner.generate_new_experiment_plan(
                    hypothesis,
                    status_callback=status_callback
                )
                
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
        self.after(200, lambda: messagebox.showinfo("Success", "Experiment plan successfully generated."))
        self.controller.next_screen()

    def on_regenerate(self):
        """Regenerate the hypothesis from scratch."""
        if not tk.messagebox.askyesno("Confirm Regeneration", 
                                      "This will completely overwrite the current hypothesis based on your research context and requirements.\n\nDo you want to continue?"):
            return

        popup = ProgressPopup(self.controller, "Regenerating Hypothesis")
        
        def task():
            try:
                def status_callback(msg):
                    self.after(0, lambda m=msg: popup.update_status(m))
                
                HypothesisBuilder.generate_new_hypothesis(status_callback=status_callback)
                
                # Reload UI
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
        # Clear only the cards, not the action buttons
        for card in self.cards:
            card.destroy()
        self.cards.clear()
        
        self.current_hypothesis = None
        self._load_hypothesis()
        
        self.controller.apply_theme_colors(self)
        
        self.after(200, lambda: messagebox.showinfo("Success", "Hypothesis was successfully regenerated!"))
