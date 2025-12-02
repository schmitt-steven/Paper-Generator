import tkinter as tk
from tkinter import ttk

from ..base_frame import BaseFrame, create_styled_text
from utils.file_utils import load_markdown, save_markdown


EXPERIMENTS_DIR = "output/experiments"
EXPERIMENTAL_PLAN_FILE = "experimental_plan.md"


class ExperimentationPlanScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.plan_text: tk.Text
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Experimentation Plan",
            next_text="Save & Continue",
        )

    def create_content(self):
        # Info text
        self._create_info_section()
        
        # Load and display the experimental plan
        self._load_plan()

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Review and edit the experimental plan below.\n"
            "This plan was generated based on your hypothesis and paper concept.\n"
            "It describes how the experiments will be conducted to test the hypothesis."
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

    def _load_plan(self):
        """Load experimental plan from file and display it."""
        try:
            plan_content = load_markdown(EXPERIMENTAL_PLAN_FILE, EXPERIMENTS_DIR)
        except FileNotFoundError:
            self._show_error(
                f"Experimental plan not found: {EXPERIMENTS_DIR}/{EXPERIMENTAL_PLAN_FILE}\n\n"
                "Please complete the previous steps first."
            )
            return
        except Exception as e:
            self._show_error(f"Error loading experimental plan: {e}")
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
            font=("SF Pro", 14),
            foreground="red",
            wraplength=500
        ).pack()

    def _create_plan_section(self, content: str):
        """Create a labeled section with an editable text area for the plan."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Experimental Plan", padding="10")
        frame.pack(fill="both", expand=True, pady=10)
        
        self.plan_text = create_styled_text(frame, height=25)
        self.plan_text.pack(fill="both", expand=True)
        self.plan_text.insert("1.0", content)

    def on_next(self):
        """Save the edited plan and proceed."""
        if not hasattr(self, 'plan_text'):
            super().on_next()
            return
        
        # Get content from text widget
        plan_content = self.plan_text.get("1.0", "end-1c")
        
        try:
            save_markdown(plan_content, EXPERIMENTAL_PLAN_FILE, EXPERIMENTS_DIR)
            print(f"[ExperimentationPlan] Saved changes to {EXPERIMENTS_DIR}/{EXPERIMENTAL_PLAN_FILE}")
        except Exception as e:
            print(f"[ExperimentationPlan] Failed to save: {e}")
        
        super().on_next()
