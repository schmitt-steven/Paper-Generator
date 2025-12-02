import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame, create_styled_text
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from utils.file_utils import save_markdown


class PaperConceptScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.file_path = "output/paper_concept.md"
        self.concept: PaperConcept | None = None
        
        # Text widgets for each section
        self.description_text: tk.Text
        self.code_snippets_text: tk.Text
        self.open_questions_text: tk.Text
        
        super().__init__(
            parent,
            controller,
            title="Paper Concept",
            next_text="Continue"
        )

    def create_content(self):
        # Info text
        self._create_info_section()
        
        # Load the paper concept
        self._load_concept()

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Review and edit the paper concept below.\n"
            "This was generated based on your code and provided information.\n"
            "It will act as a basis for the automatic literature search and paper writing."
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

    def _load_concept(self):
        """Load the paper concept from file and create UI sections."""
        try:
            self.concept = PaperConception.load_paper_concept(self.file_path)
        except FileNotFoundError:
            self._show_error(f"Paper concept not found: {self.file_path}\n\nPlease complete the previous steps first.")
            return
        except Exception as e:
            self._show_error(f"Error loading paper concept: {e}")
            return
        
        # Create the three sections
        self.description_text = self._create_section("Description", self.concept.description, height=12)
        self.code_snippets_text = self._create_section("Code Snippets", self.concept.code_snippets, height=10)
        self.open_questions_text = self._create_section("Questions for Literature Search", self.concept.open_questions, height=6)

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

    def _create_section(self, title: str, content: str, height: int = 8) -> tk.Text:
        """Create a labeled section with an editable text area."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding="10")
        frame.pack(fill="x", pady=10)
        
        text_widget = create_styled_text(frame, height=height)
        text_widget.pack(fill="x", expand=True)
        text_widget.insert("1.0", content)
        
        return text_widget

    def on_next(self):
        """Save the edited content and proceed."""
        if self.concept is None:
            super().on_next()
            return
        
        # Get content from text widgets
        description = self.description_text.get("1.0", "end-1c").strip()
        code_snippets = self.code_snippets_text.get("1.0", "end-1c").strip()
        open_questions = self.open_questions_text.get("1.0", "end-1c").strip()
        
        # Build the markdown content
        lines = [
            "# Paper Concept",
            "",
            description,
            "",
            "# Important Code Snippets",
            "",
            code_snippets,
            "",
            "# Open Questions",
            "",
            open_questions,
        ]
        content = "\n".join(lines)
        
        try:
            save_markdown(content, "paper_concept.md", "output")
            print(f"[PaperConcept] Saved changes to {self.file_path}")
        except Exception as e:
            print(f"[PaperConcept] Failed to save: {e}")
        
        super().on_next()

