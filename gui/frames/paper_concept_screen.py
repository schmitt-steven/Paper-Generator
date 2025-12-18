import tkinter as tk
from tkinter import ttk
from pathlib import Path
import threading
from ..base_frame import BaseFrame, create_scrollable_text_area, ProgressPopup
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from phases.context_analysis.user_requirements import UserRequirements
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from settings import Settings
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
            next_text="Continue",
            has_regenerate=True,
            regenerate_text="Regenerate",
            header_file_path=self.file_path
        )

    def create_content(self):
        # Info text
        self._create_info_section()

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        info_text = (
            "Review and edit the paper concept below.\n"
            "This was generated based on your code and provided information.\n"
            "It will act as a basis for the automatic literature search and paper writing."
        )

        label = ttk.Label(
            explanation_frame,
            text=info_text,
            font=self.controller.fonts.default_font,
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
        self.description_text = self._create_section("Description", self.concept.description, height=20)
        self.code_snippets_text = self._create_section("Important Code Snippets", self.concept.code_snippets, height=15)
        self.open_questions_text = self._create_section("Questions for Literature Search", self.concept.open_questions, height=15)

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

    def _create_section(self, title: str, content: str, height: int = 8) -> tk.Text:
        """Create a labeled section with an editable text area."""
        # Container
        section_container = ttk.Frame(self.scrollable_frame, padding=(0, 0, 0, 15))
        section_container.pack(fill="x")
        
        # Header
        ttk.Label(
            section_container, 
            text=title, 
            font=self.controller.fonts.sub_header_font
        ).pack(anchor="w", pady=(0, 10))
        
        container, text_widget = create_scrollable_text_area(section_container, height=height)
        container.pack(fill="x", expand=True)
        text_widget.insert("1.0", content)
        
        return text_widget

    def _save_concept(self) -> PaperConcept:
        """Save the edited content and return updated concept."""
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
        
        # Return updated concept for use in generation
        return PaperConcept(
            description=description,
            code_snippets=code_snippets,
            open_questions=open_questions
        )

    def on_next(self):
        """Save the edited content and proceed to next screen."""
        if self.concept is None:
            super().on_next()
            return
        
        # Always save first
        self._save_concept()
        
        # Continue to next screen - user will search for papers on paper selection screen
        super().on_next()
    
    def on_show(self):
        """Called when screen is shown - load concept if not already loaded."""
        # Only load if we haven't loaded yet
        if not hasattr(self, 'concept') or self.concept is None:
            if Path(self.file_path).exists():
                self._load_concept()

    def on_regenerate(self):
        """Regenerate the paper concept from scratch."""
        if not tk.messagebox.askyesno("Confirm Regeneration", 
                                      "This will completely overwrite the current paper concept based on your code and requirements.\n\nDo you want to continue?"):
            return

        popup = ProgressPopup(self.controller, "Regenerating Paper Concept")
        
        def task():
            try:
                # 1. Load User Requirements
                self.after(0, lambda: popup.update_status("Loading user requirements..."))
                user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                
                # 2. Analyze Code
                self.after(0, lambda: popup.update_status("Analyzing code files..."))
                code_analyzer = CodeAnalyzer(model_name=Settings.CODE_ANALYSIS_MODEL)
                # Hardcoded "user_files" as per project convention, can be made dynamic if needed
                code_files = code_analyzer.load_code_files("user_files") 
                analyzed_code = code_analyzer.analyze_all_files(code_files)
                
                # 3. Generate Paper Concept
                self.after(0, lambda: popup.update_status("Generating concept (this may take a while)..."))
                paper_conception = PaperConception(
                    model_name=Settings.PAPER_CONCEPTION_MODEL,
                    user_code=analyzed_code,
                    user_requirements=user_requirements
                )
                
                # This automatically saves to file
                paper_conception.build_paper_concept()
                
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
        self._load_concept()
        # Refresh the text areas with new content
        self.description_text.delete("1.0", "end")
        self.description_text.insert("1.0", self.concept.description)
        
        self.code_snippets_text.delete("1.0", "end")
        self.code_snippets_text.insert("1.0", self.concept.code_snippets)
        
        self.open_questions_text.delete("1.0", "end")
        self.open_questions_text.insert("1.0", self.concept.open_questions)

