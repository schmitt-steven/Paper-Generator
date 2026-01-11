import tkinter as tk
from tkinter import ttk
from pathlib import Path
import threading
from ..base_frame import BaseFrame, create_scrollable_text_area, ProgressPopup
from ..info_texts import PAPER_CONCEPT_INFO
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
            header_file_path=self.file_path,
            info_content=PAPER_CONCEPT_INFO
        )

    def create_content(self):
        pass

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
        
        # Create sections
        self.description_text = self._create_section("Description", self.concept.description, height=20)
        self.code_snippets_text = self._create_section("Important Code Snippets", self.concept.code_snippets, height=20)
        self.open_questions_text = self._create_section("Questions for Literature Search", self.concept.open_questions, height=20)

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
        
        self._save_concept()
        
        # Show next screen
        super().on_next()
    
    def on_show(self):
        """Called when screen is shown - load concept if not already loaded."""
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
                self.after(0, lambda: popup.update_status("Loading user requirements"))
                user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                
                # 2. Analyze Code
                self.after(0, lambda: popup.update_status("Analyzing code files"))
                code_analyzer = CodeAnalyzer(model_name=Settings.CODE_ANALYSIS_MODEL)
                # Hardcoded "user_files" as per project convention, can be made dynamic if needed
                code_files = code_analyzer.load_code_files("user_files") 
                analyzed_code = code_analyzer.analyze_all_files(code_files)
                
                # 3. Generate Paper Concept
                self.after(0, lambda: popup.update_status("Generating concept"))
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
        
        # Clear existing content before reloading
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Reload UI with new concept (this creates fresh text areas)
        self._load_concept()
        
        # Re-apply theme colors to newly created widgets (for borders)
        self.controller.apply_theme_colors(self)

