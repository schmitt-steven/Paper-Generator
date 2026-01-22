import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading
from ..base_frame import BaseFrame, CardBorderFrame, ProgressPopup
from ..info_texts import PAPER_CONCEPT_INFO
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
)
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from phases.context_analysis.user_requirements import UserRequirements
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from settings import Settings


class CollapsibleConceptCard(CardBorderFrame):
    """A collapsible card for a paper concept section (read-only, no copy button)."""
    
    def __init__(self, parent, section_name: str, content: str, controller, start_expanded: bool = False):
        super().__init__(parent, padx=1, pady=1)
        self.section_name = section_name
        self.content = content
        self.controller = controller
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
        # Don't pack yet - only show when expanded
        
        # Text widget for content (read-only)
        text_bg = TEXT_BG_DARK_ALT if self.controller.current_theme == "dark" else TEXT_BG_LIGHT_ALT
        text_fg = TEXT_FG_DARK if self.controller.current_theme == "dark" else TEXT_FG_LIGHT
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        
        self.text_widget = tk.Text(
            self.content_frame,
            height=20,
            font=self.controller.fonts.text_area_font,
            wrap="word",
            background=text_bg,
            foreground=text_fg,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=12,
            pady=10,
            yscrollcommand=scrollbar.set
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        self.text_widget.insert("1.0", self.content)
        self.text_widget.config(state="disabled")  # Read-only
    
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
    
    def collapse(self):
        """Force collapse."""
        if self.expanded:
            self.toggle()


class PaperConceptScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.file_path = "output/paper_concept.md"
        self.concept: PaperConcept | None = None
        self.concept_cards = []
        
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
            self.show_error_message("Paper Concept Error", f"Paper concept not found: {self.file_path}\n\nPlease complete the previous steps first.")
            return
        except Exception as e:
            self.show_error_message("Error", f"Error loading paper concept: {e}")
            return
        
        # Create collapsible sections
        sections = [
            ("Description", self.concept.description),
            ("Important Code Snippets", self.concept.code_snippets),
            ("Questions for Literature Search", self.concept.open_questions),
        ]
        
        self.concept_cards = []
        for i, (title, content) in enumerate(sections):
            card = CollapsibleConceptCard(
                self.scrollable_frame,
                title,
                content,
                self.controller,
                start_expanded=True
            )
            card.pack(fill="x", pady=(10 if i == 0 else 0, 8))
            self.concept_cards.append(card)



    def on_next(self):
        """Proceed to next screen (no saving needed - content is read-only)."""
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
        
        # Clear only the concept cards, not the action buttons
        for card in self.concept_cards:
            card.destroy()
        
        # Reset concept to force reload
        self.concept = None
        self.concept_cards = []
        
        # Reload UI with new concept
        self._load_concept()
        
        # Re-apply theme colors to newly created widgets
        self.controller.apply_theme_colors(self)
        
        messagebox.showinfo("Success", "Paper concept successfully regenerated.")
