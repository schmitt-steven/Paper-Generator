import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading
from ..base_frame import BaseFrame, CardBorderFrame, ProgressPopup
from ..markdown_view import MarkdownView
from ..info_texts import RESEARCH_CONTEXT_INFO
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
)
from phases.context_analysis.research_context_generator import ResearchContextGenerator, ResearchContext
from phases.context_analysis.paper_specification import PaperSpecification
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from settings import Settings


class CollapsibleContextCard(CardBorderFrame):
    """A collapsible card for a research context section (read-only, no copy button)."""
    
    def __init__(self, parent, section_name: str, content: str, controller, start_expanded: bool = False, height: int = 400):
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
        # Don't pack yet - only show when expanded
        
        # Text widget for content (read-only)
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
    
    def collapse(self):
        """Force collapse."""
        if self.expanded:
            self.toggle()


class ResearchContextScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.file_path = "output/research_context.md"
        self.context: ResearchContext | None = None
        self.context_cards = []
        
        super().__init__(
            parent,
            controller,
            title="Research Context",
            next_text="Continue",
            has_regenerate=True,
            regenerate_text="Regenerate",
            header_file_path=self.file_path,
            info_content=RESEARCH_CONTEXT_INFO
        )

    def create_content(self):
        pass

    def _load_context(self):
        """Load the research context from file and create UI sections."""
        try:
            self.context = ResearchContextGenerator.load_research_context(self.file_path)
        except FileNotFoundError:
            self.show_error_message("Research Context Error", f"Research Context not found: {self.file_path}\n\nPlease complete the previous steps first.")
            return
        except Exception as e:
            self.show_error_message("Error", f"Error loading research context: {e}")
            return
        
        # Create collapsible sections: (title, content, expand_by_default)
        sections = [
            ("Description", self.context.description, True),
            ("Code Analysis", self.context.code_snippets or "No code files provided.", bool(self.context.code_snippets)),
            ("Dataset Descriptions", self.context.dataset_descriptions or "No datasets provided.", bool(self.context.dataset_descriptions)),
            ("Open Questions", self.context.open_questions, True),
        ]

        self.context_cards = []
        for i, (title, content, expanded) in enumerate(sections):
            card = CollapsibleContextCard(
                self.scrollable_frame,
                title,
                content,
                self.controller,
                start_expanded=expanded
            )
            card.pack(fill="x", pady=10)
            self.context_cards.append(card)

    def on_next(self):
        """Proceed to next screen (no saving needed - content is read-only)."""
        super().on_next()
    
    def on_show(self):
        """Called when screen is shown - load context if not already loaded."""
        if not hasattr(self, 'context') or self.context is None:
            if Path(self.file_path).exists():
                self._load_context()

    def on_regenerate(self):
        """Regenerate the research context from scratch."""
        if not tk.messagebox.askyesno("Confirm Regeneration", 
                                      "This will completely overwrite the current research context based on your code and requirements.\n\nDo you want to continue?"):
            return

        popup = ProgressPopup(self.controller, "Regenerating Research Context")
        
        def task():
            try:
                ResearchContextGenerator.generate_new_context(progress_callback=popup.update_status)
                
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
        
        # Clear only the context cards, not the action buttons
        for card in self.context_cards:
            card.destroy()
        
        # Reset context to force reload
        self.context = None
        self.context_cards = []
        
        # Reload UI with new context
        self._load_context()
        
        # Re-apply theme colors to newly created widgets
        self.controller.apply_theme_colors(self)
        
        self.after(200, lambda: messagebox.showinfo("Success", "Research Context was regenerated successfully!"))
