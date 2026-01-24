"""
Writing Prompts Screen - Display section writing prompts with collapsible cards and copy functionality.
"""

import tkinter as tk
from tkinter import ttk
import json
from typing import Dict
from pathlib import Path


from ..base_frame import BaseFrame, CardBorderFrame
from ..info_texts import WRITING_PROMPTS_INFO
from ..markdown_view import MarkdownView
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
    SECONDARY_TEXT_DARK, SECONDARY_TEXT_LIGHT,
    LINK_COLOR_DARK, LINK_COLOR_LIGHT,
)


PROMPTS_FILE = Path("output/section_writing_prompts.json")


class CollapsiblePromptCard(CardBorderFrame):
    """A collapsible card for a paper section's writing prompt."""
    
    def __init__(self, parent, section_name: str, prompt_content: str, controller):
        super().__init__(parent, padx=1, pady=1)
        self.section_name = section_name
        self.prompt_content = prompt_content
        self.controller = controller
        self.expanded = False
        
        self._build_ui()
    
    def _build_ui(self):
        # Header frame
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        # Left side: toggle + title
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
        
        # Copy button on right
        copy_color = SECONDARY_TEXT_DARK if self.controller.current_theme == "dark" else SECONDARY_TEXT_LIGHT
        hover_color = LINK_COLOR_DARK if self.controller.current_theme == "dark" else LINK_COLOR_LIGHT
        
        copy_btn = tk.Label(
            header,
            text="Copy",
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=copy_color,
            cursor="hand2",
            padx=8,
            pady=2
        )
        copy_btn.pack(side="right", padx=(5, 0))
        
        # Hover effects
        copy_btn.bind("<Enter>", lambda e: copy_btn.config(fg=hover_color))
        copy_btn.bind("<Leave>", lambda e: copy_btn.config(fg=copy_color))
        copy_btn.bind("<Button-1>", lambda e: self._copy_prompt())
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame (hidden by default)
        self.content_frame = ttk.Frame(self, style="CardContent.TFrame", padding=0)
        # Don't pack yet - only show when expanded
        
        # Markdown View
        self.text_widget = MarkdownView(
            self.content_frame,
            font_manager=self.controller.fonts,
            theme_mode=self.controller.current_theme,
            padx=12,
            pady=10
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        self.text_widget.set_markdown(self.prompt_content)
    
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
    
    def _copy_prompt(self):
        """Copy the prompt content to clipboard."""
        self.controller.clipboard_clear()
        self.controller.clipboard_append(self.prompt_content)
        print(f"Copied {self.section_name} prompt to clipboard")


class WritingPromptsScreen(BaseFrame):
    """Screen displaying section writing prompts with collapsible cards."""
    
    def __init__(self, parent, controller):
        self.prompt_cards: Dict[str, CollapsiblePromptCard] = {}
        self._loaded = False
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Writing Prompts",
            has_next=False,
            has_back=True,
            back_text="Back",
            #header_file_path=PROMPTS_FILE,
            info_content=WRITING_PROMPTS_INFO
        )
    
    def create_content(self):
        """Create the initial UI structure."""
        pass
    
    def on_show(self):
        """Load prompts when screen is shown."""
        if self._loaded:
            return
        
        self._load_prompts()
        self._loaded = True
    
    def _load_prompts(self):
        """Load and parse the prompts file."""
        if not PROMPTS_FILE.exists():
            self.show_error_message("Prompts file not found", str(PROMPTS_FILE))
            return

        # Load JSON
        try:
            content = PROMPTS_FILE.read_text(encoding="utf-8")
            sections = json.loads(content)
            
            # Display sections
            for i, (section_name, prompt_content) in enumerate(sections.items()):
                card = CollapsiblePromptCard(
                    self.scrollable_frame,
                    section_name,
                    prompt_content.strip(),
                    self.controller
                )
                card.pack(fill="x", pady=(10 if i == 0 else 0, 8))
                self.prompt_cards[section_name] = card
                
        except Exception as e:
            self.show_error_message("Error loading prompts", str(e))
    

    
    def on_back(self):
        """Navigate back to Paper Draft screen."""
        from .paper_draft_screen import PaperDraftScreen
        self.controller.show_frame(PaperDraftScreen)
