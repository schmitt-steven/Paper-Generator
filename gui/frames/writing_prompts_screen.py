"""
Writing Prompts Screen - Display section writing prompts with collapsible cards and copy functionality.
"""

import tkinter as tk
from tkinter import ttk
import json
import re
from typing import Dict
from pathlib import Path


from ..base_frame import BaseFrame, CardBorderFrame
from ..info_texts import WRITING_PROMPTS_INFO
from ..markdown_view import MarkdownView
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    CANVAS_BG_DARK, CANVAS_BG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
    SECONDARY_TEXT_DARK, SECONDARY_TEXT_LIGHT,
    LINK_COLOR_DARK, LINK_COLOR_LIGHT,
)


PROMPTS_FILE = Path("output/section_writing_prompts.json")
REWRITE_PROMPTS_FILE = Path("output/section_rewrite_prompts.json")


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
        self.text_widget.set_markdown(self._preprocess_prompt(self.prompt_content))
    
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
    
    @staticmethod
    def _preprocess_prompt(text: str) -> str:
        """Convert structured prompt text to markdown-compatible format.

        Prompts are plain text where every newline should be a visible line break.
        Markdown collapses single newlines, so we append two trailing spaces
        to force <br> line breaks, while preserving paragraph breaks (blank lines).
        """
        # Add two trailing spaces before single newlines (markdown line break)
        # but don't touch blank lines (paragraph breaks)
        return re.sub(r'(?<!\n)\n(?!\n)', '  \n', text)

    def _copy_prompt(self):
        """Copy the prompt content to clipboard."""
        self.controller.clipboard_clear()
        self.controller.clipboard_append(self.prompt_content)
        print(f"Copied {self.section_name} prompt to clipboard")


class WritingPromptsScreen(BaseFrame):
    """Screen displaying section writing prompts with collapsible cards."""
    
    def __init__(self, parent, controller):
        self.prompt_cards: Dict[str, CollapsiblePromptCard] = {}
        self._loaded_files: set = set()
        
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
        """Load prompts when screen is shown. Reloads if new prompt files appear."""
        files_present = set()
        if PROMPTS_FILE.exists():
            files_present.add(str(PROMPTS_FILE))
        if REWRITE_PROMPTS_FILE.exists():
            files_present.add(str(REWRITE_PROMPTS_FILE))

        if files_present == self._loaded_files:
            return

        self._reload_prompts(files_present)
    
    def _reload_prompts(self, files_present: set):
        """Clear and reload all prompt files."""
        # Clear existing cards
        for card in self.prompt_cards.values():
            card.destroy()
        self.prompt_cards.clear()

        # Clear any header labels in the scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not PROMPTS_FILE.exists():
            self.show_error_message("Prompts file not found", str(PROMPTS_FILE))
            return

        try:
            self._load_prompt_group("Initial Writing Prompts", PROMPTS_FILE, is_first=True)
            if REWRITE_PROMPTS_FILE.exists():
                self._load_prompt_group("Improvement Prompts", REWRITE_PROMPTS_FILE, is_first=False)
            self._loaded_files = files_present
        except Exception as e:
            self.show_error_message("Error loading prompts", str(e))

    def _load_prompt_group(self, group_title: str, file_path: Path, is_first: bool = False):
        """Load a group of prompts from a JSON file and display them with a header."""
        content = file_path.read_text(encoding="utf-8")
        sections = json.loads(content)

        # Group header label
        fg = TEXT_FG_DARK if self.controller.current_theme == "dark" else TEXT_FG_LIGHT
        bg = CANVAS_BG_DARK if self.controller.current_theme == "dark" else CANVAS_BG_LIGHT
        header_label = tk.Label(
            self.scrollable_frame,
            text=group_title,
            font=self.controller.fonts.medium_header_font,
            fg=fg,
            bg=bg,
            anchor="w",
        )
        header_label.pack(fill="x", pady=(10 if is_first else 20, 10), padx=0)

        # Display section cards
        for section_name, prompt_content in sections.items():
            card = CollapsiblePromptCard(
                self.scrollable_frame,
                section_name,
                prompt_content.strip(),
                self.controller
            )
            card.pack(fill="x", pady=(0, 8))
            self.prompt_cards[f"{group_title}::{section_name}"] = card
    

    
    def on_back(self):
        """Navigate back to Paper Draft screen."""
        from .paper_draft_screen import PaperDraftScreen
        self.controller.show_frame(PaperDraftScreen)
