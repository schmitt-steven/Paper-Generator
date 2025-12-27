"""
Writing Prompts Screen - Display section writing prompts with collapsible cards and copy functionality.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict
from pathlib import Path
import re

from ..base_frame import BaseFrame


PROMPTS_FILE = Path("output/section_writing_prompts.md")


class CollapsiblePromptCard(ttk.Frame):
    """A collapsible card for a paper section's writing prompt."""
    
    def __init__(self, parent, section_name: str, prompt_content: str, controller):
        super().__init__(parent, style="Card.TFrame", padding=1)
        self.section_name = section_name
        self.prompt_content = prompt_content
        self.controller = controller
        self.expanded = False
        
        self._build_ui()
    
    def _build_ui(self):
        # Header frame with toggle
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        # Get colors  
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        
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
        copy_color = "#888888" if self.controller.current_theme == "dark" else "#666666"
        hover_color = "#4a9eff" if self.controller.current_theme == "dark" else "#0078d4"
        
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
        
        # Content frame (hidden by default) - no padding
        self.content_frame = ttk.Frame(self, padding=0)
        # Don't pack yet - only show when expanded
        
        # Text widget for prompt content
        text_bg = "#242424" if self.controller.current_theme == "dark" else "#ffffff"
        text_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        
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
        
        self.text_widget.insert("1.0", self.prompt_content)
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
            header_file_path=PROMPTS_FILE
        )
    
    def create_content(self):
        """Create the initial UI structure."""
        # Remove default padding from scrollable frame
        self.scrollable_frame.configure(padding=(0, 12, 0, 10))
    
    def on_show(self):
        """Load prompts when screen is shown."""
        if self._loaded:
            return
        
        self._load_prompts()
        self._loaded = True
    
    def _load_prompts(self):
        """Load and parse the prompts file."""
        if not PROMPTS_FILE.exists():
            ttk.Label(
                self.scrollable_frame,
                text=f"Prompts file not found: {PROMPTS_FILE}",
                foreground="red"
            ).pack(pady=20)
            return
        
        try:
            content = PROMPTS_FILE.read_text(encoding="utf-8")
            sections = self._parse_sections(content)
            
            for section_name, prompt_content in sections.items():
                card = CollapsiblePromptCard(
                    self.scrollable_frame,
                    section_name,
                    prompt_content.strip(),
                    self.controller
                )
                card.pack(fill="x", pady=(0, 8))
                self.prompt_cards[section_name] = card
                
        except Exception as e:
            ttk.Label(
                self.scrollable_frame,
                text=f"Error loading prompts: {e}",
                foreground="red"
            ).pack(pady=20)
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse markdown content into sections by # headers."""
        sections = {}
        
        # Split by top-level headers (# SectionName)
        pattern = r'^# (.+)$'
        parts = re.split(pattern, content, flags=re.MULTILINE)
        
        # parts[0] is content before first header (usually empty)
        # Then alternating: header, content, header, content...
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                section_name = parts[i].strip()
                section_content = parts[i + 1]
                sections[section_name] = section_content
        
        return sections
    
    def on_back(self):
        """Navigate back to Paper Draft screen."""
        from .paper_draft_screen import PaperDraftScreen
        self.controller.show_frame(PaperDraftScreen)
