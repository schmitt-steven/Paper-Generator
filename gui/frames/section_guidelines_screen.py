import tkinter as tk
from tkinter import ttk, messagebox
from typing import override
from ..base_frame import BaseFrame, create_scrollable_text_area
from ..info_texts import SECTION_GUIDELINES_INFO
from phases.paper_writing.section_guidelines import SectionGuidelinesLoader
from phases.paper_writing.data_models import Section

class SectionGuidelinesScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.file_path = "user_files/section_guidelines.md"
        super().__init__(
            parent=parent,
            controller=controller,
            title="Section Writing Guidelines",
            next_text="Save",
            has_back=True,
            header_file_path=self.file_path,
            info_content=SECTION_GUIDELINES_INFO
        )
        self.text_areas = {}

    def create_content(self):
        pass

    def _create_section_editor(self, section_name: str, section_enum: Section, content: str):
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
            text=section_name, 
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
        
        text_area = tk.Text(
            inner,
            height=12,
            wrap="word",
            font=self.controller.fonts.text_area_font,
            padx=10,
            pady=10,
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            yscrollcommand=scrollbar.set
        )
        text_area.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_area.yview)
        
        text_area.insert("1.0", content)
        
        self.controller.apply_theme_colors(text_area)
        
        self.text_areas[section_enum] = text_area

    @override
    def on_show(self):
        # Clear existing content (to avoid duplicates)
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.text_areas = {}
        
        guidelines = SectionGuidelinesLoader.load_guidelines()
        
        ordered_sections = [
            Section.ABSTRACT,
            Section.INTRODUCTION,
            Section.RELATED_WORK,
            Section.METHODS,
            Section.RESULTS,
            Section.DISCUSSION,
            Section.CONCLUSION,
            Section.ACKNOWLEDGEMENTS
        ]
        
        for section in ordered_sections:
            content = guidelines.get(section, "")
            self._create_section_editor(section.value.title(), section, content)

    @override
    def on_next(self):
        # Save content
        new_guidelines = {}
        for section, text_area in self.text_areas.items():
            content = text_area.get("1.0", "end-1c")
            if content.strip():
                new_guidelines[section] = content
        
        SectionGuidelinesLoader.save_guidelines(new_guidelines)
        
        messagebox.showinfo("Saved", "Section guidelines have been saved successfully.")
        
    @override 
    def on_back(self):
        from .settings_screen import SettingsScreen
        self.controller.show_frame(SettingsScreen)

