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
        frame_name = f"section_frame_{section_enum.name}"
        
        # Get valid parent
        container = ttk.Frame(self.scrollable_frame, style="Scrollable.TFrame")
        container.pack(fill="x", expand=True, pady=10)
        
        # Title
        ttk.Label(container, text=section_name, style="Scrollable.TLabel", font=self.controller.fonts.sub_header_font).pack(anchor="w", pady=(0, 5))

        # Create scrollable text area
        container_frame, text_area = create_scrollable_text_area(container, height=6)
        container_frame.pack(fill="both", expand=True) # Helper returns a frame containing text+scrollbar
        
        text_area.insert("1.0", content)
        
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

