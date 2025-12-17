import tkinter as tk
from tkinter import ttk, messagebox
from typing import override
from ..base_frame import BaseFrame, create_scrollable_text_area
from phases.paper_writing.section_guidelines import SectionGuidelinesLoader
from phases.paper_writing.data_models import Section

class SectionGuidelinesScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(
            parent=parent,
            controller=controller,
            title="Section Writing Guidelines",
            next_text="Save", # Replaces "Next" button text
            has_back=True
        )
        self.text_areas = {}

    def create_content(self):
        pass

    def _create_section_editor(self, section_name: str, section_enum: Section, content: str):
        # Check if we already have this section frame
        frame_name = f"section_frame_{section_enum.name}"
        
        # Determine valid parent (scrollable_frame might be re-created or not, we should use self.scrollable_frame)
        container = ttk.Frame(self.scrollable_frame)
        container.pack(fill="x", expand=True, pady=10)
        
        # Title
        ttk.Label(container, text=section_name, font=self.controller.fonts.sub_header_font).pack(anchor="w", pady=(0, 5))

        # Create scrollable text area using the helper for consistent styling
        container_frame, text_area = create_scrollable_text_area(container, height=6)
        container_frame.pack(fill="both", expand=True) # Helper returns a frame containing text+scrollbar
        
        text_area.insert("1.0", content)
        
        self.text_areas[section_enum] = text_area

    @override
    def on_show(self):
        # Clear existing content if any (to avoid duplicates if we re-create)
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.text_areas = {}
        
        # Load guidelines
        guidelines = SectionGuidelinesLoader.load_guidelines()
        
        # Define display order
        ordered_sections = [
            Section.ABSTRACT, Section.INTRODUCTION, Section.RELATED_WORK,
            Section.METHODS, Section.RESULTS, Section.DISCUSSION, 
            Section.CONCLUSION, Section.ACKNOWLEDGEMENTS
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
        # Do NOT proceed to next screen (stay here)
        
    @override 
    def on_back(self):
        # Go back to Settings screen specifically
        from .settings_screen import SettingsScreen
        self.controller.show_frame(SettingsScreen)

