from tkinter import ttk
from ..base_frame import BaseFrame, create_scrollable_text_area
import os

class UserRequirementsScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.file_path = "user_files/user_requirements.md"
        self.section_widgets = {}
        self.sections = []
        super().__init__(
            parent=parent,
            controller=controller,
            title="User Requirements",
            next_text="Continue",
            header_file_path=self.file_path
        )

    def create_content(self):
        self.sections = []  # List of tuples (header_line, text_widget)
        self.load_and_parse()

    def load_and_parse(self):
        if not os.path.exists(self.file_path):
            ttk.Label(self.scrollable_frame, text=f"File not found: {self.file_path}").pack()
            return

        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_header = None
        current_text = []

        def add_section_ui(header, text_lines):
            # Skip if header is None and text is empty (avoids duplicate default section)
            if header is None and not "".join(text_lines).strip():
                return

            # Determine title for LabelFrame
            if header:
                title = header.strip().lstrip("#").strip()
            else:
                title = "General Information"
            
            # Special handling for grouping headers
            if title in ["General Information", "Section Specifications"]:
                frame = ttk.Frame(self.scrollable_frame, padding=(0, 10))
                frame.pack(fill="x")
                ttk.Label(frame, text=title, font=self.controller.fonts.header_font).pack(anchor="w")
                # ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=5)
                
                self.sections.append((header, None))
                return

            # Container for the section
            section_container = ttk.Frame(self.scrollable_frame, padding=(0, 0, 0, 15))
            section_container.pack(fill="x")
            
            # Standalone Header
            ttk.Label(
                section_container, 
                text=title, 
                font=self.controller.fonts.sub_header_font
            ).pack(anchor="w", pady=(0, 10))
            
            container, text_widget = create_scrollable_text_area(section_container, height=6)
            container.pack(fill="x", expand=True, padx=(15, 0))
            text_widget.insert("1.0", "".join(text_lines).strip())
            
            self.sections.append((header, text_widget))

        for line in lines:
            if line.strip().startswith("#"):
                # New section
                add_section_ui(current_header, current_text)
                current_header = line # Keep the newline and # characters
                current_text = []
            else:
                current_text.append(line)
        
        # Add last section
        add_section_ui(current_header, current_text)

    def on_next(self):
        new_content = []
        for header, text_widget in self.sections:
            if text_widget:
                text = text_widget.get("1.0", "end-1c").strip()
            else:
                text = ""
            
            if header:
                new_content.append(header.strip() + "\n")
            
            if text:
                new_content.append(text + "\n\n")
            elif header:
                 # Ensure spacing if text is empty but header exists
                 new_content.append("\n")

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.writelines(new_content)
            print(f"Saved changes to {self.file_path}")
        except Exception as e:
            print(f"Failed to save file: {e}")

        super().on_next()
