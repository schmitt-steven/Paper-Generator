import tkinter as tk
from tkinter import ttk
from typing import override
from ..base_frame import BaseFrame
from utils.lm_studio_client import get_model_names
from .section_guidelines_screen import SectionGuidelinesScreen

class SettingsScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.llm_models = get_model_names(model_type="llm")
        self.embedding_models = get_model_names(model_type="embedding")
        self.vision_models = get_model_names(model_type="llm", vision_only=True) 
        
        self.settings_vars = {}
        self.author_frames = []
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Settings",
            next_text="Continue",
            has_back=False
        )

    def create_content(self):
        # Appearance (Font Size) - Moved to top
        self.create_appearance_section()
        # Info
        #guidance_frame = ttk.LabelFrame(self.scrollable_frame, text="Info", padding="10")
        #guidance_frame.pack(fill="x", padx=0, pady=5)
        #ttk.Label(guidance_frame, text="Configure the models for each phase of the paper generation process.\nEnsure LM Studio is running to populate model lists.", wraplength=600).pack(anchor="w")

        # Context Analysis Phase
        self.create_phase_section("Context Analysis", [
            ("CODE_ANALYSIS_MODEL", "Code Analysis Model", self.llm_models),
            ("NOTES_ANALYSIS_MODEL", "Notes Analysis Model", self.llm_models),
            ("PAPER_CONCEPTION_MODEL", "Paper Conception Model", self.llm_models),
        ])

        # Paper Search Phase
        self.create_phase_section("Paper Search", [
            ("LITERATURE_SEARCH_MODEL", "Literature Search Model", self.llm_models),
            ("PAPER_ANALYSIS_MODEL", "Paper Analysis Model", self.llm_models),
            ("LIMITATION_ANALYSIS_EMBEDDING_MODEL", "Limitation Analysis Embedding Model", self.embedding_models),
            ("PAPER_RANKING_EMBEDDING_MODEL", "Paper Ranking Embedding Model", self.embedding_models),
        ])

        # Hypothesis Generation Phase
        self.create_phase_section("Hypothesis Generation", [
            ("HYPOTHESIS_BUILDER_MODEL", "Hypothesis Builder Model", self.llm_models),
            ("HYPOTHESIS_BUILDER_EMBEDDING_MODEL", "Hypothesis Embedding Model", self.embedding_models),
        ])

        # Experimentation Phase
        self.create_phase_section("Experimentation", [
            ("EXPERIMENT_PLAN_MODEL", "Planning Model", self.llm_models),
            ("EXPERIMENT_CODE_WRITE_MODEL", "Coding Model", self.llm_models),
            ("EXPERIMENT_CODE_FIX_MODEL", "Error Fixing Model", self.llm_models),
            ("EXPERIMENT_CODE_IMPROVE_MODEL", "Code Improvement Model", self.llm_models),
            ("EXPERIMENT_VALIDATION_MODEL", "Validation Model", self.llm_models),
            ("EXPERIMENT_PLOT_CAPTION_MODEL", "Plot Caption Model (Vision)", self.vision_models),
            ("EXPERIMENT_VERDICT_MODEL", "Verdict Model", self.llm_models),
        ])

        # Paper Writing Phase
        # Paper Writing Phase
        paper_writing_frame = self.create_phase_section("Paper Writing", [
            ("PAPER_INDEXING_EMBEDDING_MODEL", "Paper Indexing Embedding Model", "dropdown", self.embedding_models),
            ("EVIDENCE_GATHERING_MODEL", "Evidence Gathering Model", "dropdown", self.llm_models),
            ("PAPER_WRITING_MODEL", "Paper Writing Model", "dropdown", self.llm_models),
            ("PAPER_EMBEDDING_BATCH_SIZE", "Paper Embedding Batch Size", "spinbox", (1, 128)),
            ("EVIDENCE_INITIAL_CHUNKS", "Evidence Initial Chunks", "spinbox", (1, 20)),
            ("EVIDENCE_FILTERED_CHUNKS", "Evidence Filtered Chunks", "spinbox", (1, 20)),
            ("EVIDENCE_AGENTIC_ITERATIONS", "Evidence Agentic Iterations", "spinbox", (1, 5)),
        ])
        
        # Add Edit Section Guidelines button to Paper Writing section
        # Add Edit Section Guidelines button to Paper Writing section
        # ttk.Separator(paper_writing_frame, orient="horizontal").pack(fill="x", pady=10)
        
        row_frame = ttk.Frame(paper_writing_frame)
        row_frame.pack(fill="x", pady=2)
        
        ttk.Label(row_frame, text="Writing Guidelines", width=35).pack(side="left")
        
        guidelines_btn = ttk.Button(row_frame, text="Edit", command=lambda: self.controller.show_frame(SectionGuidelinesScreen))
        guidelines_btn.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # LaTeX Generation Section (combines model and data)
        self.create_latex_generation_section()





    def create_phase_section(self, title, settings):
        from settings import Settings
        
        # content container from card
        frame = self.create_card_frame(self.scrollable_frame, title)



        for setting in settings:
            # Handle different setting definitions
            if len(setting) == 3:
                key, label_text, options = setting
                setting_type = "dropdown"
            else:
                key, label_text, setting_type, extra = setting
                
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill="x", pady=2)
            
            ttk.Label(row_frame, text=label_text, width=35).pack(side="left")
            
            var = tk.StringVar() # Or IntVar for spinbox, but StringVar works generally
            
            # Get current value from Settings
            current_value = getattr(Settings, key, "")
            
            if setting_type == "dropdown":
                dropdown = ttk.Combobox(row_frame, textvariable=var, values=options if len(setting) == 3 else extra, state="readonly", width=60)
                dropdown.pack(side="right", fill="x", expand=True, padx=(10, 0))
                
                # Set current value if valid, else default to first
                values = options if len(setting) == 3 else extra
                if current_value in values:
                    dropdown.set(current_value)
                elif values:
                    dropdown.current(0)
                    
            elif setting_type == "spinbox":
                min_val, max_val = extra
                spinbox = ttk.Spinbox(row_frame, from_=min_val, to=max_val, textvariable=var)
                spinbox.pack(side="right", fill="x", expand=True, padx=(10, 0))
                var.set(current_value if current_value else min_val)
            
            self.settings_vars[key] = var
        return frame

    def create_latex_generation_section(self):
        from settings import Settings
        
        # content container from card (for models and title)
        frame = self.create_card_frame(self.scrollable_frame, "LaTeX Generation")



        # LaTeX Generation Model
        row_frame = ttk.Frame(frame)
        row_frame.pack(fill="x", pady=2)
        
        ttk.Label(row_frame, text="LaTeX Generation Model", width=35).pack(side="left")
        
        var = tk.StringVar()
        current_value = getattr(Settings, "LATEX_GENERATION_MODEL", "")
        dropdown = ttk.Combobox(row_frame, textvariable=var, values=self.llm_models, state="readonly", width=60)
        dropdown.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        if current_value in self.llm_models:
            dropdown.set(current_value)
        elif self.llm_models:
            dropdown.current(0)
        
        self.settings_vars["LATEX_GENERATION_MODEL"] = var

        # Title
        row_frame = ttk.Frame(frame)
        row_frame.pack(fill="x", pady=2)
        ttk.Label(row_frame, text="Paper Title", width=35).pack(side="left")
        self.title_var = tk.StringVar(value=Settings.LATEX_TITLE)
        entry = ttk.Entry(row_frame, textvariable=self.title_var, width=60)
        entry.pack(side="right", fill="x", expand=True, padx=(10, 0))
        ttk.Label(frame, text="(Leave empty for LLM generated title)", font=self.controller.fonts.default_font).pack(anchor="e")

        # Authors section (Separate Card)
        authors_section = ttk.Frame(self.scrollable_frame, style="Card.TFrame", padding=1)
        authors_section.pack(fill="x", padx=0, pady=(20, 10))
        
        # Header row
        header_frame = ttk.Frame(authors_section, padding=10)
        header_frame.pack(fill="x")
        
        ttk.Label(header_frame, text="Authors", font=self.controller.fonts.sub_header_font).pack(side="left")
        
        # Buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side="right")
        
        self.remove_author_btn = ttk.Button(button_frame, text="Remove", command=self.remove_last_author)
        self.remove_author_btn.pack(side="left", padx=(0, 5))
        
        self.add_author_btn = ttk.Button(button_frame, text="Add", command=self.add_author)
        self.add_author_btn.pack(side="left")
        
        # Separator
        ttk.Separator(authors_section, orient="horizontal").pack(fill="x")
        
        # Authors container
        self.authors_container = ttk.Frame(authors_section, padding=10)
        self.authors_container.pack(fill="x")

        # Load authors from Settings
        if Settings.LATEX_AUTHORS:
            for author_data in Settings.LATEX_AUTHORS:
                self.add_author(author_data)
        else:
            # Add at least one empty author if none exist
            self.add_author()
        
        # Initialize button state
        self._update_remove_button_state()
        
    def create_appearance_section(self):
        from settings import Settings
        
        # content container from card
        frame = self.create_card_frame(self.scrollable_frame, "Appearance")


        
        row_frame = ttk.Frame(frame)
        row_frame.pack(fill="x", pady=2)
        
        ttk.Label(row_frame, text="Font Size", width=35).pack(side="left")
        
        self.font_size_var = tk.IntVar(value=getattr(Settings, "FONT_SIZE_BASE", 16))
        
        # Helper to update font immediately
        def on_font_size_change(*args):
            try:
                val = self.font_size_var.get()
                self.controller.fonts.update_base_size(int(val))
            except ValueError:
                pass

        self.font_size_var.trace_add("write", on_font_size_change)
        
        # Spinbox
        spinbox = ttk.Spinbox(
            row_frame, 
            from_=8, 
            to=32, 
            textvariable=self.font_size_var, 
            width=5
        )
        spinbox.pack(side="right", expand=True, fill="x", padx=(10, 0))
        
        self.settings_vars["FONT_SIZE_BASE"] = self.font_size_var

        # Theme Toggle (Switch)
        row_frame = ttk.Frame(frame)
        row_frame.pack(fill="x", pady=(10, 2))
        
        ttk.Label(row_frame, text="Dark Mode", width=35).pack(side="left")
        
        self.dark_mode_var = tk.BooleanVar(value=True)  # Start in dark mode
        
        def on_toggle():
            self.controller.toggle_theme()
        
        switch = ttk.Checkbutton(
            row_frame, 
            variable=self.dark_mode_var,
            style="Switch.TCheckbutton",
            command=on_toggle
        )
        switch.pack(side="right", padx=(10, 0))



    def add_author(self, data=None):
        # Add separator if not the first author
        if len(self.author_frames) > 0:
            ttk.Separator(self.authors_container, orient="horizontal").pack(fill="x", padx=10)
        
        author_frame = ttk.Frame(self.authors_container, padding="10")
        author_frame.pack(fill="x", pady=5)
        
        fields = ["Name", "Affiliation", "Department", "Address", "Email"]
        entries = {}
        
        for field in fields:
            row = ttk.Frame(author_frame)
            row.pack(fill="x")
            ttk.Label(row, text=field, width=15).pack(side="left")
            entry = ttk.Entry(row)
            entry.pack(side="right", fill="x", expand=True)
            if data:
                # Field names in Settings are lowercase keys
                entry.insert(0, data.get(field.lower(), ""))
            entries[field.lower()] = entry
        
        self.author_frames.append((author_frame, entries))
        self._update_remove_button_state()

    def remove_last_author(self):
        """Remove the last added author."""
        if len(self.author_frames) <= 1:
            return  # Don't remove if only one author
        
        # Get the last author frame and its separator (if exists)
        last_frame, _ = self.author_frames[-1]
        
        # Find and remove the separator before this frame (if it exists)
        for widget in self.authors_container.winfo_children():
            if isinstance(widget, ttk.Separator):
                # Check if this separator is right before the last frame
                widget_index = self.authors_container.winfo_children().index(widget)
                frame_index = self.authors_container.winfo_children().index(last_frame)
                if widget_index == frame_index - 1:
                    widget.destroy()
                    break
        
        # Remove the frame
        last_frame.destroy()
        self.author_frames.pop()
        self._update_remove_button_state()
    
    def _update_remove_button_state(self):
        """Update the remove button state based on number of authors."""
        if len(self.author_frames) > 1:
            self.remove_author_btn.config(state="normal")
        else:
            self.remove_author_btn.config(state="disabled")

    
    @override
    def on_next(self):
        # Save settings to Settings class
        from settings import Settings
        
        # Update simple settings
        for key, var in self.settings_vars.items():
            value = var.get()
            # Convert to int if it's a numeric setting
            if key in ["PAPER_EMBEDDING_BATCH_SIZE",
                       "EVIDENCE_INITIAL_CHUNKS",
                       "EVIDENCE_FILTERED_CHUNKS",
                       "EVIDENCE_AGENTIC_ITERATIONS",
                       "LATEX_GENERATION_MODEL",
                       "FONT_SIZE_BASE"]:
                try:
                    value = int(value)
                except ValueError:
                    print(f"Warning: Invalid integer for {key}: {value}")
                    continue
            
            if hasattr(Settings, key):
                setattr(Settings, key, value)
                # print(f"Updated {key} to {value}")
            else:
                print(f"Warning: Setting {key} not found in Settings class!")

        # Update LaTeX Title
        Settings.LATEX_TITLE = self.title_var.get()

        # Update Authors
        authors = []
        for _, entries in self.author_frames:
            author_data = {}
            for field, entry in entries.items():
                author_data[field] = entry.get()
            authors.append(author_data)
        
        Settings.LATEX_AUTHORS = authors

        # Save settings to file
        Settings.save_to_file()

        # Proceed to next screen
        super().on_next()
