import tkinter as tk
from tkinter import ttk
from typing import override
from ..base_frame import BaseFrame
from utils.lm_studio_client import get_model_names

class SettingsScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.llm_models = get_model_names("llm")
        self.embedding_models = get_model_names("embedding")
        self.vision_models = get_model_names("llm", vision_only=True) 
        
        self.settings_vars = {}
        self.author_frames = []
        
        super().__init__(parent, controller, title="Settings", next_text="Save & Continue", has_back=False)

    def create_content(self):
        # Container frame to center the content
        container = ttk.Frame(self.content_frame)
        container.pack(fill="both", expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel scrolling - better approach: bind to parent window
        # and find the scrollable widget under the mouse
        def find_scrollable_widget(widget):
            """Walk up the widget hierarchy to find the first scrollable widget (Canvas)."""
            while widget:
                if isinstance(widget, tk.Canvas):
                    return widget
                widget = widget.master
            return None
        
        def on_mousewheel(event):
            # Get the widget under the mouse cursor
            x, y = event.x_root, event.y_root
            root = self.controller
            widget = root.winfo_containing(x, y)
            
            if not widget:
                return
            
            # Widgets that should handle their own scroll/mousewheel events
            interactive_widgets = (ttk.Combobox, ttk.Spinbox, tk.Spinbox, tk.Listbox, tk.Text)
            if isinstance(widget, interactive_widgets):
                return  # Let widget handle it natively
            
            # Find the scrollable canvas in the widget hierarchy
            scrollable = find_scrollable_widget(widget)
            if scrollable == canvas:
                # Only scroll if it's our canvas
                if hasattr(event, 'delta') and event.delta:
                    # Windows
                    scrollable.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif hasattr(event, 'num'):
                    if event.num == 4:
                        # Linux - scroll up
                        scrollable.yview_scroll(-1, "units")
                    elif event.num == 5:
                        # Linux - scroll down
                        scrollable.yview_scroll(1, "units")
        
        # Bind to the root window (parent window)
        root = self.controller  # PaperGeneratorApp (tk.Tk)
        root.bind_all("<MouseWheel>", on_mousewheel)  # Windows
        root.bind_all("<Button-4>", on_mousewheel)   # Linux scroll up
        root.bind_all("<Button-5>", on_mousewheel)   # Linux scroll down

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Update canvas window width to match canvas width and constrain to max 800px, centered
        def update_canvas_window(event):
            canvas_width = event.width
            max_width = 600
            content_width = min(canvas_width, max_width)
            x_offset = (canvas_width - content_width) // 2 if canvas_width > max_width else 0
            
            canvas_items = canvas.find_all()
            if canvas_items:
                canvas.itemconfig(canvas_items[0], width=content_width)
                canvas.coords(canvas_items[0], x_offset, 0)
        
        canvas.bind("<Configure>", update_canvas_window)

        # Guidance
        guidance_frame = ttk.LabelFrame(self.scrollable_frame, text="Info", padding="10")
        guidance_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(guidance_frame, text="Configure the models for each phase of the paper generation process.\nEnsure LM Studio is running to populate model lists.", wraplength=600).pack(anchor="w")

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
        self.create_phase_section("Paper Writing", [
            ("PAPER_INDEXING_EMBEDDING_MODEL", "Paper Indexing Embedding Model", "dropdown", self.embedding_models),
            ("EVIDENCE_GATHERING_MODEL", "Evidence Gathering Model", "dropdown", self.llm_models),
            ("PAPER_WRITING_MODEL", "Paper Writing Model", "dropdown", self.llm_models),
            ("PAPER_EMBEDDING_BATCH_SIZE", "Paper Embedding Batch Size", "spinbox", (1, 128)),
            ("EVIDENCE_INITIAL_CHUNKS", "Evidence Initial Chunks", "spinbox", (1, 20)),
            ("EVIDENCE_FILTERED_CHUNKS", "Evidence Filtered Chunks", "spinbox", (1, 20)),
            ("EVIDENCE_AGENTIC_ITERATIONS", "Evidence Agentic Iterations", "spinbox", (1, 5)),
        ])

        # LaTeX Generation Section (combines model and data)
        self.create_latex_generation_section()

    def create_phase_section(self, title, settings):
        from settings import Settings
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding="10")
        frame.pack(fill="x", padx=10, pady=5)

        for setting in settings:
            # Handle different setting definitions
            if len(setting) == 3:
                key, label_text, options = setting
                setting_type = "dropdown"
            else:
                key, label_text, setting_type, extra = setting
                
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill="x", pady=2)
            
            ttk.Label(row_frame, text=label_text, width=50).pack(side="left")
            
            var = tk.StringVar() # Or IntVar for spinbox, but StringVar works generally
            
            # Get current value from Settings
            current_value = getattr(Settings, key, "")
            
            if setting_type == "dropdown":
                dropdown = ttk.Combobox(row_frame, textvariable=var, values=options if len(setting) == 3 else extra, state="readonly")
                dropdown.pack(side="right", fill="x", expand=True)
                
                # Set current value if valid, else default to first
                values = options if len(setting) == 3 else extra
                if current_value in values:
                    dropdown.set(current_value)
                elif values:
                    dropdown.current(0)
                    
            elif setting_type == "spinbox":
                min_val, max_val = extra
                spinbox = ttk.Spinbox(row_frame, from_=min_val, to=max_val, textvariable=var)
                spinbox.pack(side="right", fill="x", expand=True)
                var.set(current_value if current_value else min_val)
            
            self.settings_vars[key] = var

    def create_latex_generation_section(self):
        from settings import Settings
        frame = ttk.LabelFrame(self.scrollable_frame, text="LaTeX Generation", padding="10")
        frame.pack(fill="x", padx=10, pady=5)

        # LaTeX Generation Model
        row_frame = ttk.Frame(frame)
        row_frame.pack(fill="x", pady=2)
        ttk.Label(row_frame, text="LaTeX Generation Model", width=40).pack(side="left")
        
        var = tk.StringVar()
        current_value = getattr(Settings, "LATEX_GENERATION_MODEL", "")
        dropdown = ttk.Combobox(row_frame, textvariable=var, values=self.llm_models, state="readonly")
        dropdown.pack(side="right", fill="x", expand=True)
        
        if current_value in self.llm_models:
            dropdown.set(current_value)
        elif self.llm_models:
            dropdown.current(0)
        
        self.settings_vars["LATEX_GENERATION_MODEL"] = var

        # Title
        row_frame = ttk.Frame(frame)
        row_frame.pack(fill="x", pady=2)
        ttk.Label(row_frame, text="Paper Title", width=40).pack(side="left")
        self.title_var = tk.StringVar(value=Settings.LATEX_TITLE)
        entry = ttk.Entry(row_frame, textvariable=self.title_var)
        entry.pack(side="right", fill="x", expand=True)
        # Placeholder logic could be added here or just label text
        ttk.Label(frame, text="(Leave empty for LLM generated title)", font=("Helvetica", 8, "italic")).pack(anchor="e")

        # Authors
        self.authors_container = ttk.Frame(frame)
        self.authors_container.pack(fill="x", pady=5)
        
        ttk.Label(self.authors_container, text="Authors").pack(anchor="w")
        
        self.add_author_btn = ttk.Button(frame, text="Add Author", command=self.add_author)
        self.add_author_btn.pack(pady=5)

        # Load authors from Settings
        if Settings.LATEX_AUTHORS:
            for author_data in Settings.LATEX_AUTHORS:
                self.add_author(author_data)
        else:
            # Add at least one empty author if none exist
            self.add_author()

    def add_author(self, data=None):
        author_frame = ttk.Frame(self.authors_container, padding="5", borderwidth=1, relief="solid")
        author_frame.pack(fill="x", pady=2)
        
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

        # Remove button
        if len(self.author_frames) > 0:
            remove_btn = ttk.Button(author_frame, text="Remove", command=lambda f=author_frame: self.remove_author(f))
            remove_btn.pack(anchor="e", pady=2)
        
        self.author_frames.append((author_frame, entries))

    def remove_author(self, frame):
        frame.destroy()
        # Remove from list
        self.author_frames = [af for af in self.author_frames if af[0] != frame]

    
    @override
    def on_next(self):
        # Save settings to Settings class
        from settings import Settings
        
        # Update simple settings
        for key, var in self.settings_vars.items():
            value = var.get()
            # Convert to int if it's a numeric setting
            if key in ["PAPER_EMBEDDING_BATCH_SIZE", "EVIDENCE_INITIAL_CHUNKS", "EVIDENCE_FILTERED_CHUNKS", "EVIDENCE_AGENTIC_ITERATIONS"]:
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

        # Save settings to file for persistence
        Settings.save_to_file()

        # Proceed to next screen
        super().on_next()
