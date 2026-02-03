import tkinter as tk
from tkinter import ttk, messagebox
import shutil
from pathlib import Path
from settings import Settings, FontSize, get_available_templates

from ..base_frame import BaseFrame, CardBorderFrame
from ..info_texts import SETTINGS_INFO, LATEX_TEMPLATE_INFO
from ..theme_colors import CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT
from utils.lm_studio_client import get_model_names


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
            has_next=False,
            has_back=True,
            back_text="Save & Back",
            info_content=SETTINGS_INFO
        )

    def create_content(self):
        # Section 1: General
        self.create_general_section()
        
        # Section 2: LaTeX Template (placeholder)
        self.create_latex_template_section()
        
        # Section 3: Authors
        self.create_authors_section()
        
        # Section 4: Model Selection (all models)
        self.create_model_selection_section()
        
        # Section 5: Appearance
        self.create_appearance_section()
        
        # Clear Cache button at the bottom (standalone, red)
        self.create_clear_cache_button()

    def create_clear_cache_button(self):
        """Standalone Clear Cache button at the bottom"""
        clear_btn = ttk.Button(
            self.scrollable_frame,
            text="Clear Cache",
            command=self.clear_cache
        )
        clear_btn.pack(fill="x", pady=(10, 20))

    def create_general_section(self):
        """General section: Paper Title, API Key"""
        frame = self.create_card_frame(self.scrollable_frame, "General")

        # Paper Title
        row_frame = ttk.Frame(frame, style="CardRow.TFrame")
        row_frame.pack(fill="x", pady=2)
        ttk.Label(row_frame, text="Paper Title", width=35, style="CardRow.TLabel").pack(side="left")
        self.title_var = tk.StringVar(value=Settings.LATEX_TITLE)
        entry = ttk.Entry(row_frame, textvariable=self.title_var, width=60)
        entry.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # Acknowledgements Toggle
        row_frame = ttk.Frame(frame, style="CardRow.TFrame")
        row_frame.pack(fill="x", pady=(10, 2))
        
        ttk.Label(row_frame, text="Include Acknowledgements", width=35, style="CardRow.TLabel").pack(side="left")
        
        self.acknowledgements_var = tk.BooleanVar(value=getattr(Settings, "GENERATE_ACKNOWLEDGEMENTS", True))
        
        switch = ttk.Checkbutton(
            row_frame,
            variable=self.acknowledgements_var,
            style="CardRow.Switch.TCheckbutton"
        )
        switch.pack(side="right", padx=(10, 0))
        
        self.settings_vars["GENERATE_ACKNOWLEDGEMENTS"] = self.acknowledgements_var


        # Semantic Scholar API Key
        row_frame = ttk.Frame(frame, style="CardRow.TFrame")
        row_frame.pack(fill="x", pady=(10, 2))
        
        ttk.Label(row_frame, text="Semantic Scholar API Key", width=35, style="CardRow.TLabel").pack(side="left")
        
        self.api_key_var = tk.StringVar(value=getattr(Settings, "SEMANTIC_SCHOLAR_API_KEY", ""))
        api_key_entry = ttk.Entry(row_frame, textvariable=self.api_key_var, show="•")
        api_key_entry.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        self.settings_vars["SEMANTIC_SCHOLAR_API_KEY"] = self.api_key_var

        # User Email (for Unpaywall API)
        row_frame = ttk.Frame(frame, style="CardRow.TFrame")
        row_frame.pack(fill="x", pady=(10, 2))
        
        ttk.Label(row_frame, text="Unpaywall Email", width=35, style="CardRow.TLabel").pack(side="left")
        
        self.unpaywall_email_var = tk.StringVar(value=getattr(Settings, "UNPAYWALL_EMAIL", ""))
        email_entry = ttk.Entry(row_frame, textvariable=self.unpaywall_email_var)
        email_entry.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        self.settings_vars["UNPAYWALL_EMAIL"] = self.unpaywall_email_var

    def create_latex_template_section(self):
        """LaTeX Template section: radio buttons for template selection"""
        frame = self.create_card_frame(self.scrollable_frame, "LaTeX Template", info_content=LATEX_TEMPLATE_INFO)

        # Get available templates
        templates = get_available_templates()
        
        if not templates:
            row_frame = ttk.Frame(frame, style="CardRow.TFrame")
            row_frame.pack(fill="x", pady=2)
            ttk.Label(row_frame, text="No templates found in latex_templates/", 
                      style="CardRow.TLabel").pack(side="left")
            return
        
        # Template variable
        self.template_var = tk.StringVar(value=Settings.LATEX_TEMPLATE)
        
        # Some know acronyms that should be all uppercase
        acronyms = {"ieee", "jair", "acm", "aaai", "icml", "neurips", "cvpr", "iclr"}
        
        # Create radio button for each template
        for template in templates:
            row_frame = ttk.Frame(frame, style="CardRow.TFrame")
            row_frame.pack(fill="x", pady=5)
            
            # Format template name for display (e.g., "ieee_transaction" -> "IEEE Transaction")
            words = template.replace("_", " ").split()
            display_name = " ".join(
                word.upper() if word.lower() in acronyms else word.capitalize() 
                for word in words
            )
            
            radio = ttk.Radiobutton(
                row_frame,
                text="",
                variable=self.template_var,
                value=template,
                style="CardRow.TCheckbutton"
            )
            radio.pack(side="left", padx=(10, 0))
            
            label = ttk.Label(
                row_frame,
                text=display_name,
                style="CardRow.TLabel",
                cursor="hand2"
            )
            label.pack(side="left", padx=(8, 10))
            
            # Make label clickable to select the radio button
            label.bind("<Button-1>", lambda e, t=template: self.template_var.set(t))

    def create_authors_section(self):
        """Authors section with add/remove functionality"""
        authors_section = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        authors_section.pack(fill="x", padx=0, pady=10)
        
        # Header row
        header_frame = ttk.Frame(authors_section, style="CardHeader.TFrame", padding=10)
        header_frame.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        tk.Label(
            header_frame, 
            text="Authors", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Buttons
        button_frame = ttk.Frame(header_frame, style="CardHeader.TFrame")
        button_frame.pack(side="right")
        
        self.remove_author_btn = ttk.Button(button_frame, text="Remove", command=self.remove_last_author)
        self.remove_author_btn.pack(side="left", padx=(0, 5))
        
        self.add_author_btn = ttk.Button(button_frame, text="Add", command=self.add_author)
        self.add_author_btn.pack(side="left")
        
        # Separator
        ttk.Separator(authors_section, orient="horizontal").pack(fill="x")
        
        # Authors container
        self.authors_container = ttk.Frame(authors_section, style="CardContent.TFrame", padding=10)
        self.authors_container.pack(fill="x")

        # Load authors from Settings
        if Settings.LATEX_AUTHORS:
            for author_data in Settings.LATEX_AUTHORS:
                self.add_author(author_data)
        else:
            self.add_author()
        
        self._update_remove_button_state()

    def create_model_selection_section(self):
        """Model Selection section: all models in a flat list"""
        frame = self.create_card_frame(self.scrollable_frame, "Model Selection")

        models = [
            ("CODE_ANALYSIS_MODEL", "Code Analyzer", self.llm_models),
            ("CONTEXT_GENERATOR_MODEL", "Research Context Generator", self.llm_models),
            ("LITERATURE_SEARCH_MODEL", "Literature Searcher", self.llm_models),
            ("PAPER_RANKING_EMBEDDING_MODEL", "Paper Ranking Embedding Model", self.embedding_models),
            ("HYPOTHESIS_BUILDER_MODEL", "Hypothesis Generator", self.llm_models),
            ("EXPERIMENT_PLAN_MODEL", "Experiment Planner", self.llm_models),
            ("EXPERIMENT_CODE_WRITE_MODEL", "Experiment Coder", self.llm_models),
            ("EXPERIMENT_VALIDATION_MODEL", "Experiment Validator", self.llm_models),
            ("EXPERIMENT_PLOT_CAPTION_MODEL", "Plot Caption Generator (Vision)", self.vision_models),
            ("EXPERIMENT_VERDICT_MODEL", "Experiment Verdict Generator", self.llm_models),
            ("PAPER_INDEXING_EMBEDDING_MODEL", "Paper Indexing Embedding Model", self.embedding_models),
            ("PAPER_WRITING_MODEL", "Paper Writer", self.llm_models),
            ("LATEX_GENERATION_MODEL", "LaTeX Generator", self.llm_models),
        ]

        for key, label_text, options in models:
            row_frame = ttk.Frame(frame, style="CardRow.TFrame")
            row_frame.pack(fill="x", pady=2)
            
            ttk.Label(row_frame, text=label_text, width=35, style="CardRow.TLabel").pack(side="left")
            
            var = tk.StringVar()
            current_value = getattr(Settings, key, "")
            
            dropdown = ttk.Combobox(row_frame, textvariable=var, values=options, state="readonly", width=60)
            dropdown.pack(side="right", fill="x", expand=True, padx=(10, 0))
            
            if current_value in options:
                dropdown.set(current_value)
            elif options:
                dropdown.current(0)
            
            self.settings_vars[key] = var

    def create_appearance_section(self):
        """Appearance section: Font Size, Dark Mode"""
        frame = self.create_card_frame(self.scrollable_frame, "Appearance")

        # Font Size
        row_frame = ttk.Frame(frame, style="CardRow.TFrame")
        row_frame.pack(fill="x", pady=2)
        
        ttk.Label(row_frame, text="Font Size", width=35, style="CardRow.TLabel").pack(side="left")
        
        # Map enum names to labels
        self.font_size_options = {
            FontSize.VERY_SMALL: "Very Small",
            FontSize.SMALL: "Small",
            FontSize.MEDIUM: "Medium",
            FontSize.LARGE: "Large",
            FontSize.VERY_LARGE: "Very Large",
            FontSize.ULTRA_LARGE: "Ultra Large"
        }
        
        # Get current label from enum
        current_label = self.font_size_options.get(Settings.FONT_SIZE, "Small")
        
        self.font_size_var = tk.StringVar(value=current_label)
        
        # Reverse lookup: label -> enum
        self.label_to_enum = {v: k for k, v in self.font_size_options.items()}
        
        def on_font_size_change(*args):
            try:
                label = self.font_size_var.get()
                font_enum = self.label_to_enum.get(label, FontSize.SMALL)
                self.controller.fonts.update_base_size(font_enum.value)
            except Exception:
                pass

        self.font_size_var.trace_add("write", on_font_size_change)
        
        dropdown = ttk.Combobox(
            row_frame, 
            textvariable=self.font_size_var,
            values=list(self.font_size_options.values()),
            state="readonly",
            width=15
        )
        dropdown.pack(side="right", expand=True, fill="x", padx=(10, 0))
        
        self.settings_vars["FONT_SIZE_BASE"] = self.font_size_var

        # Dark Mode Toggle
        row_frame = ttk.Frame(frame, style="CardRow.TFrame")
        row_frame.pack(fill="x", pady=(10, 2))
        
        ttk.Label(row_frame, text="Dark Mode", width=35, style="CardRow.TLabel").pack(side="left")
        
        self.dark_mode_var = tk.BooleanVar(value=getattr(Settings, "DARK_MODE", True))
        
        def on_toggle():
            self.controller.toggle_theme()
            Settings.DARK_MODE = self.dark_mode_var.get()
        
        switch = ttk.Checkbutton(
            row_frame, 
            variable=self.dark_mode_var,
            style="CardRow.Switch.TCheckbutton",
            command=on_toggle
        )
        switch.pack(side="right", padx=(10, 0))

    def clear_cache(self):
        """Clear all cached files: output folder contents and non-essential user_files."""
        confirmed = messagebox.askyesno(
            "Clear Cache",
            "This will delete all files from the output folder and temporary files from user_files.\n\n"
            "Style Guidelines and paper specification will be preserved.\n\n"
            "Are you sure you want to continue?",
            icon="warning"
        )
        
        if not confirmed:
            return
        
        deleted_count = 0
        errors = []
        
        base_dir = Path(__file__).parent.parent.parent
        
        # Clear output folder
        output_dir = base_dir / "output"
        if output_dir.exists():
            for item in output_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        deleted_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        deleted_count += 1
                except Exception as e:
                    errors.append(f"{item.name}: {e}")
        
        # Clear user_files (except style_guidelines.md and paper_specification.md)
        user_files_dir = base_dir / "user_files"
        protected_files = {"style_guidelines.md", "paper_specification.md"}
        
        if user_files_dir.exists():
            for item in user_files_dir.iterdir():
                if item.name.lower() not in protected_files:
                    try:
                        if item.is_file():
                            item.unlink()
                            deleted_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            deleted_count += 1
                    except Exception as e:
                        errors.append(f"{item.name}: {e}")
        
        if errors:
            messagebox.showwarning(
                "Cache Cleared with Errors",
                f"Deleted {deleted_count} items.\n\nErrors:\n" + "\n".join(errors[:5])
            )
        else:
            messagebox.showinfo(
                "Cache Cleared",
                f"Successfully deleted {deleted_count} items."
            )

    def add_author(self, data=None):
        """Add an author entry."""
        if len(self.author_frames) > 0:
            ttk.Separator(self.authors_container, orient="horizontal").pack(fill="x", padx=10)
        
        author_frame = ttk.Frame(self.authors_container, style="CardRow.TFrame", padding="10")
        author_frame.pack(fill="x", pady=5)
        
        # Fields for both IEEE and JAIR templates
        fields = ["Name", "Affiliation", "Department", "City", "Country", "Address", "Email"]
        entries = {}
        
        for field in fields:
            row = ttk.Frame(author_frame, style="CardRow.TFrame")
            row.pack(fill="x")
            ttk.Label(row, text=field, width=15, style="CardRow.TLabel").pack(side="left")
            entry = ttk.Entry(row)
            entry.pack(side="right", fill="x", expand=True)
            if data:
                entry.insert(0, data.get(field.lower(), ""))
            entries[field.lower()] = entry
        
        self.author_frames.append((author_frame, entries))
        self._update_remove_button_state()

    def remove_last_author(self):
        """Remove the last added author."""
        if len(self.author_frames) <= 1:
            return
        
        last_frame, _ = self.author_frames[-1]
        
        for widget in self.authors_container.winfo_children():
            if isinstance(widget, ttk.Separator):
                widget_index = self.authors_container.winfo_children().index(widget)
                frame_index = self.authors_container.winfo_children().index(last_frame)
                if widget_index == frame_index - 1:
                    widget.destroy()
                    break
        
        last_frame.destroy()
        self.author_frames.pop()
        self._update_remove_button_state()
    
    def _update_remove_button_state(self):
        """Update the remove button state based on number of authors."""
        if len(self.author_frames) > 1:
            self.remove_author_btn.config(state="normal")
        else:
            self.remove_author_btn.config(state="disabled")

    def on_back(self):
        # Model settings that require selection
        model_settings = {
            "CODE_ANALYSIS_MODEL": "Code Analysis Model",
            "CONTEXT_GENERATOR_MODEL": "Research Context Model",
            "LITERATURE_SEARCH_MODEL": "Literature Search Model",
            "PAPER_RANKING_EMBEDDING_MODEL": "Paper Ranking Embedding Model",
            "HYPOTHESIS_BUILDER_MODEL": "Hypothesis Generation Model",
            "EXPERIMENT_PLAN_MODEL": "Experiment Planning Model",
            "EXPERIMENT_CODE_WRITE_MODEL": "Experiment Coding Model",
            "EXPERIMENT_VALIDATION_MODEL": "Experiment Validation Model",
            "EXPERIMENT_PLOT_CAPTION_MODEL": "Experiment Plot Caption Model",
            "EXPERIMENT_VERDICT_MODEL": "Experiment Verdict Model",
            "PAPER_INDEXING_EMBEDDING_MODEL": "Paper Indexing Embedding Model",
            "PAPER_WRITING_MODEL": "Paper Writing Model",
            "LATEX_GENERATION_MODEL": "LaTeX Generation Model",
        }
        
        # Check for missing model selections
        missing_models = []
        for key, label in model_settings.items():
            if key in self.settings_vars:
                value = self.settings_vars[key].get()
                if not value or value.strip() == "":
                    missing_models.append(label)
        
        if missing_models:
            messagebox.showwarning(
                "Missing Model Selection",
                "Please select a model for the following fields:\n\n" +
                "\n".join(f"• {model}" for model in missing_models)
            )
            return
        
        # Save model settings
        for key, var in self.settings_vars.items():
            value = var.get()
            if key == "FONT_SIZE_BASE":
                # Convert label to FontSize enum
                label = value
                font_enum = self.label_to_enum.get(label, FontSize.SMALL)
                Settings.FONT_SIZE = font_enum
                continue
            
            if hasattr(Settings, key):
                setattr(Settings, key, value)

        # Save LaTeX Title
        Settings.LATEX_TITLE = self.title_var.get()

        # Save LaTeX Template
        if hasattr(self, 'template_var'):
            Settings.LATEX_TEMPLATE = self.template_var.get()

        # Save Authors
        authors = []
        for _, entries in self.author_frames:
            author_data = {}
            for field, entry in entries.items():
                author_data[field] = entry.get()
            authors.append(author_data)
        
        Settings.LATEX_AUTHORS = authors

        # Persist to settings.py
        Settings.save_to_file()

        # Navigate back to Start screen
        from .start_screen import StartScreen
        self.controller.show_frame(StartScreen)

