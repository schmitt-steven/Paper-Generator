import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import shutil
import subprocess
import platform
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass

from ..base_frame import BaseFrame, CardBorderFrame, InfoPopup, ProgressPopup
from ..info_texts import START_PAGE_INFO, PAPER_SPECIFICATION_INFO, STYLE_GUIDELINES_INFO, CODE_DATASETS_INFO
from ..theme_colors import CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT, MUTED_TEXT, SECONDARY_TEXT_DARK, SECONDARY_TEXT_LIGHT, LINK_COLOR_DARK, LINK_COLOR_LIGHT
from ..icons import HoverColor
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from settings import Settings
import threading


CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.cpp', '.c', '.h', '.go',
    '.rs', '.rb', '.cs', '.swift', '.kt',
    '.scala', '.r', '.jl'
}

DATASET_EXTENSIONS = {
    '.csv', '.json',
    # '.tsv', '.jsonl',
    # '.xlsx', '.xls', '.parquet'
}

ALL_EXTENSIONS = CODE_EXTENSIONS | DATASET_EXTENSIONS


@dataclass
class UserFile:
    """Represents an uploaded user file (code or dataset)."""
    filename: str
    path: str
    file_type: str  # "code" or "dataset"
    line_count: int = 0  # for code files
    file_size: int = 0   # for dataset files (bytes)

    @property
    def file_size_display(self) -> str:
        """Human-readable file size."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 * 1024):.1f} MB"


class StartScreen(BaseFrame):
    """Start page with quick access to Settings, Paper Specification, Style Guidelines, and Code Files."""

    def __init__(self, parent, controller):
        self.user_files: list[UserFile] = []
        self.file_widgets: dict[str, ttk.Frame] = {}
        self.experiment_file: str | None = Settings.USER_EXPERIMENT_FILE or None
        
        # File paths
        self.paper_specification_path = "user_files/paper_specification.md"
        self.style_guidelines_path = "user_files/style_guidelines.md"
        
        # Check if research context exists to determine button text
        research_context_path = Path("output/research_context.md")
        next_text = "Continue" if research_context_path.exists() else "Generate Research Context"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Start",
            has_next=True,
            next_text=next_text,
            has_back=False,
            info_content=START_PAGE_INFO
        )
    
    def create_content(self):
        """Create the four main sections."""
        self._create_settings_section()
        self._create_paper_specification_section()
        self._create_style_guidelines_section()
        self._create_code_files_section()
        
    def on_next(self):
        """Handle next button click. If context doesn't exist, generate it."""
        research_context_path = Path("output/research_context.md")
        
        if research_context_path.exists():
            super().on_next()
            return
            
        # Generate context if it doesn't exist
        popup = ProgressPopup(self.controller, "Generating Research Context")
        
        def task():
            try:
                # Use the centralized generation logic
                ResearchContextGenerator.generate_new_context(progress_callback=popup.update_status)
                
                self.after(0, lambda: self._on_generation_complete(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_generation_complete(self, popup: ProgressPopup):
        """Handle generation completion."""
        popup.close()
        # Proceed to next screen (ResearchContextScreen)
        super().on_next()
    
    def on_show(self):
        """Load code files when screen is shown and update next button text."""
        self._load_existing_files()
        
        # Update next button text based on context existence
        research_context_path = Path("output/research_context.md")
        next_text = "Continue" if research_context_path.exists() else "Generate Context"
        self.set_next_text(next_text)

    # ==================== Settings Section ====================
    
    def _create_settings_section(self):
        """Create the Settings card with Edit and Show in Explorer buttons."""
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)
        
        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        tk.Label(
            header, 
            text="Settings", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # Content with Edit and Show in Explorer buttons
        content = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        content.pack(fill="x")
        content.columnconfigure(0, weight=1, uniform="buttons")
        content.columnconfigure(1, weight=1, uniform="buttons")
        
        ttk.Button(
            content, 
            text="Edit", 
            command=self._open_settings
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            content, 
            text="Show in Explorer", 
            command=lambda: self._show_in_explorer("settings.py")
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def _open_settings(self):
        """Navigate to the full Settings screen."""
        from .settings_screen import SettingsScreen
        self.controller.show_frame(SettingsScreen)

    # ==================== Paper Specification Section ====================
    
    def _create_paper_specification_section(self):
        """Create the Paper Specification card."""
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)
        
        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        tk.Label(
            header, 
            text="Paper Specification", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Info button
        info_btn = self.controller.icons.create_icon_label(
            header,
            icon_name="info",
            command=lambda: InfoPopup(self.controller, "Paper Specification", PAPER_SPECIFICATION_INFO),
            scale=1.5,
            hover_color=HoverColor.BLUE
        )
        info_btn.pack(side="right", padx=(5, 0))
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # Content with Edit and Show in Explorer buttons
        content = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        content.pack(fill="x")
        content.columnconfigure(0, weight=1, uniform="buttons")
        content.columnconfigure(1, weight=1, uniform="buttons")
        
        ttk.Button(
            content, 
            text="Edit", 
            command=self._open_paper_specification
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            content, 
            text="Show in Explorer", 
            command=lambda: self._show_in_explorer(self.paper_specification_path)
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def _open_paper_specification(self):
        """Open the paper specification file directly in the default editor."""
        self._open_in_editor(self.paper_specification_path)

    # ==================== Style Guidelines Section ====================
    
    def _create_style_guidelines_section(self):
        """Create the Style Guidelines card."""
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)
        
        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        tk.Label(
            header, 
            text="Style Guidelines", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Info button in header
        info_btn = self.controller.icons.create_icon_label(
            header,
            icon_name="info",
            command=lambda: InfoPopup(self.controller, "Style Guidelines", STYLE_GUIDELINES_INFO),
            scale=1.5,
            hover_color=HoverColor.BLUE
        )
        info_btn.pack(side="right", padx=(5, 0))
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # Content with Edit and Show in Explorer buttons
        content = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        content.pack(fill="x")
        content.columnconfigure(0, weight=1, uniform="buttons")
        content.columnconfigure(1, weight=1, uniform="buttons")
        
        ttk.Button(
            content, 
            text="Edit", 
            command=self._open_style_guidelines
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            content, 
            text="Show in Explorer", 
            command=lambda: self._show_in_explorer(self.style_guidelines_path)
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def _open_style_guidelines(self):
        """Open the style guidelines file directly in the default editor."""
        self._open_in_editor(self.style_guidelines_path)

    # ==================== Code & Data Section ====================

    def _create_code_files_section(self):
        """Create the Code & Data card with upload and file list."""
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)

        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")

        left_header = ttk.Frame(header, style="CardHeader.TFrame")
        left_header.pack(side="left")

        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT

        tk.Label(
            left_header,
            text="Code & Data",
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")

        self.count_label = tk.Label(
            left_header,
            text="0",
            font=self.controller.fonts.sub_header_font,
            fg=MUTED_TEXT,
            bg=header_bg
        )
        self.count_label.pack(side="left", padx=(10, 0))

        # Right side: Upload, Info
        info_btn = self.controller.icons.create_icon_label(
            header,
            icon_name="info",
            command=lambda: InfoPopup(self.controller, "Code & Data", CODE_DATASETS_INFO),
            scale=1.5,
            hover_color=HoverColor.BLUE
        )
        info_btn.pack(side="right", padx=(5, 0))

        ttk.Button(
            header,
            text="Upload",
            command=self._on_upload_click
        ).pack(side="right", padx=(5, 5))

        ttk.Separator(card, orient="horizontal").pack(fill="x")

        # Files list container
        self.files_list = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        self.files_list.pack(fill="x")

        # Show empty state initially
        self._show_empty_state()
    
    def _show_empty_state(self):
        """Show empty state message."""
        ttk.Label(
            self.files_list,
            text="No files uploaded yet",
            font=self.controller.fonts.default_font,
            foreground="gray"
        ).pack(pady=20)

    @staticmethod
    def _classify_file(filename: str) -> str:
        """Classify a file as 'code' or 'dataset' based on extension."""
        suffix = Path(filename).suffix.lower()
        if suffix in DATASET_EXTENSIONS:
            return "dataset"
        return "code"

    def _create_file_entry(self, parent: ttk.Frame, user_file: UserFile) -> ttk.Frame:
        """Create a single file entry widget."""
        entry_frame = ttk.Frame(parent, style="CardRow.TFrame", padding="8")
        entry_frame.pack(fill="x")

        content_row = ttk.Frame(entry_frame, style="CardRow.TFrame")
        content_row.pack(fill="x")

        content_frame = ttk.Frame(content_row, style="CardRow.TFrame")
        content_frame.pack(side="left", fill="x", expand=True)

        # Filename (clickable)
        filename_label = ttk.Label(
            content_frame,
            text=user_file.filename,
            font=self.controller.fonts.default_font,
            style="CardRow.TLabel",
            cursor="hand2"
        )
        filename_label.pack(anchor="w")

        # Info row: metric + tag
        info_frame = ttk.Frame(content_frame, style="CardRow.TFrame")
        info_frame.pack(anchor="w", pady=(2, 0))

        # Tag: "Experiment" (green), "Code" (blue), or "Dataset" (orange)
        is_experiment = (user_file.file_type == "code" and self.experiment_file == user_file.filename)
        if is_experiment:
            tag_text, tag_color = "Experiment", "#2ecc71"
        elif user_file.file_type == "dataset":
            tag_text, tag_color = "Dataset", "#e67e22"
        else:
            tag_text, tag_color = "Code", "#3498db"

        ttk.Label(
            info_frame,
            text=tag_text,
            font=self.controller.fonts.text_area_font,
            foreground=tag_color,
            style="CardRow.TLabel"
        ).pack(side="left")

        filename_label.bind("<Button-1>", lambda e, p=user_file.path: self._open_in_editor(p))

        # Separator
        ttk.Label(info_frame, text="  \u00B7  ", font=self.controller.fonts.text_area_font, foreground="gray", style="CardRow.TLabel").pack(side="left")

        # Metric: line count for code, file size for datasets
        if user_file.file_type == "dataset":
            metric_text = user_file.file_size_display
        else:
            metric_text = f"{user_file.line_count:,} lines"

        ttk.Label(
            info_frame,
            text=metric_text,
            font=self.controller.fonts.text_area_font,
            foreground="gray",
            style="CardRow.TLabel"
        ).pack(side="left")
        
        # Remove button
        x_btn = self.controller.icons.create_icon_label(
            content_row,
            icon_name="x",
            command=lambda: self._remove_file(user_file.filename)
        )
        x_btn.pack(side="right", padx=(10, 0))

        # "Use as Experiment" / "Remove Experiment" link for .py code files
        if user_file.file_type == "code" and user_file.filename.endswith('.py'):
            is_dark = self.controller.current_theme == "dark"
            normal_color = SECONDARY_TEXT_DARK if is_dark else SECONDARY_TEXT_LIGHT
            hover_color = LINK_COLOR_DARK if is_dark else LINK_COLOR_LIGHT

            if is_experiment:
                exp_label = ttk.Label(
                    content_row,
                    text="Remove Experiment",
                    font=self.controller.fonts.default_font,
                    foreground=normal_color,
                    cursor="hand2",
                    style="CardRow.TLabel"
                )
                exp_label.bind("<Button-1>", lambda e: self._deactivate_experiment())
            else:
                exp_label = ttk.Label(
                    content_row,
                    text="Use as Experiment",
                    font=self.controller.fonts.default_font,
                    foreground=normal_color,
                    cursor="hand2",
                    style="CardRow.TLabel"
                )
                exp_label.bind("<Button-1>", lambda e, f=user_file: self._on_use_as_experiment_click(f))

            exp_label.bind("<Enter>", lambda e: exp_label.configure(foreground=hover_color))
            exp_label.bind("<Leave>", lambda e: exp_label.configure(foreground=normal_color))
            exp_label.pack(side="right", padx=(10, 0))

        return entry_frame

    @staticmethod
    def _migrate_flat_files():
        """Migrate files from flat user_files/ to subfolders (backward compatibility)."""
        user_files_dir = Path("user_files")
        if not user_files_dir.exists():
            return

        code_dir = user_files_dir / "code"
        datasets_dir = user_files_dir / "datasets"

        for file_path in list(user_files_dir.iterdir()):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix in CODE_EXTENSIONS:
                code_dir.mkdir(exist_ok=True)
                dest = code_dir / file_path.name
                if not dest.exists():
                    shutil.move(str(file_path), str(dest))
                    print(f"[Migration] Moved {file_path.name} -> code/")
            elif suffix in DATASET_EXTENSIONS:
                datasets_dir.mkdir(exist_ok=True)
                dest = datasets_dir / file_path.name
                if not dest.exists():
                    shutil.move(str(file_path), str(dest))
                    print(f"[Migration] Moved {file_path.name} -> datasets/")

    def _load_existing_files(self):
        """Load existing files from user_files/code/ and user_files/datasets/ directories."""
        # Skip if already loaded
        if self.user_files:
            return

        # Migrate flat structure if needed
        self._migrate_flat_files()

        user_files_dir = Path("user_files")
        if not user_files_dir.exists():
            return

        file_paths = []

        # Load code files
        code_dir = user_files_dir / "code"
        if code_dir.exists():
            for file_path in code_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in CODE_EXTENSIONS:
                    file_paths.append(str(file_path))

        # Load dataset files
        datasets_dir = user_files_dir / "datasets"
        if datasets_dir.exists():
            for file_path in datasets_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in DATASET_EXTENSIONS:
                    file_paths.append(str(file_path))

        if file_paths:
            self._process_files(tuple(file_paths))

    def _on_upload_click(self):
        """Handle Upload button click."""
        code_pattern = " ".join(f"*{ext}" for ext in sorted(CODE_EXTENSIONS))
        dataset_pattern = " ".join(f"*{ext}" for ext in sorted(DATASET_EXTENSIONS))
        all_pattern = " ".join(f"*{ext}" for ext in sorted(ALL_EXTENSIONS))

        file_paths = filedialog.askopenfilenames(
            title="Select Code or Data Files",
            filetypes=[
                ("All supported files", all_pattern),
                ("Code files", code_pattern),
                ("Dataset files", dataset_pattern),
                ("All files", "*.*")
            ]
        )

        if not file_paths:
            return

        self._process_files(file_paths)

    def _process_files(self, file_paths: tuple):
        """Process and add code/dataset files to appropriate subfolders."""
        existing_filenames = {f.filename for f in self.user_files}
        user_files_dir = Path("user_files")
        code_dir = user_files_dir / "code"
        datasets_dir = user_files_dir / "datasets"

        for file_path in file_paths:
            src_path = Path(file_path)
            suffix = src_path.suffix.lower()

            if suffix not in ALL_EXTENSIONS:
                print(f"[StartScreen] Skipping unsupported file type: {src_path.name}")
                continue

            if src_path.name in existing_filenames:
                print(f"[StartScreen] Skipping duplicate: {src_path.name}")
                continue

            # Determine type and destination subfolder
            file_type = self._classify_file(src_path.name)
            if file_type == "dataset":
                dest_dir = datasets_dir
            else:
                dest_dir = code_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name

            # Copy if not already in the target directory
            if src_path.parent.resolve() != dest_dir.resolve():
                shutil.copy2(src_path, dest_path)

            # Compute metrics
            line_count = 0
            file_size = 0
            if file_type == "dataset":
                file_size = dest_path.stat().st_size
            else:
                try:
                    with open(dest_path, 'r', encoding='utf-8', errors='ignore') as f:
                        line_count = sum(1 for _ in f)
                except Exception:
                    line_count = 0

            user_file = UserFile(
                filename=src_path.name,
                path=str(dest_path),
                file_type=file_type,
                line_count=line_count,
                file_size=file_size
            )

            self.user_files.append(user_file)
            existing_filenames.add(src_path.name)

            if file_type == "dataset":
                print(f"[StartScreen] Added dataset: {src_path.name} ({user_file.file_size_display})")
            else:
                print(f"[StartScreen] Added code: {src_path.name} ({line_count} lines)")

        self._refresh_files_list()

    def _refresh_files_list(self):
        """Refresh the files list display."""
        for widget in self.files_list.winfo_children():
            widget.destroy()
        self.file_widgets.clear()

        if not self.user_files:
            self._show_empty_state()
        else:
            for i, user_file in enumerate(self.user_files):
                if i > 0:
                    ttk.Separator(self.files_list, orient="horizontal").pack(fill="x", padx=5)
                entry = self._create_file_entry(self.files_list, user_file)
                self.file_widgets[user_file.filename] = entry

        self._update_count()

    def _update_count(self):
        """Update the file count label."""
        self.count_label.config(text=str(len(self.user_files)))

    def _remove_file(self, filename: str):
        """Remove a user file."""
        removed = next((f for f in self.user_files if f.filename == filename), None)
        if removed:
            print(f"[StartScreen] Removed: {removed.filename}")

            file_path = Path(removed.path)
            if file_path.exists():
                file_path.unlink()

        # If the removed file was the experiment, deactivate it
        if self.experiment_file == filename:
            self._deactivate_experiment()

        self.user_files = [f for f in self.user_files if f.filename != filename]
        self._refresh_files_list()

    # ==================== User Experiment Methods ====================

    def _on_use_as_experiment_click(self, code_file: UserFile):
        """Show confirmation popup before activating user experiment."""
        confirmed = messagebox.askyesno(
            "Use as Experiment",
            f"Use '{code_file.filename}' as the experiment?\n\n"
            "This will:\n"
            "- Skip experiment plan generation\n"
            "- Skip experiment code generation\n"
            "- Run your file directly as the experiment\n\n"
            "Requirements:\n"
            "- The script must be runnable with Python\n"
            "- Save plots to a 'plots/' subdirectory using PDF format\n"
            "- Save results to 'results.json' in the working directory\n"
            "- Use matplotlib with Agg backend (no GUI windows)\n"
            "- Complete within the timeout (default 10 minutes)\n\n"
            "The working directory will be 'output/experiments/'.\n"
            "Your other uploaded code files will be copied there for imports.\n\n"
            "Do you want to continue?"
        )

        if not confirmed:
            return

        # Run LLM check in background
        self._run_experiment_code_check(code_file)

    def _run_experiment_code_check(self, code_file: UserFile):
        """Run an LLM check on the user's code and show results before activation."""
        popup = ProgressPopup(self.controller, "Checking experiment code")

        def task():
            try:
                issues = self._llm_check_experiment_code(code_file)
                self.after(0, lambda: self._on_code_check_complete(popup, code_file, issues))
            except Exception as e:
                import traceback
                traceback.print_exc()
                # If LLM check fails, proceed anyway
                self.after(0, lambda: self._on_code_check_complete(popup, code_file, None))

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _llm_check_experiment_code(self, code_file: UserFile) -> str | None:
        """Use LLM to check if the code is suitable as an experiment. Returns issues string or None."""
        import lmstudio as lms
        from utils.llm_utils import remove_thinking_blocks

        try:
            with open(code_file.path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception:
            return None

        system_prompt = (
            "You check Python scripts for compatibility as automated scientific experiments.\n"
            "The script will be executed headlessly via subprocess with cwd='output/experiments/'.\n\n"
            "Check ONLY for these aspects:\n"
            "1. Uses plt.show() or other GUI/display calls (should use Agg backend, no windows)\n"
            "2. Saves files to absolute paths instead of relative paths\n"
            "3. Missing 'plots/' directory usage or not saving figures as PDF format\n"
            "4. Missing 'results.json' for saving metrics\n"
            "5. Has interactive input (input(), sys.stdin, etc.)\n"
            "6. Imports that are clearly unavailable in a standard Python environment (assume local imports from the working directory are available)\n\n"
            "Important Rules:\n"
            "- Do NOT tell the user to create the 'plots/' directory. It is already created for them.\n\n"
            "Respond with EXACTLY one of:\n"
            "- 'OK' if nothing needs to be changed\n"
            "- A SHORT bullet list of suggested improvements (max 5 lines). "
            "Frame each point as an actionable suggestion, e.g. "
            "'Add matplotlib.use(\"Agg\") before importing pyplot' instead of 'Uses plt.show()'"
        )

        try:
            chat = lms.Chat(system_prompt)
            chat.add_user_message(f"```python\n{code_content}\n```")
            model = lms.llm(Settings.EXPERIMENT_VALIDATION_MODEL)
            result = model.respond(chat, config={"temperature": 0.0, "timeout": 60})
            response = remove_thinking_blocks(result.content).strip()
            if response.upper() == "OK":
                return None
            return response
        except Exception as e:
            print(f"[StartScreen] LLM code check failed: {e}")
            return None

    def _on_code_check_complete(self, popup: ProgressPopup, code_file: UserFile, issues: str | None):
        """Handle LLM code check completion."""
        popup.close()

        if issues:
            # Show suggestions and let user decide
            proceed = messagebox.askyesno(
                "Code Check Results",
                f"Suggested improvements for '{code_file.filename}':\n\n"
                f"{issues}\n\n"
                "You can apply these changes before running the experiment.\n"
                "Do you still want to use this file as the experiment?"
            )
            if not proceed:
                return

        self._activate_experiment(code_file)

    def _activate_experiment(self, code_file: UserFile):
        """Set the given file as the user experiment."""
        self.experiment_file = code_file.filename
        Settings.USER_EXPERIMENT_FILE = code_file.filename
        Settings.save_to_file()
        print(f"[StartScreen] Activated user experiment: {code_file.filename}")
        self._refresh_files_list()

    def _deactivate_experiment(self):
        """Remove the user experiment selection."""
        self.experiment_file = None
        Settings.USER_EXPERIMENT_FILE = ""
        Settings.save_to_file()
        print("[StartScreen] Deactivated user experiment")
        self._refresh_files_list()

    # ==================== Utility Methods ====================
    
    def _open_in_editor(self, file_path: str):
        """Open a file in the default editor."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        print(f"Opening {file_path} in editor...")
        path = os.path.abspath(file_path)
        
        if platform.system() == 'Windows':
            os.startfile(path)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', path))
        else:
            subprocess.call(('xdg-open', path))
    
    def _show_in_explorer(self, file_path: str):
        """Reveal a file in the file explorer."""
        if not file_path:
            return
        
        print(f"Showing {file_path} in explorer...")
        path = os.path.abspath(file_path)
        path = os.path.normpath(path)
        
        if platform.system() == 'Windows':
            subprocess.Popen(f'explorer /select,"{path}"')
        elif platform.system() == 'Darwin':
            subprocess.call(['open', '-R', path])
        else:
            subprocess.call(['xdg-open', os.path.dirname(path)])
