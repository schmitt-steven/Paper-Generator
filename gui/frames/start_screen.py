import tkinter as tk
from tkinter import ttk, filedialog
import shutil
import subprocess
import platform
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass

from ..base_frame import BaseFrame, CardBorderFrame, InfoPopup
from ..info_texts import START_PAGE_INFO, USER_REQUIREMENTS_INFO, WRITING_GUIDELINES_INFO, CODE_FILES_INFO
from ..theme_colors import CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT, MUTED_TEXT
from ..icons import HoverColor


ALLOWED_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.cpp', '.c', '.h', '.go',
    '.rs', '.rb', '.cs', '.swift', '.kt',
    '.scala', '.r', '.jl'
}


@dataclass
class CodeFile:
    """Represents an uploaded code file."""
    filename: str
    path: str
    line_count: int


class StartScreen(BaseFrame):
    """Start page with quick access to Settings, User Requirements, Writing Guidelines, and Code Files."""
    
    def __init__(self, parent, controller):
        self.code_files: list[CodeFile] = []
        self.file_widgets: dict[str, ttk.Frame] = {}
        
        # File paths
        self.user_requirements_path = "user_files/user_requirements.md"
        self.writing_guidelines_path = "user_files/section_guidelines.md"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Start",
            has_next=True,
            next_text="Continue",
            has_back=False,
            info_content=START_PAGE_INFO
        )
    
    def create_content(self):
        """Create the four main sections."""
        self._create_settings_section()
        self._create_user_requirements_section()
        self._create_writing_guidelines_section()
        self._create_code_files_section()
    
    def on_show(self):
        """Load code files when screen is shown."""
        self._load_existing_files()

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
        
        # Content with Edit and Show in Explorer buttons (50/50)
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

    # ==================== User Requirements Section ====================
    
    def _create_user_requirements_section(self):
        """Create the User Requirements card."""
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)
        
        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        tk.Label(
            header, 
            text="User Requirements", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Info button in header
        info_btn = self.controller.icons.create_icon_label(
            header,
            icon_name="info",
            command=lambda: InfoPopup(self.controller, "User Requirements", USER_REQUIREMENTS_INFO),
            scale=1.5,
            hover_color=HoverColor.BLUE
        )
        info_btn.pack(side="right", padx=(5, 0))
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # Content with Edit and Show in Explorer buttons (50/50)
        content = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        content.pack(fill="x")
        content.columnconfigure(0, weight=1, uniform="buttons")
        content.columnconfigure(1, weight=1, uniform="buttons")
        
        ttk.Button(
            content, 
            text="Edit", 
            command=self._open_user_requirements
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            content, 
            text="Show in Explorer", 
            command=lambda: self._show_in_explorer(self.user_requirements_path)
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def _open_user_requirements(self):
        """Open the user requirements file directly in the default editor."""
        self._open_in_editor(self.user_requirements_path)

    # ==================== Writing Guidelines Section ====================
    
    def _create_writing_guidelines_section(self):
        """Create the Writing Guidelines card."""
        card = CardBorderFrame(self.scrollable_frame, padx=1, pady=1)
        card.pack(fill="x", pady=10)
        
        # Header
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        tk.Label(
            header, 
            text="Writing Guidelines", 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Info button in header
        info_btn = self.controller.icons.create_icon_label(
            header,
            icon_name="info",
            command=lambda: InfoPopup(self.controller, "Writing Guidelines", WRITING_GUIDELINES_INFO),
            scale=1.5,
            hover_color=HoverColor.BLUE
        )
        info_btn.pack(side="right", padx=(5, 0))
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        # Content with Edit and Show in Explorer buttons (50/50)
        content = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        content.pack(fill="x")
        content.columnconfigure(0, weight=1, uniform="buttons")
        content.columnconfigure(1, weight=1, uniform="buttons")
        
        ttk.Button(
            content, 
            text="Edit", 
            command=self._open_writing_guidelines
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            content, 
            text="Show in Explorer", 
            command=lambda: self._show_in_explorer(self.writing_guidelines_path)
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def _open_writing_guidelines(self):
        """Open the writing guidelines file directly in the default editor."""
        self._open_in_editor(self.writing_guidelines_path)

    # ==================== Code Files Section ====================
    
    def _create_code_files_section(self):
        """Create the Code Files card with upload and file list."""
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
            text="Code Files", 
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
            command=lambda: InfoPopup(self.controller, "Code Files", CODE_FILES_INFO),
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
            text="No code files uploaded yet",
            font=self.controller.fonts.default_font,
            foreground="gray"
        ).pack(pady=20)
    
    def _create_file_entry(self, parent: ttk.Frame, code_file: CodeFile) -> ttk.Frame:
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
            text=code_file.filename,
            font=self.controller.fonts.default_font,
            style="CardRow.TLabel",
            cursor="hand2"
        )
        filename_label.pack(anchor="w")
        filename_label.bind("<Button-1>", lambda e, p=code_file.path: self._open_in_editor(p))
        
        # Line count
        line_text = f"{code_file.line_count:,} lines"
        ttk.Label(
            content_frame,
            text=line_text,
            font=self.controller.fonts.text_area_font,
            foreground="gray",
            style="CardRow.TLabel"
        ).pack(anchor="w", pady=(2, 0))
        
        # Remove button
        x_btn = self.controller.icons.create_icon_label(
            content_row,
            icon_name="x",
            command=lambda: self._remove_file(code_file.filename)
        )
        x_btn.pack(side="right", padx=(10, 0))
        
        return entry_frame
    
    def _load_existing_files(self):
        """Load existing code files from user_files/ directory."""
        # Skip if already loaded
        if self.code_files:
            return
        
        user_files_dir = Path("user_files")
        if not user_files_dir.exists():
            return
        
        # Find all code files in user_files/
        code_file_paths = []
        for file_path in user_files_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                code_file_paths.append(str(file_path))
        
        if code_file_paths:
            self._process_files(tuple(code_file_paths))
    
    def _on_upload_click(self):
        """Handle Upload button click."""
        ext_pattern = " ".join(f"*{ext}" for ext in sorted(ALLOWED_EXTENSIONS))
        
        file_paths = filedialog.askopenfilenames(
            title="Select Code Files",
            filetypes=[
                ("Code files", ext_pattern),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        self._process_files(file_paths)
    
    def _process_files(self, file_paths: tuple):
        """Process and add code files."""
        existing_filenames = {f.filename for f in self.code_files}
        user_files_dir = Path("user_files")
        user_files_dir.mkdir(exist_ok=True)
        
        for file_path in file_paths:
            src_path = Path(file_path)
            
            if src_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                print(f"[StartScreen] Skipping unsupported file type: {src_path.name}")
                continue
            
            if src_path.name in existing_filenames:
                print(f"[StartScreen] Skipping duplicate: {src_path.name}")
                continue
            
            dest_path = user_files_dir / src_path.name
            
            if src_path.parent.resolve() != user_files_dir.resolve():
                shutil.copy2(src_path, dest_path)
            
            try:
                with open(dest_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                line_count = 0
            
            code_file = CodeFile(
                filename=src_path.name,
                path=str(dest_path),
                line_count=line_count
            )
            
            self.code_files.append(code_file)
            existing_filenames.add(src_path.name)
            print(f"[StartScreen] Added: {src_path.name} ({line_count} lines)")
        
        self._refresh_files_list()
    
    def _refresh_files_list(self):
        """Refresh the files list display."""
        for widget in self.files_list.winfo_children():
            widget.destroy()
        self.file_widgets.clear()
        
        if not self.code_files:
            self._show_empty_state()
        else:
            for i, code_file in enumerate(self.code_files):
                if i > 0:
                    ttk.Separator(self.files_list, orient="horizontal").pack(fill="x", padx=5)
                entry = self._create_file_entry(self.files_list, code_file)
                self.file_widgets[code_file.filename] = entry
        
        self._update_count()
    
    def _update_count(self):
        """Update the file count label."""
        self.count_label.config(text=str(len(self.code_files)))
    
    def _remove_file(self, filename: str):
        """Remove a code file."""
        removed = next((f for f in self.code_files if f.filename == filename), None)
        if removed:
            print(f"[StartScreen] Removed: {removed.filename}")
            
            file_path = Path(removed.path)
            if file_path.exists():
                file_path.unlink()
        
        self.code_files = [f for f in self.code_files if f.filename != filename]
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
