import tkinter as tk
from tkinter import ttk, filedialog
import shutil
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from ..base_frame import BaseFrame, create_gray_button


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


class CodeFilesScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.code_files: List[CodeFile] = []
        self.file_widgets: Dict[str, ttk.Frame] = {}
        
        # UI elements (initialized in create_content)
        self.upload_btn: ttk.Button
        self.count_label: ttk.Label
        self.files_list: ttk.Frame
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Code Files",
            next_text="Generate Paper Concept"
        )

    def create_content(self):
        # Explanation text
        self._create_info_section()
        
        # Code files section
        self._create_files_section()

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame, padding=(0, 0, 0, 0))
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Optionally upload your code relevant to the paper you want to write about here.\n"
            "Your code will be analyzed to better understand the topic of the paper.\n"
            "Additionally, the code will be used as the basis for writing experiments.\n"
            "Adding comments at critical points in the code could help the model understand the code better."
        )

        label = ttk.Label(
            explanation_frame,
            text=explanation_text,
            font=("SF Pro", 14),
            foreground="gray",
            justify="left"
        )
        label.pack(anchor="w", fill="x")

        def set_wraplength(event):
            label.config(wraplength=event.width-10)
        label.bind("<Configure>", set_wraplength)

    def _create_files_section(self):
        """Create the code files upload section."""
        section_frame = ttk.Frame(
            self.scrollable_frame,
            style="Card.TFrame",
            padding=1
        )
        section_frame.pack(fill="x", pady=10)
        
        inner_frame = ttk.Frame(section_frame, padding="10")
        inner_frame.pack(fill="x", expand=True)
        
        # Header row
        header_frame = ttk.Frame(section_frame, padding=10)
        header_frame.pack(fill="x")
        
        left_header = ttk.Frame(header_frame)
        left_header.pack(side="left")
        
        ttk.Label(left_header, text="Your Code Files", font=("SF Pro", 14, "bold")).pack(side="left")
        self.count_label = ttk.Label(left_header, text="0", font=("SF Pro", 14), foreground="gray")
        self.count_label.pack(side="left", padx=(10, 0))
        
        self.upload_btn = ttk.Button(header_frame, text="Upload", command=self._on_upload_click)
        self.upload_btn.pack(side="right")
        
        # Separator
        ttk.Separator(section_frame, orient="horizontal").pack(fill="x")
        
        # Files list container
        self.files_list = ttk.Frame(section_frame, padding=10)
        self.files_list.pack(fill="x")
        
        # Show empty state initially
        self._show_empty_state()

    def _show_empty_state(self):
        """Show empty state message."""
        ttk.Label(
            self.files_list,
            text="No code files uploaded yet",
            font=("SF Pro", 14),
            foreground="gray"
        ).pack(pady=20)

    def _create_file_entry(self, parent: ttk.Frame, code_file: CodeFile) -> ttk.Frame:
        """Create a single file entry widget."""
        entry_frame = ttk.Frame(parent, padding="8")
        entry_frame.pack(fill="x")
        
        content_row = ttk.Frame(entry_frame)
        content_row.pack(fill="x")
        
        content_frame = ttk.Frame(content_row)
        content_frame.pack(side="left", fill="x", expand=True)
        
        # Filename
        ttk.Label(
            content_frame,
            text=code_file.filename,
            font=("SF Pro", 14, "bold")
        ).pack(anchor="w")
        
        # Line count
        line_text = f"{code_file.line_count:,} lines"
        ttk.Label(
            content_frame,
            text=line_text,
            font=("SF Pro", 12),
            foreground="gray"
        ).pack(anchor="w", pady=(2, 0))
        
        # Trash button
        trash_btn = create_gray_button(
            content_row,
            text="\U0001F5D1",
            command=lambda: self._remove_file(code_file.filename),
            width=3
        )
        trash_btn.pack(side="right", padx=(10, 0))
        
        return entry_frame

    def _on_upload_click(self):
        """Handle Upload button click."""
        # Build file type filter from allowed extensions
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
            
            # Check extension is allowed
            if src_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                print(f"[CodeFiles] Skipping unsupported file type: {src_path.name}")
                continue
            
            # Skip duplicates
            if src_path.name in existing_filenames:
                print(f"[CodeFiles] Skipping duplicate: {src_path.name}")
                continue
            
            # Copy to user_files/
            dest_path = user_files_dir / src_path.name
            shutil.copy2(src_path, dest_path)
            
            # Count lines
            try:
                with open(dest_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                line_count = 0
            
            # Create CodeFile entry
            code_file = CodeFile(
                filename=src_path.name,
                path=str(dest_path),
                line_count=line_count
            )
            
            self.code_files.append(code_file)
            existing_filenames.add(src_path.name)
            print(f"[CodeFiles] Added: {src_path.name} ({line_count} lines)")
        
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
            print(f"[CodeFiles] Removed: {removed.filename}")
            
            # Delete the file from user_files/
            file_path = Path(removed.path)
            if file_path.exists():
                file_path.unlink()
        
        self.code_files = [f for f in self.code_files if f.filename != filename]
        self._refresh_files_list()

