import webbrowser
import platform
import subprocess
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pymupdf  # fitz

from ..base_frame import BaseFrame, ProgressPopup
from ..info_texts import RESULT_INFO
from phases.latex_generation.paper_converter import PaperConverter, LaTeXMetadata
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.paper_search.literature_search import LiteratureSearch
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.context_analysis.paper_conception import PaperConception
from phases.context_analysis.user_requirements import UserRequirements

PDF_PATH = "output/latex/result/paper.pdf"
PAPER_DRAFT_FILE = "output/paper_draft.md"
HYPOTHESES_FILE = "output/hypothesis.md"

MAX_PREVIEW_PAGES = 1

class ResultScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(
            parent=parent,
            controller=controller,
            title="Result",
            has_next=True,  # "Compile TeX Only" (Continue)
            next_text="Compile current TeX",
            has_regenerate=True, # "Update & Compile" (Regenerate)
            regenerate_text="Rebuild TeX Project",
            info_content=RESULT_INFO
        )
        self.preview_images = [] # Keep references to prevent GC

    def create_content(self):
        # Buttons Section
        btn_frame = ttk.Frame(self.scrollable_frame, style="Scrollable.TFrame")
        btn_frame.pack(fill="x", pady=10)
        
        # Grid connection for centering
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        
        # View Paper
        view_btn = ttk.Button(
            btn_frame,
            text="View Paper",
            command=self._open_pdf,
            state="normal"
        )
        view_btn.grid(row=0, column=0, padx=5, sticky="ew")
        
        # Show File
        show_btn = ttk.Button(
            btn_frame,
            text="Show in Explorer",
            command=self._show_file,
            state="normal"
        )
        show_btn.grid(row=0, column=1, padx=5, sticky="ew")
        
        # Preview Section
        self.preview_container = self.create_card_frame(self.scrollable_frame, "Preview")
        # Preview will be loaded in on_show

    def on_show(self):
        """Called when the screen is shown."""
        self.show_preview()

    def show_preview(self):
        """Render PDF pages as images in the preview container."""
        # Clear existing
        for widget in self.preview_container.winfo_children():
            widget.destroy()
        self.preview_images = []

        path = Path(PDF_PATH)
        if not path.exists():
            self.show_error_message("PDF Not Found", f"PDF not found at {path}.\nPlease compile the project.")
            return

        try:
            print(f"Opening PDF for preview: {path.absolute()}")
            doc = pymupdf.open(str(path))
            
            if len(doc) == 0:
                print("PDF is empty")
                self.show_error_message("Empty PDF", "The generated PDF file is empty.")
                return

            # Show pages
            for page_num in range(min(len(doc), MAX_PREVIEW_PAGES)):
                try:
                    page = doc.load_page(page_num)
                    # Force alpha=False to get RGB with white background (standard for papers)
                    pix = page.get_pixmap(dpi=150, alpha=False) 
                    
                    if not pix.samples:
                         print(f"No pixel samples for page {page_num}")
                         continue

                    # Determine mode based on channels
                    mode = "RGB" if pix.n == 3 else "RGBA"
                    if pix.n == 4:
                        # Should not happen with alpha=False but safety first
                        pix = page.get_pixmap(dpi=150, alpha=False)
                        mode = "RGB"

                    # Convert to PIL Image
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    
                    # Resize if needed to fit width (assuming standard letter/A4 aspect)
                    target_width = 600 # approximate max width to fit in card
                    if img.width > target_width:
                        ratio = target_width / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                    
                    tk_img = ImageTk.PhotoImage(img)
                    self.preview_images.append(tk_img)
                    
                    # Container for page
                    page_frame = ttk.Frame(self.preview_container, style="CardRow.TFrame", padding=10)
                    page_frame.pack(fill="x")
                    
                    # Page Image
                    lbl = ttk.Label(page_frame, image=tk_img, style="CardRow.TLabel")
                    lbl.pack()
                    
                    # Separator between pages (only if showing multiple preview pages)
                    pages_to_show = min(len(doc), MAX_PREVIEW_PAGES)
                    if page_num < pages_to_show - 1:
                        ttk.Separator(self.preview_container, orient="horizontal").pack(fill="x", padx=50, pady=10)
                except Exception as e_page:
                    print(f"Error processing page {page_num}: {e_page}")
                    ttk.Label(self.preview_container, text=f"Error processing page {page_num}.", foreground="red", style="CardRow.TLabel").pack()

            doc.close()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating preview: {e}")
            self.show_error_message("Preview Error", f"Error generating preview: {e}")


    def _open_pdf(self):
        """Open the generated PDF in the default browser/viewer."""
        path = Path(PDF_PATH)
        if path.exists():
            try:
                webbrowser.open(f"file://{path.absolute()}")
            except Exception as e:
                print(f"Error opening PDF: {e}")
                self.show_error_message("PDF Error", f"Could not open PDF: {e}")
        else:
            self.show_error_message("PDF Not Found", f"PDF not found at {path}")

    def _show_file(self):
        """Show the PDF file in the system file explorer."""
        path = Path(PDF_PATH).absolute()
        if not path.exists():
             self.show_error_message("File Not Found", f"File not found: {path}")
             return

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", "/select,", str(path)])
            elif platform.system() == "Darwin":
                subprocess.run(["open", "-R", str(path)])
            else:
                # Linux
                subprocess.run(["xdg-open", str(path.parent)])
        except Exception as e:
             self.show_error_message("System Error", f"Error showing file: {e}")

    def on_next(self):
        """Next button now triggers Compile TeX Only."""
        self._on_compile_tex_only()

    def on_regenerate(self):
        """Regenerate button now triggers Update & Compile."""
        self._on_update_and_compile()

    def _on_compile_tex_only(self):
        """Compile the existing LaTeX project without regenerating from Markdown."""
        if not tk.messagebox.askyesno(
            "Confirm Compilation", 
            "This will compile the current LaTeX files in 'output/latex'.\n\nDo you want to continue?"
        ):
            return

        popup = ProgressPopup(self.controller, "Compiling LaTeX")
        
        def task():
            try:
                converter = PaperConverter()
                latex_dir = Path("output/latex")
                
                if not latex_dir.exists():
                     self.after(0, lambda: popup.show_error("LaTeX output directory not found."))
                     return

                # Compile LaTeX
                self.after(0, lambda: popup.update_status("Compiling LaTeX to PDF"))
                success = converter.compile_latex(latex_dir)
                
                if success:
                    self.after(0, lambda: self._on_recompile_success(popup))
                else:
                    self.after(0, lambda: popup.show_error("LaTeX compilation failed. Check logs."))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_update_and_compile(self):
        """Reload all data, regenerate LaTeX from Markdown draft, then compile."""
        if not tk.messagebox.askyesno(
            "Confirm Update & Compile", 
            "This will overwrite any manual changes made to the LaTeX files (e.g. paper.tex) with the current content of the Markdown draft.\n\nDo you want to continue?"
        ):
            return

        popup = ProgressPopup(self.controller, "Updating & Compiling")
        
        def task():
            try:
                # Load paper draft
                self.after(0, lambda: popup.update_status("Loading paper draft"))
                # Note: PaperWritingPipeline.load_paper_draft is a static method
                paper_draft = PaperWritingPipeline.load_paper_draft(PAPER_DRAFT_FILE)
                
                # Load indexed papers
                self.after(0, lambda: popup.update_status("Loading indexed papers"))
                indexed_papers = LiteratureSearch.load_papers("output/papers.json")
                
                # Load experiment result (optional)
                experiment_result = None
                experiment_result_file = "output/experiments/experiment_result.json"
                if Path(experiment_result_file).exists():
                    self.after(0, lambda: popup.update_status("Loading experiment results"))
                    experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
                
                # Create metadata
                self.after(0, lambda: popup.update_status("Generating LaTeX project"))
                metadata = LaTeXMetadata.from_settings(generated_title=paper_draft.title)
                
                # Convert to LaTeX
                converter = PaperConverter()
                latex_dir = converter.convert_to_latex(
                    paper_draft=paper_draft,
                    metadata=metadata,
                    indexed_papers=indexed_papers,
                    experiment_result=experiment_result,
                    progress_callback=lambda msg: self.after(0, lambda: popup.update_status(msg))
                )
                
                # Compile LaTeX
                self.after(0, lambda: popup.update_status("Compiling LaTeX to PDF"))
                success = converter.compile_latex(latex_dir)
                
                if success:
                    self.after(0, lambda: self._on_recompile_success(popup))
                else:
                    self.after(0, lambda: popup.show_error("LaTeX compilation failed. Check logs."))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_recompile_success(self, popup: ProgressPopup):
        popup.close()
        # Refresh preview
        self.show_preview()
        messagebox.showinfo("Success", "PDF compiled successfully!")


