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
from phases.document_generation.paper_converter import PaperConverter, LaTeXMetadata
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.literature_search.literature_search import LiteratureSearch
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from phases.context_analysis.paper_specification import PaperSpecification

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


        # Bind resize event to adjust card height
        self._canvas.bind("<Configure>", self._update_card_height, add="+")

    def _update_card_height(self, event=None):
        """Dynamically adjust card height to fill available space."""
        if not self.preview_scroll_canvas:
            return
            
        # Get canvas height
        canvas_height = self._canvas.winfo_height()
        
        # Determine target height (canvas height - margins - card header)
        # Margins: BaseFrame padding ~20, Card margin ~20, Card header ~40, plus safe zone
        target_height = canvas_height - 160 
        
        # Minimum safe height
        if target_height < 400:
            target_height = 400
            
        # Update the preview canvas height
        self.preview_scroll_canvas.configure(height=target_height)

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
        view_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        # Show File
        show_btn = ttk.Button(
            btn_frame,
            text="Show in Explorer",
            command=self._show_file,
            state="normal"
        )
        show_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        # Preview Section

        self.preview_container = self.create_card_frame(self.scrollable_frame, "Preview")
        
        # Create internal scrolling mechanism for Preview
        # Use a canvas + scrollbar inside the card content frame
        self.preview_container.grid_rowconfigure(0, weight=1)
        self.preview_container.grid_columnconfigure(0, weight=1)

        self.preview_scroll_canvas = tk.Canvas(self.preview_container, highlightthickness=0)
        self.preview_scrollbar = ttk.Scrollbar(self.preview_container, orient="vertical", command=self.preview_scroll_canvas.yview)
        
        self.preview_scroll_canvas.configure(yscrollcommand=self.preview_scrollbar.set)

        self.preview_scroll_canvas.pack(side="left", fill="both", expand=True)
        self.preview_scrollbar.pack(side="right", fill="y")

        # Inner frame to hold the actual images
        style = ttk.Style()
        bg_color = style.lookup("CardContent.TFrame", "background")
        if not bg_color: # Fallback
             bg_color = self.preview_container.cget("background")

        self.preview_scroll_canvas.configure(bg=bg_color)
        
        # Use a standard Frame for the inner content to attach to window
        self.preview_inner_frame = ttk.Frame(self.preview_scroll_canvas, style="CardContent.TFrame")
        self.preview_window_id = self.preview_scroll_canvas.create_window((0, 0), window=self.preview_inner_frame, anchor="nw")

        # Bindings for scrolling
        self.preview_inner_frame.bind("<Configure>", self._update_preview_scrollregion)
        self.preview_scroll_canvas.bind("<Configure>", self._update_preview_window_width)
        

        # Force initial height calculation
        self.after(100, self._update_card_height)

    def on_show(self):
        """Called when the screen is shown."""
        self.show_preview()

    def _update_preview_scrollregion(self, event=None):
        self.preview_scroll_canvas.configure(scrollregion=self.preview_scroll_canvas.bbox("all"))

    def _update_preview_window_width(self, event):
        self.preview_scroll_canvas.itemconfig(self.preview_window_id, width=event.width)

    def _is_mouse_in_widget(self, widget_to_check):
        try:
            x, y = self.winfo_pointerx(), self.winfo_pointery()
            widget = self.winfo_containing(x, y)
            while widget:
                if widget is widget_to_check:
                    return True
                widget = widget.master
        except:
            pass
        return False

    def _on_mousewheel(self, event):
        """Override global mousewheel to check for preview scrolling first."""
        if self._is_mouse_in_widget(self.preview_container):
            self._on_preview_mousewheel(event)
        else:
            super()._on_mousewheel(event)

    def _on_preview_mousewheel(self, event):
        if self.preview_inner_frame.winfo_reqheight() <= self.preview_scroll_canvas.winfo_height():
            return
            
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta > 0:
            delta = -1
        else:
            delta = 1
        self.preview_scroll_canvas.yview_scroll(delta, "units")

    def show_preview(self):
        """Render PDF pages as images in the preview container."""
        # Guard against destroyed widget (e.g. after reload_content rebuilt the frame)
        if not self.preview_inner_frame.winfo_exists():
            return

        # Clear existing
        for widget in self.preview_inner_frame.winfo_children():
            widget.destroy()
        self.preview_images = []

        path = Path(PDF_PATH)
        if not path.exists():
            ttk.Label(self.preview_inner_frame, text=f"PDF not found at {path}.\nPlease compile the project.", foreground="red", justify="center").pack(pady=40)
            return

        try:
            print(f"Opening PDF for preview: {path.absolute()}")
            doc = pymupdf.open(str(path))
            
            if len(doc) == 0:
                print("PDF is empty")
                ttk.Label(self.preview_inner_frame, text="The generated PDF file is empty.", foreground="red").pack(pady=40)
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
                    page_frame = ttk.Frame(self.preview_inner_frame, style="CardRow.TFrame", padding=10)
                    page_frame.pack(fill="x")
                    
                    # Page Image
                    lbl = ttk.Label(page_frame, image=tk_img, style="CardRow.TLabel")
                    lbl.pack()
                    
                    # Separator between pages (only if showing multiple preview pages)
                    pages_to_show = min(len(doc), MAX_PREVIEW_PAGES)
                    if page_num < pages_to_show - 1:
                        ttk.Separator(self.preview_inner_frame, orient="horizontal").pack(fill="x", padx=50, pady=10)
                except Exception as e_page:
                    print(f"Error processing page {page_num}: {e_page}")
                    ttk.Label(self.preview_inner_frame, text=f"Error processing page {page_num}.", foreground="red", style="CardRow.TLabel").pack()

            doc.close()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating preview: {e}")
            ttk.Label(self.preview_inner_frame, text=f"Error generating preview:\n{e}", foreground="red", justify="left").pack(pady=20)


    def _open_pdf(self):
        """Open the generated PDF in the default browser/viewer."""
        path = Path(PDF_PATH)
        if path.exists():
            try:
                webbrowser.open(f"file://{path.absolute()}")
            except Exception as e:
                print(f"Error opening PDF: {e}")
                messagebox.showerror("PDF Error", f"Could not open PDF:\n{e}")
        else:
            messagebox.showerror("PDF Not Found", f"PDF not found at {path}")

    def _show_file(self):
        """Show the PDF file in the system file explorer."""
        path = Path(PDF_PATH).absolute()
        if not path.exists():
             messagebox.showerror("File Not Found", f"File not found: {path}")
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
             messagebox.showerror("System Error", f"Error showing file:\n{e}")

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
        self.after(200, lambda: messagebox.showinfo("Success", "PDF compiled successfully!"))


