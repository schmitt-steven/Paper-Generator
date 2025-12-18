import webbrowser
import platform
import subprocess
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pymupdf  # fitz

from ..base_frame import BaseFrame, ProgressPopup
from phases.latex_generation.paper_converter import PaperConverter

PDF_PATH = "output/latex/result/paper.pdf"

MAX_PREVIEW_PAGES = 1

class ResultScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(
            parent=parent,
            controller=controller,
            title="Result",
            has_next=False,
            has_regenerate=True,
            regenerate_text="Recompile"
        )
        self.preview_images = [] # Keep references to prevent GC

    def create_content(self):
        # Buttons Section
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(fill="x", pady=20)
        
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
        view_btn.grid(row=0, column=0, padx=10, sticky="e")
        
        # Show File
        show_btn = ttk.Button(
            btn_frame,
            text="Show in Explorer",
            command=self._show_file,
            state="normal"
        )
        show_btn.grid(row=0, column=1, padx=10, sticky="w")
        
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
            ttk.Label(self.preview_container, text="PDF not found yet.").pack(pady=20)
            return

        try:
            print(f"Opening PDF for preview: {path.absolute()}")
            doc = pymupdf.open(str(path))
            
            if len(doc) == 0:
                print("PDF is empty")
                ttk.Label(self.preview_container, text="PDF is empty.", foreground="gray").pack(pady=20)
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

                    #print(f"Page {page_num}: {pix.width}x{pix.height}, channels={pix.n}")

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
                    page_frame = ttk.Frame(self.preview_container, padding=10)
                    page_frame.pack(fill="x")
                    
                    # Page Image
                    lbl = ttk.Label(page_frame, image=tk_img)
                    lbl.pack()
                    
                    # Separator between pages
                    if page_num < len(doc) - 1:
                        ttk.Separator(self.preview_container, orient="horizontal").pack(fill="x", padx=50, pady=10)
                except Exception as e_page:
                    print(f"Error processing page {page_num}: {e_page}")
                    ttk.Label(self.preview_container, text=f"Error processing page {page_num}.", foreground="red").pack()

            doc.close()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating preview: {e}")
            ttk.Label(self.preview_container, text=f"Preview error: {e}", foreground="red").pack(pady=20)

    def _open_pdf(self):
        """Open the generated PDF in the default browser/viewer."""
        path = Path(PDF_PATH)
        if path.exists():
            try:
                webbrowser.open(f"file://{path.absolute()}")
            except Exception as e:
                print(f"Error opening PDF: {e}")
                self._show_error(f"Could not open PDF: {e}")
        else:
            self._show_error(f"PDF not found at {path}")

    def _show_file(self):
        """Show the PDF file in the system file explorer."""
        path = Path(PDF_PATH).absolute()
        if not path.exists():
             self._show_error(f"File not found: {path}")
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
             self._show_error(f"Error showing file: {e}")

    def on_regenerate(self):
        """Recompile the LaTeX project."""
        popup = ProgressPopup(self.controller, "Recompiling LaTeX")
        
        def task():
            try:
                converter = PaperConverter()
                latex_dir = Path("output/latex")
                
                if not latex_dir.exists():
                     self.after(0, lambda: popup.show_error("LaTeX output directory not found."))
                     return

                # Compile LaTeX
                self.after(0, lambda: popup.update_status("Compiling LaTeX"))
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

    def _show_error(self, msg):
        err_label = ttk.Label(self.scrollable_frame, text=msg, foreground="red")
        err_label.pack(pady=5)
