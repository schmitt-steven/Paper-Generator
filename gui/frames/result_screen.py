import webbrowser
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame, ProgressPopup
from phases.latex_generation.paper_converter import PaperConverter

PDF_PATH = "output/latex/result/paper.pdf"

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

    def create_content(self):
        # Open PDF Button
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(fill="x", pady=20)
        
        pdf_btn = ttk.Button(
            btn_frame,
            text="View generated paper",
            command=self._open_pdf,
            state="normal"
        )
        pdf_btn.pack(pady=10, ipadx=20)

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
        # Optionally refresh or just show success? 
        # Since we just recompiled, maybe just stay here. 
        # But UI doesn't need to change much unless we want to show a toast.
        pass

    def _show_error(self, msg):
        err_label = ttk.Label(self.scrollable_frame, text=msg, foreground="red")
        err_label.pack(pady=5)
