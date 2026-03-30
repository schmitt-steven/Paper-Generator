import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import os
import subprocess
import platform
import threading

from ..base_frame import BaseFrame, ProgressPopup, CardBorderFrame
from ..info_texts import EXPERIMENT_RESULTS_INFO
from ..markdown_view import MarkdownView
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT,
)
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.context_analysis.research_context_generator import ResearchContextGenerator
from phases.context_analysis.paper_specification import PaperSpecification
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from settings import Settings


PAPER_DRAFT_FILE = Path("output/paper_draft.md")
HYPOTHESES_FILE = "output/hypothesis.md"


class CollapsibleTextCard(CardBorderFrame):
    """A collapsible card for text content (read-only)."""
    
    def __init__(self, parent, section_name: str, content: str, controller, start_expanded: bool = False, code_font: bool = False):
        super().__init__(parent, padx=1, pady=1)
        self.section_name = section_name
        self.content = content
        self.controller = controller
        self.expanded = False
        self.code_font = code_font
        
        self._build_ui()
        
        if start_expanded:
            self.expand()

        self.bind("<<ThemeChanged>>", self._on_theme_changed, add="+")

    def _on_theme_changed(self, event=None):
        """Update colors on theme change."""
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        # Update header widgets
        self.left_frame.config(bg=header_bg)
        self.toggle_label.config(bg=header_bg, fg=header_fg)
        self.title_label.config(bg=header_bg, fg=header_fg)
    
    def _build_ui(self):
        # Header
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        # Clickable header
        self.left_frame = tk.Frame(header, bg=header_bg)
        self.left_frame.pack(side="left", fill="x", expand=True)
        self.left_frame.bind("<Button-1>", lambda e: self.toggle())
        
        self.toggle_label = tk.Label(
            self.left_frame,
            text="▶",
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.toggle_label.pack(side="left", padx=(0, 10))
        self.toggle_label.bind("<Button-1>", lambda e: self.toggle())
        
        self.title_label = tk.Label(
            self.left_frame,
            text=self.section_name,
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", lambda e: self.toggle())
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame
        self.content_frame = ttk.Frame(self, style="CardContent.TFrame", padding=0)
        
        # Markdown View
        self.text_widget = MarkdownView(
            self.content_frame,
            font_manager=self.controller.fonts,
            theme_mode=self.controller.current_theme,
            padx=12,
            pady=10
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        self.text_widget.set_markdown(self.content)
        
        def adjust_height():
            # Estimate height based on content length
            # Base padding + per line height
            # Assumes approx 90 chars per line wrap width
            lines = self.content.split('\n')
            estimated_lines = 0.0
            
            for line in lines:
                length = len(line)
                if length == 0:
                    estimated_lines += 1
                else:
                    estimated_lines += (length / 90.0) + 1.0
            
            # 20px per line approx + 18px padding (reduced from 40 to avoid extra empty line)
            pixel_height = int(estimated_lines * 22) + 18
            
            # Clamp height (min 60, max 500)
            pixel_height = max(60, min(pixel_height, 500))
            
            self.text_widget.configure(height=pixel_height)
        
        # Determine height after a brief delay
        self.text_widget.after(100, adjust_height)

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_label.config(text="▼")
            self.content_frame.pack(fill="both", expand=True)
        else:
            self.toggle_label.config(text="▶")
            self.content_frame.pack_forget()

    def expand(self):
        if not self.expanded:
            self.toggle()


class CollapsibleCodeCard(CardBorderFrame):
    """A collapsible card for experiment code with Edit and Execute buttons."""
    
    def __init__(self, parent, section_name: str, content: str, controller, 
                 on_edit=None, on_show_explorer=None, on_execute=None, start_expanded: bool = False):
        super().__init__(parent, padx=1, pady=1)
        self.section_name = section_name
        self.content = content
        self.controller = controller
        self.on_edit = on_edit
        self.on_show_explorer = on_show_explorer
        self.on_execute = on_execute
        self.expanded = False
        
        self._build_ui()
        
        if start_expanded:
            self.expand()

        self.bind("<<ThemeChanged>>", self._on_theme_changed, add="+")

    def _on_theme_changed(self, event=None):
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        self.left_frame.config(bg=header_bg)
        self.toggle_label.config(bg=header_bg, fg=header_fg)
        self.title_label.config(bg=header_bg, fg=header_fg)
        self.btn_frame.config(bg=header_bg)
        
        # Update text widget
        # Note: self.text_widget might be disabled, so we might need to enable/disable or just config working?
        # tk.Text background can be changed even if disabled usually.
        text_bg = TEXT_BG_DARK_ALT if self.controller.current_theme == "dark" else TEXT_BG_LIGHT_ALT
        text_fg = TEXT_FG_DARK if self.controller.current_theme == "dark" else TEXT_FG_LIGHT
        self.text_widget.config(background=text_bg, foreground=text_fg)
    
    def _build_ui(self):
        # Header
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        # Clickable Left Header (Toggle + Title)
        self.left_frame = tk.Frame(header, bg=header_bg)
        self.left_frame.pack(side="left", fill="x", expand=True)
        self.left_frame.bind("<Button-1>", lambda e: self.toggle())
        
        self.toggle_label = tk.Label(
            self.left_frame,
            text="▶",
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.toggle_label.pack(side="left", padx=(0, 10))
        self.toggle_label.bind("<Button-1>", lambda e: self.toggle())
        
        self.title_label = tk.Label(
            self.left_frame,
            text=self.section_name,
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", lambda e: self.toggle())
        
        # Buttons on Right
        self.btn_frame = tk.Frame(header, bg=header_bg)
        self.btn_frame.pack(side="right")
        
        if self.on_execute:
            execute_btn = ttk.Button(self.btn_frame, text="Execute", command=self.on_execute)
            execute_btn.pack(side="left", padx=(0, 10))
        
        if self.on_edit:
            edit_btn = ttk.Button(self.btn_frame, text="Edit", command=self.on_edit)
            edit_btn.pack(side="left", padx=(0, 10))
        
        if self.on_show_explorer:
            explorer_btn = ttk.Button(self.btn_frame, text="Show in Explorer", command=self.on_show_explorer)
            explorer_btn.pack(side="left", padx=(0))
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame
        self.content_frame = ttk.Frame(self, style="CardContent.TFrame", padding=0)
        
        # Text widget for code
        text_bg = TEXT_BG_DARK_ALT if self.controller.current_theme == "dark" else TEXT_BG_LIGHT_ALT
        text_fg = TEXT_FG_DARK if self.controller.current_theme == "dark" else TEXT_FG_LIGHT
        
        num_lines = self.content.count('\n') + 1
        height = min(num_lines + 5, 40)
        
        self.text_widget = tk.Text(
            self.content_frame,
            height=height,
            font=self.controller.fonts.code_font,
            wrap="none",
            background=text_bg,
            foreground=text_fg,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=12,
            pady=10
        )
        
        # Scrollbars
        v_bar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.text_widget.yview)
        h_bar = ttk.Scrollbar(self.content_frame, orient="horizontal", command=self.text_widget.xview)
        self.text_widget.configure(yscrollcommand=v_bar.set, xscrollcommand=h_bar.set)
        v_bar.pack(side="right", fill="y")
        h_bar.pack(side="bottom", fill="x")
             
        self.text_widget.pack(side="left", fill="both", expand=True)
        
        self.text_widget.insert("1.0", self.content)
        self.text_widget.config(state="disabled")

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_label.config(text="▼")
            self.content_frame.pack(fill="both", expand=True)
        else:
            self.toggle_label.config(text="▶")
            self.content_frame.pack_forget()

    def expand(self):
        if not self.expanded:
            self.toggle()


class CollapsibleFigureCard(CardBorderFrame):
    """A collapsible card for figures (read-only)."""
    
    def __init__(self, parent, title: str, image_path: str, caption: str, controller, start_expanded: bool = False):
        super().__init__(parent, padx=1, pady=1)
        self.title_text = title
        self.image_path = image_path
        self.caption = caption
        self.controller = controller
        self.expanded = False
        
        self._build_ui()
        
        if start_expanded:
            self.expand()

    def _on_theme_changed(self, event=None):
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        self.left_frame.config(bg=header_bg)
        self.toggle_label.config(bg=header_bg, fg=header_fg)
        self.title_label.config(bg=header_bg, fg=header_fg)
            
            
    def _build_ui(self):
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        self.left_frame = tk.Frame(header, bg=header_bg)
        self.left_frame.pack(side="left", fill="x", expand=True)
        self.left_frame.bind("<Button-1>", lambda e: self.toggle())
        
        self.toggle_label = tk.Label(
            self.left_frame,
            text="▶",
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.toggle_label.pack(side="left", padx=(0, 10))
        self.toggle_label.bind("<Button-1>", lambda e: self.toggle())
        
        self.title_label = tk.Label(
            self.left_frame,
            text=self.title_text,
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", lambda e: self.toggle())
        
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame
        self.content_frame = ttk.Frame(self, style="CardContent.TFrame", padding=10)
        
        # Image rendering logic
        try:
            path = Path(self.image_path)
            if not path.exists():
                path = Path("output/experiments/plots") / path.name
                
            pil_img = None
            if path.exists():
                if path.suffix.lower() == '.pdf':
                     doc = fitz.open(path)
                     page = doc[0]
                     mat = fitz.Matrix(2, 2)
                     pix = page.get_pixmap(matrix=mat)
                     mode = "RGBA" if pix.alpha else "RGB"
                     pil_img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                     doc.close()
                else:
                     pil_img = Image.open(path)
                
                if pil_img:
                    # Resize max width 600
                    width = 600
                    w_percent = (width / float(pil_img.size[0]))
                    h_size = int((float(pil_img.size[1]) * float(w_percent)))
                    pil_img = pil_img.resize((width, h_size), Image.Resampling.LANCZOS)
                    
                    self.tk_img = ImageTk.PhotoImage(pil_img) # Keep ref
                    ttk.Label(self.content_frame, image=self.tk_img).pack(pady=(0, 10))
            else:
                 ttk.Label(self.content_frame, text=f"Image not found: {self.image_path}", foreground="red").pack()
                 
        except Exception as e:
            print(f"Error loading image {self.image_path}: {e}")
            ttk.Label(self.content_frame, text=f"Error loading image: {e}", foreground="red").pack()

        # Caption (Read-only)
        if self.caption:
            caption_label = ttk.Label(
                self.content_frame, 
                text=self.caption,
                font=self.controller.fonts.text_area_font,
                wraplength=600,
                justify="left",
                style="CardRow.TLabel"
            )
            caption_label.pack(fill="x")
            
            # Dynamic wrap
            def update_wrap(event):
                 caption_label.config(wraplength=event.width - 20)
            self.content_frame.bind("<Configure>", update_wrap)

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_label.config(text="▼")
            self.content_frame.pack(fill="both", expand=True)
        else:
            self.toggle_label.config(text="▶")
            self.content_frame.pack_forget()

    def expand(self):
        if not self.expanded:
            self.toggle()


class ExperimentResultsScreen(BaseFrame):
    def __init__(self, parent, controller):
        # Dynamic button text based on whether paper draft exists
        paper_draft_file = Path("output/paper_draft.md")
        next_text = "Continue" if paper_draft_file.exists() else "Write Paper"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Experiment Results",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate",
            header_file_path=Path("output/experiments/experiment_result.json"),
            info_content=EXPERIMENT_RESULTS_INFO
        )
        
        self.cards = []
        self.current_code_path = None

    def on_back(self):
        """Override back navigation to skip ExperimentPlanScreen when user experiment is active."""
        if Settings.USER_EXPERIMENT_FILE:
            self.controller.current_screen_index -= 1  # skip plan screen
        super().on_back()

    def create_content(self):
        self.results_container = ttk.Frame(self.scrollable_frame, style="Scrollable.TFrame")
        self.results_container.pack(fill="x", expand=True)

    def _load_and_display_results(self):
        """Load and display experiment results."""
        if not Path(HYPOTHESES_FILE).exists():
            self.show_error_message("Hypothesis Error", f"Hypotheses file not found: {HYPOTHESES_FILE}")
            return
        
        try:
            # Clear previous
            for widget in self.results_container.winfo_children():
                widget.destroy()
            self.cards.clear()

            # Load
            selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
            if selected_hypothesis is None:
                self.show_error_message("Hypothesis Error", "No hypothesis found")
                return
            
            experiment_result_file = Path("output/experiments/experiment_result.json")
            if not experiment_result_file.exists():
                self.show_error_message("Result Error", f"Experiment result not found: {experiment_result_file}")
                return
            
            experiment_result = ExperimentRunner.load_experiment_result(str(experiment_result_file))
            self.experiment_result = experiment_result
            
            # --- 1. Verdict Section ---
            verdict = experiment_result.hypothesis_evaluation.verdict
            reasoning = experiment_result.hypothesis_evaluation.reasoning
            verdict_text = f"**Verdict:** {verdict}\n\n**Reasoning:** {reasoning}"
            
            verdict_card = CollapsibleTextCard(
                self.results_container,
                "Verdict",
                verdict_text,
                self.controller,
                start_expanded=True
            )
            verdict_card.pack(fill="x", pady=10)
            self.cards.append(verdict_card)
            
            # --- 2. Code Section ---
            code_path = Path("output/experiments/experiment.py")
            if not code_path.exists():
                 if Path("rbql_vs_q_gemini.py").exists(): code_path = Path("rbql_vs_q_gemini.py")
            
            self.current_code_path = code_path
            code_content = "# No code found"
            if code_path.exists():
                try:
                    with open(code_path, "r", encoding="utf-8") as f:
                        code_content = f.read()
                except:
                    pass
            
            code_card = CollapsibleCodeCard(
                self.results_container,
                "Experiment Code",
                code_content,
                self.controller,
                on_edit=self._open_code_in_editor,
                on_show_explorer=self._show_code_in_explorer,
                on_execute=self._execute_current_code,
                start_expanded=False
            )
            code_card.pack(fill="x", pady=10)
            self.cards.append(code_card)
            
            # --- 3. Figures Section ---
            if experiment_result.plots:
                 for idx, plot in enumerate(experiment_result.plots, 1):
                      fig_card = CollapsibleFigureCard(
                           self.results_container,
                           f"Figure {idx}: {Path(plot.filename).name}",
                           plot.filename,
                           plot.caption,
                           self.controller,
                           start_expanded=True
                      )
                      fig_card.pack(fill="x", pady=10)
                      self.cards.append(fig_card)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.show_error_message("Error", f"Error loading results: {e}")

    def _open_code_in_editor(self):
        """Open the experiment code file in the system's default editor."""
        if not self.current_code_path or not self.current_code_path.exists():
            tk.messagebox.showwarning("No Code", "No experiment code file found.")
            return
        
        path = str(self.current_code_path.absolute())
        print(f"Opening {path} in editor")
        
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':
                subprocess.call(('open', path))
            else:
                subprocess.call(('xdg-open', path))
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not open file: {e}")

    def _show_code_in_explorer(self):
        """Reveal the experiment code file in the file explorer."""
        if not self.current_code_path or not self.current_code_path.exists():
            return
             
        print(f"Showing {self.current_code_path} in explorer")
        path = os.path.abspath(self.current_code_path)
        path = os.path.normpath(path)
        
        if platform.system() == 'Windows':
            subprocess.call(['explorer', '/select,', path])
        elif platform.system() == 'Darwin':
            subprocess.call(['open', '-R', path])
        else:
            # Linux - try to just open the directory
            subprocess.call(['xdg-open', os.path.dirname(path)])

    def _execute_current_code(self):
        """Execute the current experiment code and refresh the UI with new results."""
        if not self.current_code_path or not self.current_code_path.exists():
            tk.messagebox.showwarning("No Code", "No experiment code file found.")
            return
        
        if not tk.messagebox.askyesno("Execute Code", "Execute the current experiment code?\n\nThis will run the code and regenerate the verdict and figures."):
            return
        
        popup = ProgressPopup(self.controller, "Executing Experiment")
        
        def task():
            try:
                self.after(0, lambda: popup.update_status("Executing code"))
                
                runner = ExperimentRunner()
                
                # Clear old plots before execution
                plots_dir = os.path.join(runner.base_output_dir, "plots")
                if os.path.exists(plots_dir):
                    for file in os.listdir(plots_dir):
                        file_path = os.path.join(plots_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(f"Warning: Failed to delete {file_path}: {e}")
                
                code_file_abs = str(self.current_code_path.absolute())
                
                # Execute the code file
                execution_result = runner.executor.execute_file(
                    code_file_abs,
                    output_dir=runner.base_output_dir
                )
                
                if execution_result.return_code != 0:
                    error_msg = execution_result.stderr or "Unknown error"
                    self.after(0, lambda e=error_msg: popup.show_error(f"Execution failed: {e[:500]}"))
                    return
                
                self.after(0, lambda: popup.update_status("Validating results"))
                
                # Load the hypothesis and experiment plan
                hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
                experiment_plan = runner.load_experiment_plan()
                
                # Validate results
                validation_result = runner._validate_experiment_results(
                    execution_result,
                    experiment_plan,
                    hypothesis,
                    code_file_abs
                )
                
                self.after(0, lambda: popup.update_status("Generating verdict and captions"))
                
                # Generate plot captions if plots exist
                plot_captions = []
                if execution_result.plot_files:
                    plot_captions = runner._generate_plot_captions(
                        execution_result.plot_files,
                        hypothesis,
                        experiment_plan,
                        execution_result.stdout
                    )
                
                # Determine verdict
                import lmstudio as lms
                from settings import Settings
                import textwrap
                from phases.experimentation.experiment_state import VerdictResult, HypothesisEvaluation, ExperimentResult
                from utils.llm_utils import remove_thinking_blocks
                
                verdict = "inconclusive"
                reasoning = ""
                
                # Truncate stdout
                stdout_summary = execution_result.stdout
                if len(stdout_summary) > 2000:
                    stdout_summary = stdout_summary[:500] + "\n...[truncated output]...\n" + stdout_summary[-1500:]
                
                # Check validation status for warning
                validation_warning = ""
                if not validation_result.is_valid:
                    validation_warning = f"\n[VALIDATION WARNING]\nThe code validator flagged potential issues: {validation_result.reasoning}\nReview the results carefully to ensure they are valid despite this warning.\n"
                
                # Determine verdict using shared Runner logic
                # (Passing plot_captions directly as list of Plot objects)
                verdict, reasoning = runner._determine_verdict(
                    hypothesis,
                    stdout_summary,
                    plot_captions,
                    validation_warning
                )
                
                # Create result object (mocking lms parsed result for compatibility or just creating new)
                # The existing code expects verdict_result object
                verdict_result = VerdictResult(verdict=verdict, reasoning=reasoning)

                # Load existing result to preserve some fields
                experiment_code = ""
                try:
                    with open(self.current_code_path, 'r', encoding='utf-8') as f:
                        experiment_code = f.read()
                except:
                    pass
                
                # Create new experiment result
                new_result = ExperimentResult(
                    hypothesis=hypothesis,
                    experiment_plan=experiment_plan,
                    experiment_code=experiment_code,
                    execution_result=execution_result,
                    validation_result=validation_result,
                    hypothesis_evaluation=HypothesisEvaluation(
                        hypothesis_id=hypothesis.id,
                        verdict=verdict,
                        reasoning=reasoning
                    ),
                    plots=plot_captions,
                    fix_attempts=0,
                    validation_attempts=1,
                    execution_time=None
                )
                
                # Save the new result
                runner.save_experiment_result(new_result)
                
                self.after(0, lambda: self._on_execute_success(popup))
                
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                traceback.print_exc()
                self.after(0, lambda err=tb_str: popup.show_error(err))
        
        threading.Thread(target=task, daemon=True).start()
    
    def _on_execute_success(self, popup):
        """Handle successful code execution."""
        popup.close()
        self._results_loaded = False  # Force reload
        self._load_and_display_results()
        self.controller.apply_theme_colors(self)

    def on_show(self):
        # If user experiment is active and no results exist yet, auto-run it
        experiment_result_file = Path("output/experiments/experiment_result.json")
        if Settings.USER_EXPERIMENT_FILE and not experiment_result_file.exists():
            self._run_user_experiment()
            return

        if not hasattr(self, '_results_loaded') or not self._results_loaded:
            self._load_and_display_results()
            self._results_loaded = True

        # Update next button text based on paper draft existence
        paper_draft_file = Path("output/paper_draft.md")
        next_text = "Continue" if paper_draft_file.exists() else "Write Paper"
        self.set_next_text(next_text)

        # Disable regenerate button if user provided their own experiment file
        if hasattr(self, 'regenerate_btn'):
            if Settings.USER_EXPERIMENT_FILE:
                self.regenerate_btn.config(state="disabled")
                # Add a tooltip or just rely on state
            else:
                self.regenerate_btn.config(state="normal")

    def _run_user_experiment(self):
        """Run the user-provided experiment file."""
        popup = ProgressPopup(self.controller, "Running User Experiment")

        def task():
            try:
                def status_callback(status: str):
                    self.after(0, lambda s=status: popup.update_status(s))

                ExperimentRunner.run_user_experiment(status_callback=status_callback)
                self.after(0, lambda: self._on_user_experiment_success(popup))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))

        threading.Thread(target=task, daemon=True).start()

    def _on_user_experiment_success(self, popup):
        """Handle successful user experiment completion."""
        popup.close()
        self._results_loaded = False
        self._load_and_display_results()
        self.controller.apply_theme_colors(self)
            
    def on_next(self):
        from pathlib import Path
        paper_draft_file = Path("output/paper_draft.md")
        if paper_draft_file.exists():
            super().on_next()
        else:
            self._run_paper_generation()

    def _run_paper_generation(self):
        """Run the critique-based paper writing pipeline."""
        popup = ProgressPopup(self.controller, "Writing Paper")
        
        def task():
            try:
                def status_callback(status: str):
                    self.after(0, lambda s=status: popup.update_status(s))
                
                PaperWritingPipeline.generate_new_draft(status_callback=status_callback)
                
                self.after(0, lambda: self._on_generation_success(popup))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        threading.Thread(target=task, daemon=True).start()

    def _on_generation_success(self, popup):
        popup.close()
        self.controller.next_screen()

    def on_regenerate(self):
        is_user_experiment = bool(Settings.USER_EXPERIMENT_FILE)
        if is_user_experiment:
            # Should be disabled in UI, but just in case
            return

        confirm_msg = "Re-run the experiment from scratch?\nThis will clear the output directory and generate/execute new code based on the current experiment plan."
        
        if not tk.messagebox.askyesno("Regenerate", confirm_msg):
            return

        popup = ProgressPopup(self.controller, "Regenerating Experiment")

        def task():
            try:
                def status_callback(status: str):
                    self.after(0, lambda s=status: popup.update_status(s))

                if is_user_experiment:
                    ExperimentRunner.run_user_experiment(status_callback=status_callback)
                else:
                    ExperimentRunner.run_new_experiment(status_callback=status_callback)
                self.after(0, lambda: self._on_rerun_success(popup))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))

        threading.Thread(target=task, daemon=True).start()

    def _on_rerun_success(self, popup):
        popup.close()
        self._load_and_display_results()
        # Apply theme colors to newly created cards (fixes missing borders)
        self.controller.apply_theme_colors(self)
