import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

from ..base_frame import BaseFrame, ProgressPopup, create_text_area
from utils.file_utils import load_markdown, save_markdown
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.paper_search.literature_search import LiteratureSearch
from phases.latex_generation.paper_converter import PaperConverter
from phases.latex_generation.metadata import LaTeXMetadata
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner


PAPER_DRAFT_FILE = "paper_draft.md"
OUTPUT_DIR = "output"
HYPOTHESES_FILE = "output/hypotheses.json"

LATEX_PAPER_FILE = Path("output/latex/paper.tex")


class PaperDraftScreen(BaseFrame):
    def __init__(self, parent, controller):
        self.draft_text: tk.Text
        
        # Dynamic button text based on whether output file exists
        next_text = "Continue" if LATEX_PAPER_FILE.exists() else "Generate LaTeX"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Paper Draft",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate"
        )

    def create_content(self):
        # Info text
        self._create_info_section()
        
        # Don't load draft yet - wait until screen is shown
        # This prevents loading on app startup

    def _create_info_section(self):
        """Create the info text section."""
        explanation_frame = ttk.Frame(self.scrollable_frame)
        explanation_frame.pack(fill="x", pady=(0, 10))
        
        explanation_text = (
            "Review and edit the paper draft below.\n"
            "This draft was generated based on your hypothesis, experiments, and literature.\n"
            "It will be converted to LaTeX format in the next step."
        )

        label = ttk.Label(
            explanation_frame,
            text=explanation_text,
            font=self.controller.fonts.default_font,
            foreground="gray",
            justify="left"
        )
        label.pack(anchor="w", fill="x")

        def set_wraplength(event):
            label.config(wraplength=event.width - 10)
        label.bind("<Configure>", set_wraplength)

    def _load_draft(self):
        """Load paper draft from file and display it."""
        try:
            draft_content = load_markdown(PAPER_DRAFT_FILE, OUTPUT_DIR)
        except FileNotFoundError:
            self._show_error(
                f"Paper draft not found: {OUTPUT_DIR}/{PAPER_DRAFT_FILE}\n\n"
                "Please complete the previous steps first."
            )
            return
        except Exception as e:
            self._show_error(f"Error loading paper draft: {e}")
            return
        
        # Create the editable text area
        self._create_draft_section(draft_content)

    def _show_error(self, message: str):
        """Display an error message."""
        error_frame = ttk.Frame(self.scrollable_frame, padding="20")
        error_frame.pack(fill="x", pady=20)
        
        ttk.Label(
            error_frame,
            text=message,
            font=self.controller.fonts.default_font,
            foreground="red",
            wraplength=500
        ).pack()

    def _create_draft_section(self, content: str):
        """Create a labeled section with an editable text area for the draft."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Paper Draft", padding="10")
        frame.pack(fill="both", expand=True, pady=10)
        
        # Container for text + scrollbar
        editor_container = ttk.Frame(frame)
        editor_container.pack(fill="both", expand=True)

        v_scroll = ttk.Scrollbar(editor_container, orient="vertical")
        
        # Create text area with fixed height
        self.draft_text = tk.Text(
            editor_container,
            height=35,  # Fixed height
            wrap="word",
            font=self.controller.fonts.text_area_font,
            padx=8,
            pady=8,
            spacing2=4,
            spacing3=4,
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            yscrollcommand=v_scroll.set
        )
        
        v_scroll.config(command=self.draft_text.yview)
        
        v_scroll.pack(side="right", fill="y")
        self.draft_text.pack(side="left", fill="both", expand=True)
        
        self.draft_text.insert("1.0", content)
        
        # Bind mousewheel to allow normal scrolling inside the text area
        # We don't propagate to parent canvas anymore since it's a scrollable container itself
        def on_text_mousewheel(event):
            pass
            
        # self.draft_text.bind("<MouseWheel>", on_text_mousewheel)

    def _save_draft(self):
        """Save the edited draft."""
        if not hasattr(self, 'draft_text'):
            return
        
        # Get content from text widget
        draft_content = self.draft_text.get("1.0", "end-1c")
        
        try:
            save_markdown(draft_content, PAPER_DRAFT_FILE, OUTPUT_DIR)
            print(f"[PaperDraft] Saved changes to {OUTPUT_DIR}/{PAPER_DRAFT_FILE}")
        except Exception as e:
            print(f"[PaperDraft] Failed to save: {e}")

    def on_next(self):
        """Save the edited draft and proceed or generate LaTeX."""
        # Always save first
        self._save_draft()
        
        # Check if output exists
        if LATEX_PAPER_FILE.exists():
            super().on_next()
        else:
            self._run_generation()

    def _run_generation(self):
        """Run LaTeX conversion with progress popup."""
        popup = ProgressPopup(self.controller, "Generating LaTeX")
        
        def task():
            try:
                # Load paper draft
                self.after(0, lambda: popup.update_status("Loading paper draft"))
                paper_draft = PaperWritingPipeline.load_paper_draft(f"{OUTPUT_DIR}/{PAPER_DRAFT_FILE}")
                
                # Load indexed papers
                self.after(0, lambda: popup.update_status("Loading indexed papers"))
                indexed_papers = LiteratureSearch.load_papers("output/papers.json")
                
                # Load experiment result
                self.after(0, lambda: popup.update_status("Loading experiment results"))
                hypotheses = HypothesisBuilder.load_hypotheses(HYPOTHESES_FILE)
                selected_hypothesis = None
                for hyp in hypotheses:
                    if hyp.selected_for_experimentation:
                        selected_hypothesis = hyp
                        break
                if selected_hypothesis is None and hypotheses:
                    selected_hypothesis = hypotheses[0]
                
                experiment_result = None
                # Load experiment result (simplified filename without hypothesis ID)
                experiment_result_file = "output/experiments/experiment_result.json"
                if Path(experiment_result_file).exists():
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
                )
                
                # Compile LaTeX
                self.after(0, lambda: popup.update_status("Compiling LaTeX to PDF"))
                success = converter.compile_latex(latex_dir)
                
                if success:
                    # Success - close popup and proceed
                    self.after(0, lambda: self._on_generation_success(popup))
                else:
                    self.after(0, lambda: popup.show_error("LaTeX compilation failed. Check logs for details."))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_generation_success(self, popup: ProgressPopup):
        """Handle successful generation."""
        popup.close()
        self.controller.next_screen()
    
    def on_show(self):
        """Called when screen is shown - load draft if not already loaded."""
        # Only load if we haven't loaded yet
        if not hasattr(self, 'draft_text'):
            draft_path = Path(OUTPUT_DIR) / PAPER_DRAFT_FILE
            if draft_path.exists():
                self._load_draft()
