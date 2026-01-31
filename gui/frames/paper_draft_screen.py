import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

from ..base_frame import BaseFrame, ProgressPopup, create_scrollable_text_area, CardBorderFrame
from ..markdown_view import MarkdownView
from ..info_texts import PAPER_DRAFT_INFO
from ..theme_colors import (
    CARD_HEADER_BG_DARK, CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT, TEXT_FG_DARK, TEXT_FG_LIGHT
)
from .writing_prompts_screen import WritingPromptsScreen
from utils.file_utils import load_markdown, save_markdown
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.paper_search.literature_search import LiteratureSearch
from phases.latex_generation.paper_converter import PaperConverter
from phases.latex_generation.paper_converter import LaTeXMetadata
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.context_analysis.user_requirements import UserRequirements
from phases.context_analysis.paper_conception import PaperConception
from settings import Settings


PAPER_DRAFT_FILE = "paper_draft.md"
OUTPUT_DIR = "output"
HYPOTHESES_FILE = "output/hypothesis.md"
LATEX_PAPER_FILE = Path("output/latex/paper.tex")


class CollapsibleDraftCard(CardBorderFrame):
    """A collapsible card for the paper draft (read-only)."""
    
    def __init__(self, parent, title: str, content: str, controller, 
                 on_show_prompts=None, start_expanded: bool = True):
        super().__init__(parent, padx=1, pady=1)
        self.title_text = title
        self.content = content
        self.controller = controller
        self.on_show_prompts = on_show_prompts
        self.expanded = False
        
        self._build_ui()
        
        if start_expanded:
            self.expand()
    
    def _build_ui(self):
        # Header
        header = ttk.Frame(self, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        
        # Left side: Toggle + Title
        left_frame = tk.Frame(header, bg=header_bg)
        left_frame.pack(side="left", fill="x", expand=True)
        left_frame.bind("<Button-1>", lambda e: self.toggle())
        
        self.toggle_label = tk.Label(
            left_frame,
            text="▶",
            font=self.controller.fonts.default_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.toggle_label.pack(side="left", padx=(0, 10))
        self.toggle_label.bind("<Button-1>", lambda e: self.toggle())
        
        self.title_label = tk.Label(
            left_frame,
            text=self.title_text,
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg,
            cursor="hand2"
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", lambda e: self.toggle())
        
        # Right side: Buttons
        btn_frame = tk.Frame(header, bg=header_bg)
        btn_frame.pack(side="right")
        
        # Show Prompts Button (Custom)
        if self.on_show_prompts:
            prompts_btn = ttk.Button(btn_frame, text="Show Prompts", command=self.on_show_prompts)
            prompts_btn.pack(side="left")
            
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        
        # Content frame
        self.content_frame = ttk.Frame(self, style="CardContent.TFrame", padding=0)
        
        # Markdown View
        self.text_widget = MarkdownView(
            self.content_frame,
            font_manager=self.controller.fonts,
            theme_mode=self.controller.current_theme,
            padx=12,
            pady=10,
            height=400 # Default height
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        self.text_widget.set_markdown(self.content)

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


class PaperDraftScreen(BaseFrame):
    def __init__(self, parent, controller):
        next_text = "Continue" if LATEX_PAPER_FILE.exists() else "Generate LaTeX"
        
        super().__init__(
            parent=parent,
            controller=controller,
            title="Paper Draft",
            next_text=next_text,
            has_regenerate=True,
            regenerate_text="Regenerate",
            header_file_path=Path(OUTPUT_DIR) / PAPER_DRAFT_FILE,
            info_content=PAPER_DRAFT_INFO
        )
        self.card = None
        
        # Bind resize event to adjust card height
        self._canvas.bind("<Configure>", self._update_card_height, add="+")

    def _update_card_height(self, event=None):
        """Dynamically adjust card height to fill available space."""
        if not self.card:
            return
            
        # Get canvas height
        canvas_height = self._canvas.winfo_height()
        
        # Determine target height (canvas height - margins - card header)
        # Margins: BaseFrame padding ~20, Card margin ~20, Card header ~40, plus safe zone
        target_height = canvas_height - 147 
        
        # Minimum safe height
        if target_height < 400:
            target_height = 400
            
        # Update the card
        if hasattr(self.card, 'text_widget'):
            self.card.text_widget.configure(height=target_height)

    def create_content(self):
        """Create the container."""
        # Content is dynamic based on file presence
        pass
    
    def _show_prompts(self):
        """Navigate to the Writing Prompts screen."""
        self.controller.show_frame(WritingPromptsScreen)

    def _load_draft(self):
        """Load paper draft from file and display it."""
        # Clear existing card and reload content
        if self.card:
            self.card.destroy()
            self.card = None
            
        try:
            draft_content = load_markdown(PAPER_DRAFT_FILE, OUTPUT_DIR)
            # Replace .pdf with .png for preview purposes (tkinterweb can't show PDFs)
            draft_content = draft_content.replace(".pdf", "_preview.png")
            draft_content = draft_content.replace("experiments/plots/", "output/experiments/plots/")
        except (FileNotFoundError, Exception) as e:
            msg = f"Paper draft not found: {OUTPUT_DIR}/{PAPER_DRAFT_FILE}" if isinstance(e, FileNotFoundError) else f"Error loading draft: {e}"
            self.show_error_message("Paper Draft Error", msg)
            
            # Add a button to regenerate if missing (appended after error message)
            ttk.Button(self.scrollable_frame, text="Generate Draft", command=self.on_regenerate).pack(pady=10)
            return
        
        # Create Card
        self.card = CollapsibleDraftCard(
            self.scrollable_frame,
            "Draft",
            draft_content,
            self.controller,
            on_show_prompts=self._show_prompts,
            start_expanded=True
        )
        self.card.pack(fill="x", pady=10)
        
        # Force initial resize
        self.after(100, self._update_card_height)



    def on_next(self):
        """Proceed to next screen or generate LaTeX."""
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
                # Load hypothesis
                self.after(0, lambda: popup.update_status("Loading hypothesis"))
                selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
                
                if selected_hypothesis is None:
                    raise ValueError("No hypothesis found")
                
                experiment_result = None
                # Load experiment result
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
                    progress_callback=lambda msg: self.after(0, lambda: popup.update_status(msg))
                )
                
                # Compile LaTeX
                self.after(0, lambda: popup.update_status("Compiling LaTeX to PDF"))
                success = converter.compile_latex(latex_dir)
                
                if success:
                    # Close popup and continue
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
        """Called when screen is shown - load draft."""
        # Always reload to ensure content is fresh (since it's read-only in GUI now)
        draft_path = Path(OUTPUT_DIR) / PAPER_DRAFT_FILE
        if draft_path.exists():
            self._load_draft()
        else:
             msg = f"No paper draft found at {draft_path}.\nPlease generate the draft."
             self.show_error_message("Paper Draft Not Found", msg)
             ttk.Button(self.scrollable_frame, text="Generate Draft", command=self.on_regenerate).pack(pady=10)

    def on_regenerate(self):
        """Regenerate the paper draft from scratch."""
        if not tk.messagebox.askyesno(
            "Confirm Regeneration", 
            "This will regenerate the entire paper draft based on the current data. Any manual edits will be lost.\n\nDo you want to continue?"
        ):
            return
        
        self._run_paper_generation(is_regeneration=True)

    def _run_paper_generation(self, is_regeneration: bool = False):
        """Generate paper draft using critique-based pipeline.
        
        Args:
            is_regeneration: True if regenerating (refresh text), False if initial (load draft)
        """
        title = "Regenerating Paper Draft" if is_regeneration else "Generating Paper Draft"
        popup = ProgressPopup(self.controller, title)
        
        def task():
            try:
                # 1. Load context
                self.after(0, lambda: popup.update_status("Loading context"))
                
                paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
                
                # Load papers for the pipeline
                papers = LiteratureSearch.load_papers("output/papers.json")
                
                # Load experiment result
                experiment_result = None
                experiment_result_file = "output/experiments/experiment_result.json"
                if Path(experiment_result_file).exists():
                    experiment_result = ExperimentRunner.load_experiment_result(experiment_result_file)
                else:
                    raise ValueError("No experiment results found. Please run experiments first.")
                
                user_requirements = None
                try:
                    user_requirements = UserRequirements.load_user_requirements("user_files/user_requirements.md")
                except:
                    pass
                
                # 2. Create pipeline and write paper
                pipeline = PaperWritingPipeline()
                
                def status_update(msg):
                    self.after(0, lambda: popup.update_status(msg))
                
                self.after(0, lambda: popup.update_status("Starting paper generation"))
                
                pipeline.write_paper(
                    paper_concept=paper_concept,
                    experiment_result=experiment_result,
                    papers=papers,
                    user_requirements=user_requirements,
                    status_callback=status_update
                )
                
                # 3. Complete
                if is_regeneration:
                    self.after(0, lambda: self._on_regeneration_complete(popup))
                else:
                    self.after(0, lambda: self._on_generation_complete(popup))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): popup.show_error(err))

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_generation_complete(self, popup: ProgressPopup):
        """Handle initial generation completion."""
        popup.close()
        # Load the newly generated draft
        self._load_draft()

    def _on_regeneration_complete(self, popup: ProgressPopup):
        """Handle regeneration completion."""
        popup.close()
        # Load the newly generated draft
        self._load_draft()
        self.after(200, lambda: tk.messagebox.showinfo("Success", "Paper draft successfully regenerated."))

