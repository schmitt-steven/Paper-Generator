import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

from ..base_frame import BaseFrame, ProgressPopup, create_scrollable_text_area
from .writing_prompts_screen import WritingPromptsScreen
from utils.file_utils import load_markdown, save_markdown
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.paper_search.literature_search import LiteratureSearch
from phases.latex_generation.paper_converter import PaperConverter
from phases.latex_generation.metadata import LaTeXMetadata
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.context_analysis.user_requirements import UserRequirements
from phases.context_analysis.paper_conception import PaperConception
from settings import Settings


PAPER_DRAFT_FILE = "paper_draft.md"
OUTPUT_DIR = "output"
HYPOTHESES_FILE = "output/hypothesis.md"

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
            regenerate_text="Regenerate",
            header_file_path=Path(OUTPUT_DIR) / PAPER_DRAFT_FILE
        )

    def create_content(self):
        """Create the card container - content is added when draft loads."""
        # Create the main card that will hold everything
        self.card = ttk.Frame(self.scrollable_frame, style="Card.TFrame", padding=1)
        self.card.pack(fill="both", expand=True)
        
        # Header with title and button
        header = ttk.Frame(self.card, style="CardHeader.TFrame", padding=(10, 8))
        header.pack(fill="x")
        
        # Get colors
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        
        # Title on left
        tk.Label(
            header,
            text="Paper Draft",
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Show Prompts button on right
        ttk.Button(
            header,
            text="Show Prompts",
            command=self._show_prompts
        ).pack(side="right")
        
        ttk.Separator(self.card, orient="horizontal").pack(fill="x")
        
        # Content frame for the text area (no padding)
        self.card_content = ttk.Frame(self.card, padding=0)
        self.card_content.pack(fill="both", expand=True)
    
    def _show_prompts(self):
        """Navigate to the Writing Prompts screen."""
        self.controller.show_frame(WritingPromptsScreen)

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
        
        # Create the editable text area inside the card
        self._create_draft_section(draft_content)

    def _show_error(self, message: str):
        """Display an error message."""
        error_frame = ttk.Frame(self.card_content, padding="20")
        error_frame.pack(fill="x", pady=20)
        
        ttk.Label(
            error_frame,
            text=message,
            font=self.controller.fonts.default_font,
            foreground="red",
            wraplength=500
        ).pack()

    def _create_draft_section(self, content: str):
        """Create the text area inside the card content."""
        # Container for text + scrollbar (no outer padding)
        container, self.draft_text = create_scrollable_text_area(
            self.card_content,
            height=40,
            font=self.controller.fonts.text_area_font
        )
        container.pack(fill="both", expand=True)
        
        self.draft_text.insert("1.0", content)

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
                # Load hypothesis
                self.after(0, lambda: popup.update_status("Loading hypothesis"))
                selected_hypothesis = HypothesisBuilder.load_hypothesis(HYPOTHESES_FILE)
                
                if selected_hypothesis is None:
                    raise ValueError("No hypothesis found")
                
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
        """Called when screen is shown - load draft."""
        draft_path = Path(OUTPUT_DIR) / PAPER_DRAFT_FILE
        
        # Load draft if we haven't yet and it exists
        if not hasattr(self, 'draft_text'):
            if draft_path.exists():
                self._load_draft()
            else:
                # This shouldn't happen since Evidence Screen generates before navigating
                print("[PaperDraftScreen] Warning: No paper draft found")

    def on_regenerate(self):
        """Regenerate the paper draft from scratch."""
        if not tk.messagebox.askyesno(
            "Confirm Regeneration", 
            "This will regenerate the entire paper draft based on your experiment results and evidence. Any manual edits will be lost.\n\nDo you want to continue?"
        ):
            return
        
        self._run_paper_generation(is_regeneration=True)

    def _run_paper_generation(self, is_regeneration: bool = False):
        """Generate paper draft from edited evidence.
        
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
                
                # 2. Initialize pipeline
                pipeline = PaperWritingPipeline()
                
                # 3. Write Paper using edited evidence from evidence.json
                def status_update(msg):
                    self.after(0, lambda: popup.update_status(msg))
                
                self.after(0, lambda: popup.update_status("Starting paper generation"))
                
                pipeline.write_paper_from_evidence(
                    paper_concept=paper_concept,
                    experiment_result=experiment_result,
                    user_requirements=user_requirements,
                    status_callback=status_update
                )
                
                # 4. Complete
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
        
        # Refresh text widget
        try:
             content = load_markdown(PAPER_DRAFT_FILE, OUTPUT_DIR)
             self.draft_text.delete("1.0", "end")
             self.draft_text.insert("1.0", content)
        except Exception as e:
             print(f"Error refreshing draft text: {e}")
