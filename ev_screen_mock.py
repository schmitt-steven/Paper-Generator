import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Paper:
    id: str
    title: str
    authors: str
    year: int
    citations: int

@dataclass
class EvidenceChunk:
    id: str
    section: str
    paper: Paper
    summary: str
    llm_score: float
    vector_score: float

# Mock data
PAPERS = [
    Paper("p1", "Attention Is All You Need", "Vaswani et al.", 2017, 95000),
    Paper("p2", "BERT: Pre-training of Deep Bidirectional Transformers", "Devlin et al.", 2019, 78000),
    Paper("p3", "Language Models are Few-Shot Learners", "Brown et al.", 2020, 42000),
    Paper("p4", "Chain-of-Thought Prompting", "Wei et al.", 2022, 8500),
]

SECTIONS = ["Introduction", "Related Work", "Methods", "Results", "Discussion", "Conclusion"]

class CollapsibleSection(ttk.Frame):
    """A collapsible section widget with header and content area."""
    
    def __init__(self, parent, title: str, on_delete_chunk=None):
        super().__init__(parent)
        self.title = title
        self.expanded = False
        self.chunks: List[EvidenceChunk] = []
        self.on_delete_chunk = on_delete_chunk
        self.chunk_widgets = {}
        
        # Header frame
        self.header = ttk.Frame(self)
        self.header.pack(fill=tk.X, pady=(0, 2))
        
        self.toggle_btn = ttk.Button(self.header, text="▶", width=3, command=self.toggle)
        self.toggle_btn.pack(side=tk.LEFT)
        
        self.title_label = ttk.Label(self.header, text=f"{title} (0 chunks)", font=("Segoe UI", 10, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=5)
        
        # Content frame (hidden by default)
        self.content = ttk.Frame(self)
        self.chunks_container = ttk.Frame(self.content)
        self.chunks_container.pack(fill=tk.BOTH, expand=True, padx=(20, 0))
        
    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_btn.config(text="▼")
            self.content.pack(fill=tk.BOTH, expand=True)
        else:
            self.toggle_btn.config(text="▶")
            self.content.pack_forget()
    
    def update_title(self):
        self.title_label.config(text=f"{self.title} ({len(self.chunks)} chunks)")
    
    def add_chunk(self, chunk: EvidenceChunk):
        self.chunks.append(chunk)
        self._render_chunk(chunk)
        self.update_title()
    
    def _render_chunk(self, chunk: EvidenceChunk):
        frame = ttk.LabelFrame(self.chunks_container, text=f"[{chunk.paper.year}] {chunk.paper.title[:50]}...")
        frame.pack(fill=tk.X, pady=3, padx=2)
        
        # Metadata row
        meta = ttk.Frame(frame)
        meta.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(meta, text=f"Authors: {chunk.paper.authors}", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Label(meta, text=f"Citations: {chunk.paper.citations:,}", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=10)
        ttk.Label(meta, text=f"Scores: LLM={chunk.llm_score:.2f} | Vec={chunk.vector_score:.2f}", 
                  font=("Segoe UI", 8), foreground="gray").pack(side=tk.LEFT, padx=10)
        
        del_btn = ttk.Button(meta, text="✕", width=3, 
                             command=lambda c=chunk, f=frame: self._delete_chunk(c, f))
        del_btn.pack(side=tk.RIGHT)
        
        # Summary text
        text = scrolledtext.ScrolledText(frame, height=3, wrap=tk.WORD, font=("Segoe UI", 9))
        text.pack(fill=tk.X, padx=5, pady=(0, 5))
        text.insert(tk.END, chunk.summary)
        text.config(state=tk.DISABLED)
        
        self.chunk_widgets[chunk.id] = frame
    
    def _delete_chunk(self, chunk: EvidenceChunk, frame: ttk.Frame):
        if messagebox.askyesno("Delete Chunk", "Remove this evidence chunk?"):
            self.chunks.remove(chunk)
            frame.destroy()
            del self.chunk_widgets[chunk.id]
            self.update_title()
            if self.on_delete_chunk:
                self.on_delete_chunk(chunk)


class AddChunkDialog(tk.Toplevel):
    """Dialog for adding a new evidence chunk."""
    
    def __init__(self, parent, papers: List[Paper], sections: List[str], on_add):
        super().__init__(parent)
        self.title("Add New Evidence Chunk")
        self.geometry("500x450")
        self.resizable(False, False)
        self.on_add = on_add
        self.papers = papers
        
        self.transient(parent)
        self.grab_set()
        
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Section selection
        ttk.Label(main, text="Target Section(s):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        
        self.section_vars = {}
        sections_frame = ttk.Frame(main)
        sections_frame.pack(fill=tk.X, pady=(2, 10))
        
        for i, sec in enumerate(sections):
            var = tk.BooleanVar()
            self.section_vars[sec] = var
            cb = ttk.Checkbutton(sections_frame, text=sec, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5)
        
        # Paper selection
        ttk.Label(main, text="Source Paper:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        
        self.paper_var = tk.StringVar()
        self.paper_combo = ttk.Combobox(main, textvariable=self.paper_var, width=60)
        self.paper_values = [f"[{p.year}] {p.title} - {p.authors}" for p in papers]
        self.paper_combo['values'] = self.paper_values
        self.paper_combo.pack(fill=tk.X, pady=(2, 10))
        
        # Bind for filtering as user types
        self.paper_var.trace_add('write', self._filter_papers)
        
        # Quote/Summary text
        ttk.Label(main, text="Evidence Text / Summary:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        
        self.text_input = scrolledtext.ScrolledText(main, height=10, wrap=tk.WORD, font=("Segoe UI", 9))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(2, 10))
        
        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Add Chunk", command=self._submit).pack(side=tk.RIGHT)
    
    def _submit(self):
        selected_sections = [s for s, v in self.section_vars.items() if v.get()]
        paper_idx = next((i for i, p in enumerate(self.papers) 
                         if f"[{p.year}] {p.title}" in self.paper_var.get()), None)
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not selected_sections:
            messagebox.showwarning("Missing Info", "Select at least one section.")
            return
        if paper_idx is None:
            messagebox.showwarning("Missing Info", "Select a source paper.")
            return
        if not text:
            messagebox.showwarning("Missing Info", "Enter the evidence text.")
            return
        
        self.on_add(selected_sections, self.papers[paper_idx], text)
        self.destroy()
    
    def _filter_papers(self, *args):
        typed = self.paper_var.get().lower()
        if not typed:
            self.paper_combo['values'] = self.paper_values
        else:
            filtered = [p for p in self.paper_values if typed in p.lower()]
            self.paper_combo['values'] = filtered if filtered else self.paper_values


class EvidenceManagerScreen(ttk.Frame):
    """Main evidence management screen."""
    
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.pack(fill=tk.BOTH, expand=True)
        
        self.papers = PAPERS
        self.sections = SECTIONS
        self.section_widgets: dict[str, CollapsibleSection] = {}
        self.chunk_counter = 0
        
        self._build_ui()
        self._load_mock_data()
    
    def _build_ui(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header, text="Evidence Manager", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        
        ttk.Button(header, text="+ Add Chunk", command=self._open_add_dialog).pack(side=tk.RIGHT)
        ttk.Button(header, text="Collapse All", command=self._collapse_all).pack(side=tk.RIGHT, padx=5)
        ttk.Button(header, text="Expand All", command=self._expand_all).pack(side=tk.RIGHT, padx=5)
        
        # Stats bar
        self.stats_label = ttk.Label(self, text="", font=("Segoe UI", 9), foreground="gray")
        self.stats_label.pack(fill=tk.X)
        
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Scrollable sections container
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self.sections_frame = ttk.Frame(canvas)
        
        self.sections_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sections_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Create collapsible sections
        for section in self.sections:
            widget = CollapsibleSection(self.sections_frame, section, on_delete_chunk=self._on_chunk_deleted)
            widget.pack(fill=tk.X, pady=2)
            self.section_widgets[section] = widget
        
        self._update_stats()
    
    def _load_mock_data(self):
        """Load some mock evidence chunks."""
        mock_chunks = [
            ("Introduction", PAPERS[0], "The Transformer architecture relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. This enables significantly more parallelization during training."),
            ("Introduction", PAPERS[2], "Large language models demonstrate emergent few-shot learning capabilities, performing tasks with minimal examples provided in-context."),
            ("Methods", PAPERS[0], "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."),
            ("Methods", PAPERS[3], "Chain-of-thought prompting enables complex reasoning by generating intermediate reasoning steps before the final answer."),
            ("Related Work", PAPERS[1], "BERT obtains state-of-the-art results on eleven NLP tasks by pre-training bidirectional representations through masked language modeling."),
            ("Results", PAPERS[2], "GPT-3 achieves 81.5% accuracy on the LAMBADA dataset, demonstrating strong performance on language understanding benchmarks."),
        ]
        
        for section, paper, summary in mock_chunks:
            self._add_chunk_to_section(section, paper, summary)
    
    def _add_chunk_to_section(self, section: str, paper: Paper, summary: str):
        self.chunk_counter += 1
        chunk = EvidenceChunk(
            id=f"chunk_{self.chunk_counter}",
            section=section,
            paper=paper,
            summary=summary,
            llm_score=0.75 + (self.chunk_counter % 3) * 0.08,
            vector_score=0.65 + (self.chunk_counter % 4) * 0.07
        )
        self.section_widgets[section].add_chunk(chunk)
        self._update_stats()
    
    def _open_add_dialog(self):
        AddChunkDialog(self.winfo_toplevel(), self.papers, self.sections, self._on_add_chunk)
    
    def _on_add_chunk(self, sections: List[str], paper: Paper, text: str):
        for section in sections:
            self._add_chunk_to_section(section, paper, text)
        messagebox.showinfo("Success", f"Added chunk to {len(sections)} section(s).")
    
    def _on_chunk_deleted(self, chunk: EvidenceChunk):
        self._update_stats()
    
    def _expand_all(self):
        for widget in self.section_widgets.values():
            if not widget.expanded:
                widget.toggle()
    
    def _collapse_all(self):
        for widget in self.section_widgets.values():
            if widget.expanded:
                widget.toggle()
    
    def _update_stats(self):
        total = sum(len(w.chunks) for w in self.section_widgets.values())
        papers_used = len(set(c.paper.id for w in self.section_widgets.values() for c in w.chunks))
        self.stats_label.config(text=f"Total: {total} chunks from {papers_used} papers")


def main():
    root = tk.Tk()
    root.title("Evidence Manager - Paper Writing Pipeline")
    root.geometry("800x650")
    
    style = ttk.Style()
    style.theme_use('clam')
    
    app = EvidenceManagerScreen(root)
    root.mainloop()


if __name__ == "__main__":
    main()