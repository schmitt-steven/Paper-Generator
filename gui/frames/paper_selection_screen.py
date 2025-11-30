import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame

class PaperSelectionScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, title="Paper Selection", next_text="Confirm Selection")

    def create_content(self):
        ttk.Label(self.content_frame, text="Select relevant papers from search results.").pack(pady=20)
        # Placeholder for list of papers
