import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame


class PaperDraftScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(
            parent=parent,
            controller=controller,
            title="Paper Draft Review",
            next_text="Continue"
        )

    def create_content(self):
        ttk.Label(
            self.scrollable_frame,
            text="Paper Draft Review",
            font=("SF Pro", 14)
        ).pack(pady=20)
        # Placeholder for paper draft display

