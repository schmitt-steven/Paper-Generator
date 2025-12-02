import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame


class ExperimentResultsScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(
            parent=parent,
            controller=controller,
            title="Experiment Results",
            next_text="Write Paper"
        )

    def create_content(self):
        ttk.Label(
            self.scrollable_frame,
            text="Experiment Results",
            font=("SF Pro", 14)
        ).pack(pady=20)
        # Placeholder for experiment results display

