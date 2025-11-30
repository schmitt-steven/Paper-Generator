import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame

class ExperimentationPlanScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, title="Experimentation Plan", next_text="Run Experiments")

    def create_content(self):
        ttk.Label(self.content_frame, text="Design the experiments to test the hypothesis.").pack(pady=20)
        # Placeholder for experiment plan
