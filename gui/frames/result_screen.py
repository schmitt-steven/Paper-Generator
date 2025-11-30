import tkinter as tk
from tkinter import ttk
from ..base_frame import BaseFrame

class ResultScreen(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, title="Result", has_next=False)

    def create_content(self):
        ttk.Label(self.content_frame, text="Final Paper Generation Result").pack(pady=20)
        # Placeholder for result display
