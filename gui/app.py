import tkinter as tk
from tkinter import ttk
import sv_ttk
import sys
import os
from .frames import (
    SettingsScreen,
    UserRequirementsScreen,
    CodeFilesScreen,
    PaperConceptScreen,
    PaperSelectionScreen,
    HypothesisScreen,
    ExperimentationPlanScreen,
    ExperimentResultsScreen,
    PaperDraftScreen,
    ResultScreen
)

class PaperGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()


        # Windows DPI awareness (fixes blurry text on Windows)
        if sys.platform == "win32":
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(2)
        
        # macOS/Linux: Scaling variables for high-DPI displays
        MACOS_SCALING = 1.0
        LINUX_SCALING = 1.0
        
        if sys.platform == "darwin":  # macOS
            try:
                current_scaling = float(self.tk.call('tk', 'scaling'))
                if MACOS_SCALING != 1.0:
                    self.tk.call('tk', 'scaling', '-displayof', '.', MACOS_SCALING)
            except:
                pass
        elif sys.platform.startswith('linux'):
            try:
                current_scaling = float(self.tk.call('tk', 'scaling'))
                if LINUX_SCALING != 1.0:
                    self.tk.call('tk', 'scaling', '-displayof', '.', LINUX_SCALING)
            except:
                pass
        
        # Apply Sun Valley theme
        sv_ttk.set_theme("dark")
        style = ttk.Style()
        style.layout("TButton", style.layout("Accent.TButton"))
        style.configure("TButton", **style.configure("Accent.TButton"))
        style.map("TButton", **style.map("Accent.TButton"))
        
        self.title("Paper Generator")
        self.geometry("800x600")
        
        # Container that holds all the frames
        self.container = ttk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        self.screen_order = [
            SettingsScreen,
            UserRequirementsScreen,
            CodeFilesScreen,
            PaperConceptScreen,
            PaperSelectionScreen,
            HypothesisScreen,
            ExperimentationPlanScreen,
            ExperimentResultsScreen,
            PaperDraftScreen,
            ResultScreen
        ]
        self.current_screen_index = 0
        
        self.init_frames()
        self.show_frame(self.screen_order[0])

    def init_frames(self):
        for Frame in self.screen_order:
            frame = Frame(parent=self.container, controller=self)
            self.frames[Frame] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def next_screen(self):
        self.current_screen_index += 1
        if self.current_screen_index < len(self.screen_order):
            next_class = self.screen_order[self.current_screen_index]
            self.show_frame(next_class)
        else:
            self.destroy()

    def previous_screen(self):
        self.current_screen_index -= 1
        if self.current_screen_index >= 0:
            previous_class = self.screen_order[self.current_screen_index]
            self.show_frame(previous_class)

if __name__ == "__main__":
    app: PaperGeneratorApp = PaperGeneratorApp()
    app.mainloop()
