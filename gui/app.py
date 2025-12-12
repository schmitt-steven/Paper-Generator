from tkinter.ttk import Style


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
from .fonts import FontManager
from settings import Settings

class PaperGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()


        # Windows DPI awareness (fixes blurry text on Windows)
        if sys.platform == "win32":
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(2)
        
        # macOS/Linux: Scaling vars for high-DPI displays
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

        # Global font config
        self.fonts = FontManager(self, base_size=getattr(Settings, "FONT_SIZE_BASE", 16))
        
        # Configure styles initially
        self.configure_styles()
        
        # Register callback to re-configure styles when fonts change
        # This ensures widgets like Combobox update their internal font references
        self.fonts.add_callback(self.configure_styles)

        self.title("Paper Generator")
        
        # Start with window maximized
        if sys.platform == "win32":
            self.state('zoomed')
        elif sys.platform == "darwin":
            self.update_idletasks()
            try:
                self.wm_attributes('-zoomed', True)
            except:
                width = self.winfo_screenwidth()
                height = self.winfo_screenheight()
                self.geometry(f"{width}x{height}")
        else:
            self.update_idletasks()
            width = self.winfo_screenwidth()
            height = self.winfo_screenheight()
            self.geometry(f"{width}x{height}")
        
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
        # Show initial frame and call on_show
        initial_frame = self.frames[self.screen_order[0]]
        initial_frame.tkraise()
        if hasattr(initial_frame, 'on_show'):
            initial_frame.on_show()

    def configure_styles(self):
        """Configure ttk styles with current fonts."""
        style = ttk.Style()
        
        # Configure default fonts for common ttk widgets
        style.configure("TLabel", font=self.fonts.default_font)
        style.configure("TButton", font=self.fonts.default_font)
        style.configure("TEntry", font=self.fonts.text_field_font)
        style.configure("TFrame", font=self.fonts.default_font)
        style.configure("TLabelframe.Label", font=self.fonts.default_font)
        
        # Override button styling (after setting default)
        # Note: We need to be careful not to reset layout if it's already set
        # But configuring font is safe
        try:
             style.layout("TButton", style.layout("Accent.TButton"))
             style.configure("TButton", font=self.fonts.default_font, **style.configure("Accent.TButton"))
             style.map("TButton", **style.map("Accent.TButton"))
        except:
             pass # Theme might not be fully loaded or compatible
        
        # Custom Listbox (Dropdown Menu) styling
        style.configure("TCombobox", font=self.fonts.default_font)
        style.configure("TSpinbox", font=self.fonts.text_field_font)
        style.configure("TEntry", font=self.fonts.text_field_font)
        
        # Explicitly set font for TCombobox sub-elements via option_add
        # This is often needed for the Entry part of the Combobox to pick up changes
        self.option_add("*TCombobox*Font", self.fonts.default_font)
        self.option_add("*TCombobox.Font", self.fonts.default_font)
        
        # Ensure Entry and Spinbox widgets also use the correct font
        self.option_add("*Entry.Font", self.fonts.text_field_font)
        self.option_add("*Spinbox.Font", self.fonts.text_field_font)
        # Add T-prefix variants for ttk widgets
        self.option_add("*TEntry.Font", self.fonts.text_field_font)
        self.option_add("*TSpinbox.Font", self.fonts.text_field_font)
        
        # Set default font for Text widgets (text areas)
        self.option_add("*Text.Font", self.fonts.text_area_font)
        style.configure("ComboboxPopdownFrame", relief="flat", background="#2b2b2b")
        # For listbox inside combobox, we try to set it but it's tricky with ttk
        self.option_add("*TCombobox*Listbox*Font", self.fonts.default_font)
        self.option_add("*TCombobox*Listbox*Background", "#2b2b2b")
        self.option_add("*TCombobox*Listbox*Foreground", "#ffffff")
        self.option_add("*TCombobox*Listbox*selectBackground", "#404040")
        self.option_add("*TCombobox*Listbox*selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox*relief", "flat")
        self.option_add("*TCombobox*Listbox*borderWidth", 5)
        self.option_add("*TCombobox*Listbox*highlightThickness", 0)

    def init_frames(self):
        for Frame in self.screen_order:
            frame = Frame(parent=self.container, controller=self)
            self.frames[Frame] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        # Call on_show if the frame has this method (for lazy loading)
        if hasattr(frame, 'on_show'):
            frame.on_show()

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
