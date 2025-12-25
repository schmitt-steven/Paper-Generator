from tkinter.ttk import Style


import tkinter as tk
from tkinter import ttk, messagebox
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
    EvidenceScreen,
    PaperDraftScreen,
    ResultScreen,
    SectionGuidelinesScreen
)
from .fonts import FontManager
from .icons import IconManager
from settings import Settings
from utils.lm_studio_client import is_lm_studio_running

class PaperGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Check if LM Studio is running before initializing the app
        if not is_lm_studio_running():
            self.withdraw()  # Hide the main window
            messagebox.showwarning(
                "LM Studio Not Running",
                "LM Studio must be running in the background.\n\n"
                "Please start LM Studio and try again."
            )
            self.destroy()
            return


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
        
        # Apply Sun Valley theme (use saved preference)
        saved_dark_mode = getattr(Settings, "DARK_MODE", True)
        self.current_theme = "dark" if saved_dark_mode else "light"
        sv_ttk.set_theme(self.current_theme)

        # Global font config
        self.fonts = FontManager(self, base_size=getattr(Settings, "FONT_SIZE_BASE", 16))
        
        # Icon manager for theme-aware icons
        self.icons = IconManager(self)
        
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
            EvidenceScreen,
            PaperDraftScreen,
            ResultScreen,
            SectionGuidelinesScreen
        ]
        self.current_screen_index = 0
        
        self.init_frames()
        
        # Defer application of custom theme colors to override defaults
        self.after(100, self.apply_theme_colors)
        
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
        
        # Card header styling (different background for the title row)
        if self.current_theme == "dark":
            self._card_header_bg = "#252525"  # Slightly lighter than card background
            self._navbar_bg = "#1a1a1a"       # Match text area background
        else:
            self._card_header_bg = "#f0f0f0"  # Slightly darker than card background
            self._navbar_bg = "#f1f1f1"       # Light gray for navbar in light mode
        
        style.configure("CardHeader.TFrame", background=self._card_header_bg)
        style.configure("CardHeader.TLabel", background=self._card_header_bg)
        style.configure("NavBar.TFrame", background=self._navbar_bg)
        style.configure("NavBar.TLabel", background=self._navbar_bg)
        
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
        
        # Theme-aware Combobox Listbox styling (store as instance vars for update_combobox_styles)
        if self.current_theme == "dark":
            self._listbox_bg = "#2b2b2b"
            self._listbox_fg = "#ffffff"
            self._listbox_select_bg = "#404040"
            self._listbox_select_fg = "#ffffff"
            self._listbox_border = "#404040"
            popdown_bg = "#2b2b2b"
        else:
            self._listbox_bg = "#ffffff"
            self._listbox_fg = "#1c1c1c"
            self._listbox_select_bg = "#0078d4"
            self._listbox_select_fg = "#ffffff"
            self._listbox_border = "#cccccc"
            popdown_bg = "#ffffff"
        
        style.configure("ComboboxPopdownFrame", relief="flat", background=popdown_bg)
        # For listbox inside combobox
        self.option_add("*TCombobox*Listbox*Font", self.fonts.default_font)
        self.option_add("*TCombobox*Listbox*Background", self._listbox_bg)
        self.option_add("*TCombobox*Listbox*Foreground", self._listbox_fg)
        self.option_add("*TCombobox*Listbox*selectBackground", self._listbox_select_bg)
        self.option_add("*TCombobox*Listbox*selectForeground", self._listbox_select_fg)
        self.option_add("*TCombobox*Listbox*relief", "solid")
        self.option_add("*TCombobox*Listbox*borderWidth", 1)
        self.option_add("*TCombobox*Listbox*highlightThickness", 1)
        self.option_add("*TCombobox*Listbox*highlightBackground", self._listbox_border)
        self.option_add("*TCombobox*Listbox*highlightColor", self._listbox_border)

    def update_combobox_styles(self, widget=None):
        """Recursively update all Combobox dropdown listbox styles for theme changes."""
        if widget is None:
            widget = self
        
        # Check if this is a Combobox
        if isinstance(widget, ttk.Combobox):
            try:
                # Get the popdown listbox and configure it directly
                # The listbox is accessed via the popdown toplevel
                popdown = widget.tk.call("ttk::combobox::PopdownWindow", widget)
                listbox = popdown + ".f.l"
                widget.tk.call(listbox, "configure",
                    "-background", self._listbox_bg,
                    "-foreground", self._listbox_fg,
                    "-selectbackground", self._listbox_select_bg,
                    "-selectforeground", self._listbox_select_fg,
                    "-relief", "solid",
                    "-borderwidth", 1,
                    "-highlightthickness", 1,
                    "-highlightbackground", self._listbox_border,
                    "-highlightcolor", self._listbox_border,
                    "-font", self.fonts.default_font
                )
            except:
                pass  # Popdown may not exist yet if dropdown hasn't been opened
        
        # Recurse through children
        for child in widget.winfo_children():
            self.update_combobox_styles(child)

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
        # Apply theme colors to newly shown frame (handles lazy-loaded widgets)
        self.apply_theme_colors(frame)

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


    def toggle_theme(self):
        if self.current_theme == 'dark':
            self.current_theme = 'light'
        else:
            self.current_theme = 'dark'
        
        sv_ttk.set_theme(self.current_theme)
        # Re-configure styles to ensure consistency
        self.configure_styles()
        # Update existing Combobox dropdown listbox styles
        self.update_combobox_styles()
        # Update icons for new theme
        self.icons._clear_cache()
        self.icons.update_icon_labels()
        self.apply_theme_colors()

    def apply_theme_colors(self, widget=None):
        """Recursively apply theme colors to TextBorderFrame and Text widgets."""
        if widget is None:
            widget = self
            
        # Define colors
        if self.current_theme == "dark":
            text_bg = "#1a1a1a"     # Dark Gray (lighter than pure black)
            text_fg = "#ffffff"
            border_color = "#2A2A2A" # Subtle dark border
            insert_bg = "#ffffff"
            card_header_bg = "#252525"
            card_header_fg = "#ffffff"
            canvas_bg = "#1c1c1c"
        else:
            text_bg = "#ffffff"     # Pure White
            text_fg = "#1c1c1c"
            border_color = "#cccccc" # Light Gray
            insert_bg = "#1c1c1c"
            card_header_bg = "#f0f0f0"
            card_header_fg = "#1c1c1c"
            canvas_bg = "#fafafa"
            
        # Import wrapper class here
        from .base_frame import TextBorderFrame
        
        # Apply to BorderFrame (The container)
        if isinstance(widget, TextBorderFrame):
            try:
                widget.configure(background=border_color)
            except:
                pass
        
        # Apply to Canvas (scrollable area background)
        if isinstance(widget, tk.Canvas):
            try:
                widget.configure(background=canvas_bg)
            except:
                pass
                
        # Apply to Text (The content)
        if isinstance(widget, tk.Text):
            try:
                widget.configure(
                    background=text_bg,
                    foreground=text_fg,
                    insertbackground=insert_bg,
                    highlightthickness=0,
                    relief="flat"
                )
            except:
                pass
        
        # Apply to tk.Label inside CardHeader.TFrame or NavBar.TFrame
        if isinstance(widget, tk.Label):
            try:
                parent = widget.master
                grandparent = parent.master if parent else None
                
                # Check if parent is using CardHeader.TFrame or NavBar.TFrame style
                # Also check if grandparent is CardHeader (for nested tk.Frame containers)
                is_card_header = False
                is_navbar = False
                
                if isinstance(parent, ttk.Frame):
                    style = str(parent.cget('style'))
                    is_card_header = 'CardHeader' in style
                    is_navbar = 'NavBar' in style
                elif isinstance(parent, tk.Frame) and isinstance(grandparent, ttk.Frame):
                    # Label inside tk.Frame inside CardHeader.TFrame
                    style = str(grandparent.cget('style'))
                    is_card_header = 'CardHeader' in style
                    # Also update the tk.Frame's background
                    parent.configure(background=card_header_bg)
                
                if is_card_header:
                    # Update background, but preserve gray foreground for count labels
                    current_fg = str(widget.cget('fg'))
                    if current_fg in ['gray', '#888888', '#666666']:
                        # This is a count label - keep gray foreground
                        widget.configure(background=card_header_bg, fg="#666666")
                    else:
                        # This is a title label - use theme foreground
                        widget.configure(background=card_header_bg, foreground=card_header_fg)
                elif is_navbar:
                    # NavBar uses _navbar_bg color (set in configure_styles)
                    navbar_bg = self._navbar_bg
                    navbar_fg = "#ffffff" if self.current_theme == "dark" else "#1c1c1c"
                    widget.configure(background=navbar_bg, foreground=navbar_fg)
            except:
                pass
                
        # Recurse
        for child in widget.winfo_children():
            self.apply_theme_colors(child)


if __name__ == "__main__":
    app: PaperGeneratorApp = PaperGeneratorApp()
    app.mainloop()

