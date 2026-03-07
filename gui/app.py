from tkinter.ttk import Style
import tkinter as tk
from tkinter import ttk, messagebox
import sv_ttk
import sys
import os
from .theme_colors import (
    BORDER_DARK, BORDER_LIGHT,
    SEPARATOR_DARK, SEPARATOR_LIGHT,
    CARD_HEADER_BG_DARK, CARD_HEADER_BG_LIGHT,
    NAVBAR_BG_DARK, NAVBAR_BG_LIGHT,
    CARD_BG_DARK, CARD_BG_LIGHT,
    CANVAS_BG_DARK, CANVAS_BG_LIGHT,
    LISTBOX_BG_DARK, LISTBOX_BG_LIGHT,
    LISTBOX_FG_DARK, LISTBOX_FG_LIGHT,
    LISTBOX_SELECT_BG_DARK, LISTBOX_SELECT_BG_LIGHT,
    LISTBOX_SELECT_FG_DARK, LISTBOX_SELECT_FG_LIGHT,
    POPDOWN_BG_DARK, POPDOWN_BG_LIGHT,
    TEXT_BG_DARK, TEXT_BG_LIGHT,
    TEXT_FG_DARK, TEXT_FG_LIGHT,
    INSERT_BG_DARK, INSERT_BG_LIGHT,
    CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
)
from .frames import (
    StartScreen,
    SettingsScreen,
    ResearchContextScreen,
    LiteratureSearchScreen,
    HypothesisScreen,
    ExperimentPlanScreen,
    ExperimentResultsScreen,
    PaperDraftScreen,
    ResultScreen,
    WritingPromptsScreen
)
from .fonts import FontManager
from .icons import IconManager
from settings import Settings
from utils.lm_studio_client import is_lm_studio_running

class PaperGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        if not is_lm_studio_running():
            self.withdraw()  # Hide main window
            messagebox.showwarning(
                "LM Studio Not Running",
                "LM Studio must be running in the background.\n\n"
                "Please start LM Studio and open the app again."
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
        
        # Apply Sun Valley tkinter theme 
        saved_dark_mode = getattr(Settings, "DARK_MODE", True)
        self.current_theme = "dark" if saved_dark_mode else "light"
        sv_ttk.set_theme(self.current_theme)

        # Global font config
        self.fonts = FontManager(self, base_size=Settings.FONT_SIZE.value)
        
        # Icon manager for theme-aware icons
        self.icons = IconManager(self)
        
        # Configure styles initially
        self.configure_styles()
        
        # Register callback to re-configure styles when fonts change
        # This ensures widgets like Combobox update their internal font references
        self.fonts.add_callback(self.configure_styles)
        self.fonts.add_callback(self.icons.update_icon_labels)

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
            StartScreen,
            ResearchContextScreen,
            LiteratureSearchScreen,
            HypothesisScreen,
            ExperimentPlanScreen,
            ExperimentResultsScreen,
            PaperDraftScreen,
            ResultScreen
        ]
        self.current_screen_index = 0
        
        self.init_frames()
        
        # Defer app of custom theme colors to override defaults
        self.after(100, self.apply_theme_colors)
        
        # Disable scrolling on Comboboxes to prevent accidental changes
        self.unbind_class("TCombobox", "<MouseWheel>")
        self.unbind_class("TCombobox", "<Button-4>")
        self.unbind_class("TCombobox", "<Button-5>")
        
        # Show initial frame and call on_show
        initial_frame = self.frames[self.screen_order[0]]
        initial_frame.tkraise()
        if hasattr(initial_frame, 'on_show'):
            initial_frame.on_show()

    def configure_styles(self):
        """Configure ttk styles with current fonts."""
        style = ttk.Style()
        
        # Configure default fonts for ttk widgets
        style.configure("TLabel", font=self.fonts.default_font)
        style.configure("TButton", font=self.fonts.default_font)
        style.configure("TEntry", font=self.fonts.text_field_font)
        style.configure("TFrame", font=self.fonts.default_font)
        style.configure("TLabelframe.Label", font=self.fonts.default_font)
        
        # Override button styling
        try:
             style.layout("TButton", style.layout("Accent.TButton"))
             style.configure("TButton", font=self.fonts.default_font, **style.configure("Accent.TButton"))
             style.map("TButton", **style.map("Accent.TButton"))
        except:
             pass
        
        # Danger button style (red)
        style.configure("Danger.TButton", 
                        font=self.fonts.default_font,
                        background="#dc3545",
                        foreground="white")
        style.map("Danger.TButton",
                  background=[("active", "#c82333"), ("pressed", "#bd2130")],
                  foreground=[("active", "white"), ("pressed", "white")])

        # Card header styling 
        if self.current_theme == "dark":
            self._card_header_bg = CARD_HEADER_BG_DARK
            self._navbar_bg = NAVBAR_BG_DARK
        else:
            self._card_header_bg = CARD_HEADER_BG_LIGHT
            self._navbar_bg = NAVBAR_BG_LIGHT
        
        style.configure("CardHeader.TFrame", background=self._card_header_bg)
        style.configure("CardHeader.TLabel", background=self._card_header_bg)
        style.configure("NavBar.TFrame", background=self._navbar_bg)
        style.configure("NavBar.TLabel", background=self._navbar_bg)
        
        # Card frame border color (padding=1 trick uses card background as border)
        if self.current_theme == "dark":
            self._card_border = BORDER_DARK
            self._card_content_bg = CARD_BG_DARK if CARD_BG_DARK else "#1c1c1c"
        else:
            self._card_border = BORDER_LIGHT
            self._card_content_bg = CARD_BG_LIGHT
        style.configure("Card.TFrame", background=self._card_border)
        style.configure("CardContent.TFrame", background=self._card_content_bg)
        style.configure("CardRow.TFrame", background=self._card_content_bg)
        style.configure("CardRow.TLabel", background=self._card_content_bg)
        style.configure("CardRow.TCheckbutton", background=self._card_content_bg)
        style.configure("CardRow.Switch.TCheckbutton", background=self._card_content_bg)
        
        # Canvas/scrollable area background
        if self.current_theme == "dark":
            self._canvas_bg = CANVAS_BG_DARK
        else:
            self._canvas_bg = CANVAS_BG_LIGHT
        
        style.configure("Scrollable.TFrame", background=self._canvas_bg)
        style.configure("Scrollable.TLabel", background=self._canvas_bg)
        
        # Separator styling
        separator_color = SEPARATOR_DARK if self.current_theme == "dark" else SEPARATOR_LIGHT
        style.configure("TSeparator", background=separator_color)
        
        # Custom Listbox (Dropdown Menu) styling
        style.configure("TCombobox", font=self.fonts.default_font)
        style.configure("TSpinbox", font=self.fonts.text_field_font)
        style.configure("TEntry", font=self.fonts.text_field_font)
        
        # Set font for TCombobox sub-elements via option_add
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
        
        # Combobox Listbox styling (stored as instance vars for update_combobox_styles)
        if self.current_theme == "dark":
            self._listbox_bg = LISTBOX_BG_DARK
            self._listbox_fg = LISTBOX_FG_DARK
            self._listbox_select_bg = LISTBOX_SELECT_BG_DARK
            self._listbox_select_fg = LISTBOX_SELECT_FG_DARK
            self._listbox_border = BORDER_DARK
            popdown_bg = POPDOWN_BG_DARK
        else:
            self._listbox_bg = LISTBOX_BG_LIGHT
            self._listbox_fg = LISTBOX_FG_LIGHT
            self._listbox_select_bg = LISTBOX_SELECT_BG_LIGHT
            self._listbox_select_fg = LISTBOX_SELECT_FG_LIGHT
            self._listbox_border = BORDER_LIGHT
            popdown_bg = POPDOWN_BG_LIGHT
        
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
                # Get popdown listbox and configure it directly
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
                pass

        for child in widget.winfo_children():
            self.update_combobox_styles(child)

    def init_frames(self):
        for Frame in self.screen_order:
            frame = Frame(parent=self.container, controller=self)
            self.frames[Frame] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Init additional frames not in main navigation
        extra_frames = [SettingsScreen, WritingPromptsScreen]
        for Frame in extra_frames:
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
        
        # Broadcast theme change event for widgets that handle it themselves (e.g. MarkdownView)
        self.event_generate("<<ThemeChanged>>")

    def apply_theme_colors(self, widget=None):
        """Recursively apply theme colors to TextBorderFrame and Text widgets."""
        if widget is None:
            widget = self
            
        # Define colors
        if self.current_theme == "dark":
            text_bg = TEXT_BG_DARK
            text_fg = TEXT_FG_DARK
            border_color = BORDER_DARK
            insert_bg = INSERT_BG_DARK
            card_header_bg = CARD_HEADER_BG_DARK
            card_header_fg = CARD_HEADER_FG_DARK
        else:
            text_bg = TEXT_BG_LIGHT
            text_fg = TEXT_FG_LIGHT
            border_color = BORDER_LIGHT
            insert_bg = INSERT_BG_LIGHT
            card_header_bg = CARD_HEADER_BG_LIGHT
            card_header_fg = CARD_HEADER_FG_LIGHT
            
        # Import wrapper classes
        from .base_frame import TextBorderFrame, CardBorderFrame
        
        # Apply to TextBorderFrame (the container for text areas)
        if isinstance(widget, TextBorderFrame):
            try:
                widget.configure(background=border_color)
            except:
                pass
        
        # Apply to CardBorderFrame (the container for cards)
        if isinstance(widget, CardBorderFrame):
            try:
                widget.configure(background=border_color)
            except:
                pass
        
        # Apply to Canvas (scrollable area background)
        if isinstance(widget, tk.Canvas):
            try:
                widget.configure(background=self._canvas_bg)
            except:
                pass
                
        # Apply to Text (the content)
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
                    # Update background
                    current_fg = str(widget.cget('fg'))
                    if current_fg in ['gray', '#888888', '#666666']:
                        widget.configure(background=card_header_bg, fg="#666666")
                    else:
                        widget.configure(background=card_header_bg, foreground=card_header_fg)
                elif is_navbar:
                    navbar_bg = self._navbar_bg
                    navbar_fg = "#ffffff" if self.current_theme == "dark" else "#1c1c1c"
                    widget.configure(background=navbar_bg, foreground=navbar_fg)
            except:
                pass
                
        for child in widget.winfo_children():
            self.apply_theme_colors(child)


if __name__ == "__main__":
    app: PaperGeneratorApp = PaperGeneratorApp()
    app.mainloop()
