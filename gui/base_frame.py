from tkinter.ttk import Frame
import tkinter as tk
from tkinter import ttk
import os
import subprocess
import platform

from .icons import HoverColor
from .markdown_view import MarkdownView
from .theme_colors import (
    NAVBAR_BG_DARK, NAVBAR_BG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT,
    TEXT_FG_DARK, TEXT_FG_LIGHT,
    CARD_HEADER_BG_DARK, CARD_HEADER_BG_LIGHT,
    CARD_HEADER_FG_DARK, CARD_HEADER_FG_LIGHT,
)


MAX_WIDTH = 700
TEXT_AREA_SPACING = 4
TEXT_AREA_PADX = 10
TEXT_AREA_PADY = 10


def create_text_area(parent, height: int = 6, **kwargs) -> tk.Text:
    """Creates a styled multi-line text area widget."""
    text = tk.Text(
        parent,
        height=height,
        wrap="word",
        padx=TEXT_AREA_PADX,
        pady=TEXT_AREA_PADY,
        spacing2=TEXT_AREA_SPACING,
        spacing3=TEXT_AREA_SPACING,
        highlightthickness=0,
        borderwidth=0,
        relief="flat",
        **kwargs
    )
    return text


def create_scrollable_text_area(parent, height: int = 6, **kwargs) -> tuple[tk.Frame, tk.Text]:
    """
    Create a consistently styled multi-line text area widget with a vertical scrollbar.
    Returns (container_frame, text_widget).
    Caller must pack/grid the container_frame, NOT the text_widget.
    """
    # Container with border (simulated by background color + padding)
    # We use TextBorderFrame so app.py can identify and style it
    container = TextBorderFrame(parent, padx=1, pady=1)
    
    # Inner frame for contents (to hold text + scrollbar)
    inner = ttk.Frame(container)
    inner.pack(fill="both", expand=True)
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(inner, orient="vertical")
    scrollbar.pack(side="right", fill="y")
    
    # Text widget
    text = create_text_area(
        inner, 
        height=height, 
        yscrollcommand=scrollbar.set,
        **kwargs
    )
    text.pack(side="left", fill="both", expand=True)
    
    scrollbar.config(command=text.yview)
    
    return container, text


def create_gray_button(parent, text: str, command, **kwargs) -> ttk.Label:
    """Create a gray-styled clickable label (for trash/remove buttons)."""
    label = ttk.Label(parent, text=text, foreground="gray", cursor="hand2", **kwargs)
    label.bind("<Button-1>", lambda e: command())
    return label


class ProgressPopup(tk.Toplevel):
    """Simple modal progress popup."""
    
    def __init__(self, parent: tk.Tk, initial_status: str = "Processing"):
        super().__init__(parent)
        self.parent = parent
        self._is_error = False
        self._disabled_buttons = []
        
        # Disable all buttons in parent window
        self._disable_parent_buttons()
        
        # Basic window setup
        self.title("Processing...")
        self.transient(parent)
        self.resizable(False, False)
        
        self._popup_width = 450
        self._popup_height = 125
        self.minsize(self._popup_width, self._popup_height)
        self.geometry(f"{self._popup_width}x{self._popup_height}")
        
        self.content_frame = ttk.Frame(self, padding=(40, 30))
        self.content_frame.pack(fill="both", expand=True)
        self.content_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(
            self.content_frame, 
            text=initial_status, 
            font=self.parent.fonts.default_font,
            anchor="center",
            justify="center"
        )
        self.status_label.grid(row=0, column=0, pady=(0, 15), sticky="ew")
        
        # Spinner
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0
        self.spinner_label = ttk.Label(
            self.content_frame, 
            text=self._spinner_chars[0], 
            font=("", 24), 
            foreground="gray",
            anchor="center"
        )
        self.spinner_label.grid(row=1, column=0, sticky="ew")
        
        self.close_btn = ttk.Button(self.content_frame, text="Close", command=self.close)
        
        # Center on parent
        self._center_on_parent()
        
        # Make modal
        self.grab_set()
        self.focus_set()
        
        # Handle window close (X button) to ensure buttons are reenabled
        self.protocol("WM_DELETE_WINDOW", self.close)
        
        self._animate_spinner()
    
    def _disable_parent_buttons(self):
        """Find and disable all buttons in parent window."""
        self._disabled_buttons = []
        self._find_and_disable_buttons(self.parent)
    
    def _find_and_disable_buttons(self, widget):
        """Recursively find and disable all buttons."""
        for child in widget.winfo_children():
            if isinstance(child, (ttk.Button, tk.Button)):
                try:
                    # Only disable if currently enabled
                    if str(child.cget('state')) != 'disabled':
                        child.config(state='disabled')
                        self._disabled_buttons.append(child)
                except:
                    pass
            self._find_and_disable_buttons(child)
    
    def _enable_parent_buttons(self):
        """Re-enable previously disabled buttons."""
        for btn in self._disabled_buttons:
            try:
                btn.config(state='normal')
            except:
                pass
        self._disabled_buttons = []
    
    def _animate_spinner(self):
        if self._is_error or not self.winfo_exists():
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        self.spinner_label.config(text=self._spinner_chars[self._spinner_idx])
        self.after(80, self._animate_spinner)
    
    def _center_on_parent(self):
        """Center popup on parent window."""
        self.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - self._popup_width) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - self._popup_height) // 2
        self.geometry(f"+{x}+{y}")

    def update_status(self, status: str):
        """Update status text. Call from main thread via parent.after(0, ...)"""
        if self.winfo_exists() and not self._is_error:
            self.status_label.config(text=status)
    
    def show_error(self, error_message: str):
        """Show error with close button in a scrollable, copyable text widget."""
        if not self.winfo_exists():
            return
        self._is_error = True
        
        # Hide status and spinner labels
        self.status_label.pack_forget()
        self.spinner_label.pack_forget()
        
        # Clear existing content frame and rebuild for error
        self.content_frame.destroy()
        self.content_frame = ttk.Frame(self, padding=20)
        self.content_frame.pack(fill="both", expand=True)
        
        # Create error label
        error_label = ttk.Label(self.content_frame, text="Error:", font=self.parent.fonts.header_font, foreground="red")
        error_label.pack(anchor="w", pady=(0, 10))
        
        # Create scrollable text widget for error message
        text_frame = ttk.Frame(self.content_frame)
        text_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Text widget with scrollbar
        text_widget = tk.Text(
            text_frame,
            wrap="word",
            foreground="red",
            height=15,
            padx=10,
            pady=10,
            state="normal"
        )
        text_widget.pack(side="left", fill="both", expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert error message
        text_widget.insert("1.0", error_message)
        text_widget.config(state="disabled")  # Make read-only but still selectable/copyable
        
        # Make window resizable and larger
        self.resizable(True, True)
        self.geometry("700x500")
        
        # Create/update close button
        self.close_btn = ttk.Button(self.content_frame, text="Close", command=self.close)
        self.close_btn.pack(pady=(10, 0))
    
    def close(self):
        """Close the popup and re-enable buttons."""
        # Re-enable buttons first
        self._enable_parent_buttons()
        
        try:
            self.grab_release()
        except:
            pass
        try:
            self.destroy()
        except:
            pass


class InfoPopup(tk.Toplevel):
    """Simple info popup with styled header and close button."""
    
    def __init__(self, parent: tk.Tk, screen_title: str, content: str):
        super().__init__(parent)
        self.parent = parent
        
        # Window setup
        popup_title = f"About: {screen_title}"
        self.title(popup_title)
        self.transient(parent)
        self.resizable(True, True)
        self.minsize(500, 350)
        
        # Center on parent
        width, height = 800, 600
        self.geometry(f"{width}x{height}")
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - width) // 2
        y = parent.winfo_y() + (parent.winfo_height() - height) // 2
        self.geometry(f"+{x}+{y}")
        
        # Get theme colors
        is_dark = parent.current_theme == "dark"
        navbar_bg = getattr(parent, '_navbar_bg', NAVBAR_BG_DARK if is_dark else NAVBAR_BG_LIGHT)
        navbar_fg = TEXT_FG_DARK if is_dark else TEXT_FG_LIGHT
        
        # Main container
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(self, style="NavBar.TFrame", padding=(15, 10))
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Centered title
        tk.Label(
            header_frame,
            text=popup_title,
            font=parent.fonts.sub_header_font,
            bg=navbar_bg,
            fg=navbar_fg
        ).pack(expand=True)
        
        ttk.Separator(self, orient="horizontal").grid(row=0, column=0, sticky="sew")
        
        # Content area
        content_frame = ttk.Frame(self, style="Scrollable.TFrame")
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        md_label = MarkdownView(
            content_frame,
            font_manager=parent.fonts,
            theme_mode="dark" if is_dark else "light",
            padx=15,
            pady=15
        )
        md_label.grid(row=0, column=0, sticky="nsew")
        md_label.set_markdown(content)
        
        # Footer with close button
        ttk.Separator(self, orient="horizontal").grid(row=2, column=0, sticky="ew")
        
        footer_frame = ttk.Frame(self, style="NavBar.TFrame", padding=(15, 10))
        footer_frame.grid(row=3, column=0, sticky="ew")
        
        ttk.Button(footer_frame, text="Close", command=self.destroy).pack(side="right")
        
        self.grab_set()
        self.focus_set()


class BaseFrame(ttk.Frame):
    def __init__(self,
                 parent,
                 controller,
                 title="Screen",
                 has_next=True,
                 next_text="Next",
                 has_back=True,
                 back_text="Back",
                 has_regenerate=False,
                 regenerate_text="Regenerate",
                 header_file_path=None,
                 info_content=None):
        super().__init__(parent)
        self.controller = controller
        self.title = title
        self.has_next = has_next
        self.next_text = next_text
        self.has_back = has_back
        self.back_text = back_text
        self.has_regenerate = has_regenerate
        self.regenerate_text = regenerate_text
        self.header_file_path = header_file_path
        self.info_content = info_content
        
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=0)
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(self, style="NavBar.TFrame", padding=(10, 12))
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Info button on right side
        if self.info_content:

            info_btn = self.controller.icons.create_icon_label(
                header_frame,
                icon_name="info",
                command=self._show_info_popup,
                scale=1.75,
                hover_color=HoverColor.BLUE
            )
            info_btn.pack(side="right", padx=(10, 15))
        
        # Shared container for Title + Buttons to center them together
        center_container = ttk.Frame(header_frame, style="NavBar.TFrame")
        center_container.pack(expand=True)
        
        # Title
        navbar_bg = getattr(self.controller, '_navbar_bg', NAVBAR_BG_DARK)
        navbar_fg = TEXT_FG_DARK if self.controller.current_theme == "dark" else TEXT_FG_LIGHT
        tk.Label(
            center_container, 
            text=self.title, 
            font=self.controller.fonts.header_font,
            bg=navbar_bg,
            fg=navbar_fg
        ).pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, sticky="ew")
        
        # Content container
        content_container = ttk.Frame(self, style="Scrollable.TFrame")
        self.content_container = content_container # Save ref for updates
        content_container.grid(row=2, column=0, sticky="nsew")
        content_container.grid_columnconfigure(0, weight=1)
        
        # Calculate dynamic width based on font
        self.content_width = self.controller.fonts.measure_width(55)
        content_container.grid_columnconfigure(1, weight=0, minsize=self.content_width)
        
        content_container.grid_columnconfigure(2, weight=1)
        content_container.grid_rowconfigure(0, weight=1)
        
        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background") or "#1c1c1c"
        
        self._canvas = tk.Canvas(content_container, highlightthickness=0, bg=bg_color)
        self._canvas.grid(row=0, column=1, sticky="nsew")
        
        self.scrollable_frame = ttk.Frame(self._canvas, style="Scrollable.TFrame", padding=(10, 10, 10, 10))
        self._window_id = self._canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bindings
        self.scrollable_frame.bind("<Configure>", self._update_scrollregion)
        self._canvas.bind("<Configure>", self._update_window_width)
        
        # Mousewheel - bind globally with add="+" to not overwrite other frame bindings
        # Each frame's _on_mousewheel checks if mouse is over it
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")  # Windows/macOS
        self._canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")    # Linux scroll up
        self._canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")    # Linux scroll down
        
        # Action Buttons at top of scrollable content (if file path provided)
        if self.header_file_path:
            actions_frame = ttk.Frame(self.scrollable_frame, style="Scrollable.TFrame")
            actions_frame.pack(fill="x", pady=(10, 10))
            actions_frame.columnconfigure(0, weight=1, uniform="actions")
            actions_frame.columnconfigure(1, weight=1, uniform="actions")
            actions_frame.columnconfigure(2, weight=1, uniform="actions")
            
            # "Open in Editor" Button
            ttk.Button(
                actions_frame, 
                text="Open in Editor", 
                command=self._open_in_editor,
                style="Accent.TButton" 
            ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
            
            # "Show in Explorer" Button
            ttk.Button(
                actions_frame, 
                text="Show in Explorer", 
                command=self._show_in_explorer
            ).grid(row=0, column=1, sticky="ew", padx=5)

            # "Reload" File Button
            ttk.Button(
                actions_frame, 
                text="Reload", 
                command=self.reload_content
            ).grid(row=0, column=2, sticky="ew", padx=(5, 0))
        
        self.create_content()
        
        # Nav bar
        if self.has_next or self.has_back or self.has_regenerate:
            ttk.Separator(self, orient="horizontal").grid(row=3, column=0, sticky="ew")
            nav_container = ttk.Frame(self, style="NavBar.TFrame")
            self.nav_container = nav_container
            nav_container.grid(row=4, column=0, sticky="ew")
            nav_container.grid_columnconfigure(0, weight=1)
            nav_container.grid_columnconfigure(1, weight=0, minsize=self.content_width)
            nav_container.grid_columnconfigure(2, weight=1)
            
            nav_frame = ttk.Frame(nav_container, style="NavBar.TFrame", padding=(10, 12, 10, 12))
            nav_frame.grid(row=0, column=1, sticky="ew")
            
            style = ttk.Style()
            style.configure("Nav.TButton", font=self.controller.fonts.nav_button_font)
            
            if self.has_back:
                self.back_btn = ttk.Button(nav_frame, text=self.back_text, command=self.on_back, style="Nav.TButton")
                self.back_btn.pack(side="left")
            if self.has_next:
                self.next_btn = ttk.Button(nav_frame, text=self.next_text, command=self.on_next, style="Nav.TButton")
                self.next_btn.pack(side="right")
            if self.has_regenerate:
                self.regenerate_btn = ttk.Button(nav_frame, text=self.regenerate_text, command=self.on_regenerate, style="Nav.TButton")
                self.regenerate_btn.pack(side="right", padx=(0, 10))

        # Register font updates
        self.controller.fonts.add_callback(self._update_layout)

    def create_content(self):
        ttk.Label(self.scrollable_frame, text=f"Content for {self.title}").pack()

    def on_next(self):
        self.controller.next_screen()

    def on_back(self):
        self.controller.previous_screen()

    def on_regenerate(self):
        self.controller.next_screen()

    def on_show(self):
        """
        Called when the screen is shown. 
        Subclasses can override this to load data lazily or refresh dynamic content.
        """
        pass

    def set_next_text(self, text: str):
        """Update the text of the next button."""
        self.next_text = text
        if hasattr(self, 'next_btn'):
            self.next_btn.config(text=text)

    def set_regenerate_text(self, text: str):
        """Update the text of the regenerate button."""
        self.regenerate_text = text
        if hasattr(self, 'regenerate_btn'):
            self.regenerate_btn.config(text=text)

    def _update_layout(self):
        """Update layout when font size changes."""
        self.content_width = self.controller.fonts.measure_width(55)
        
        if hasattr(self, 'content_container'):
            self.content_container.grid_columnconfigure(1, minsize=self.content_width)
            
        if hasattr(self, 'nav_container'):
            self.nav_container.grid_columnconfigure(1, minsize=self.content_width)
            
        # Also update window width calculation
        self._canvas.event_generate("<Configure>")

    def _update_scrollregion(self, event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
    
    def _update_window_width(self, event):
        self._canvas.itemconfig(self._window_id, width=event.width)
    
    def _is_mouse_over_frame(self):
        """Check if mouse pointer is currently over this frame's content area."""
        try:
            # Get the widget under the mouse pointer
            x, y = self.winfo_pointerx(), self.winfo_pointery()
            widget = self.winfo_containing(x, y)
            
            # Go up the widget tree to see if we are inside this frame
            while widget:
                if widget is self:
                    return True
                if widget is self._canvas:
                    return True
                if widget is self.scrollable_frame:
                    return True
                widget = widget.master
        except:
            pass
        return False
    
    def _on_mousewheel(self, event):
        # Only scroll if mouse is over this frames content area
        if not self._is_mouse_over_frame():
            return
        
        # Dont scroll if content fits
        if self.scrollable_frame.winfo_reqheight() <= self._canvas.winfo_height():
            return
        
        # Dont hijack scroll from widgets that scroll themselves
        widget = event.widget
        if widget.winfo_class() in ("Listbox", "Text", "TCombobox", "Treeview"):
            return
        
        # Platform specific delta
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta > 0:
            delta = -1
        else:
            delta = 1
        
        self._canvas.yview_scroll(delta, "units")
    
    def _show_info_popup(self):
        """Show info popup with this screen's info_content."""
        if self.info_content:
            InfoPopup(self.controller, self.title, self.info_content)

    def _open_in_editor(self):
        """Open the header file in the default editor."""
        if not self.header_file_path or not os.path.exists(self.header_file_path):
            print(f"File not found: {self.header_file_path}")
            return
            
        print(f"Opening {self.header_file_path} in editor...")
        path = os.path.abspath(self.header_file_path)
        
        if platform.system() == 'Windows':
            os.startfile(path)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', path))
        else:
            subprocess.call(('xdg-open', path))

    def _show_in_explorer(self):
        """Reveal the header file in the file explorer."""
        if not self.header_file_path:
             return
             
        print(f"Showing {self.header_file_path} in explorer...")
        path = os.path.abspath(self.header_file_path)
        path = os.path.normpath(path)
        
        if platform.system() == 'Windows':
            subprocess.Popen(f'explorer /select,"{path}"')
        elif platform.system() == 'Darwin':
            subprocess.call(['open', '-R', path])
        else:
            # Linux - simple attempt to open folder
            subprocess.call(['xdg-open', os.path.dirname(path)])

    def show_error_message(self, title: str, message: str):
        """Display an error message for this screen in a copyable text widget.
        clears the scrollable frame first.
        """
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        error_frame = ttk.Frame(self.scrollable_frame)
        error_frame.pack(fill="x", pady=20, padx=20)
        
        # Error Title
        ttk.Label(
            error_frame, 
            text=f"{title}:", 
            foreground="red", 
            font=self.controller.fonts.sub_header_font
        ).pack(anchor="w")
        
        # Scrollable Text Container
        container = ttk.Frame(error_frame)
        container.pack(fill="x", expand=True)
        
        # Copyable Text Widget
        text_bg = TEXT_BG_DARK_ALT if self.controller.current_theme == "dark" else TEXT_BG_LIGHT_ALT
        error_text = tk.Text(
            container, 
            height=10, 
            wrap="word", 
            relief="flat", 
            font=self.controller.fonts.default_font,
            padx=10, 
            pady=10,
            bg=text_bg,
            highlightthickness=0
        )
        
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=error_text.yview)
        error_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        error_text.pack(side="left", fill="both", expand=True)

        error_text.insert("1.0", str(message))
        error_text.config(state="disabled", foreground="red")

    def reload_content(self):
        """
        Reload the content of the screen. 
        Default implementation clears the scrollable frame and calls create_content() and on_show().
        Subclasses can override this or ensure their create_content/on_show handles stateless re-loading.
        """
        print(f"Reloading screen: {self.title}")
        
        # Clear all widgets in scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Reset state attributes if they exist to force re-fetches
        common_attrs = [
            'draft_text', 
            'plan_text', 
            'context', 
            'hypotheses', 
            'current_hypothesis',
            '_results_loaded',
            'cards'
        ]
        for attr in common_attrs:
            if hasattr(self, attr):
                try:
                    if attr == 'cards':
                        # Reset cards to empty list
                        setattr(self, attr, [])
                    else:
                        delattr(self, attr)
                except:
                    setattr(self, attr, None)

        # Re-create action buttons at top of scrollable content (if file path provided)
        if self.header_file_path:
            actions_frame = ttk.Frame(self.scrollable_frame, style="Scrollable.TFrame")
            actions_frame.pack(fill="x", pady=(10, 10))
            actions_frame.columnconfigure(0, weight=1, uniform="actions")
            actions_frame.columnconfigure(1, weight=1, uniform="actions")
            actions_frame.columnconfigure(2, weight=1, uniform="actions")
            
            # "Open in Editor" Button
            ttk.Button(
                actions_frame, 
                text="Open in Editor", 
                command=self._open_in_editor,
                style="Accent.TButton" 
            ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
            
            # "Show in Explorer" Button
            ttk.Button(
                actions_frame, 
                text="Show in Explorer", 
                command=self._show_in_explorer
            ).grid(row=0, column=1, sticky="ew", padx=5)

            # "Reload" File Button
            ttk.Button(
                actions_frame, 
                text="Reload", 
                command=self.reload_content
            ).grid(row=0, column=2, sticky="ew", padx=(5, 0))

        # Re-create static content (info sections, etc.)
        self.create_content()
        
        # Trigger on_show to load dynamic content
        self.on_show()
        
        # Re-apply theme colors to created widgets (TextBorderFrame, Text etc.)
        self.controller.apply_theme_colors(self)

    def create_card_frame(self, parent, title, info_content=None):
        """Helper to create a unified card-styled section with title and separator.
        
        Args:
            parent: Parent widget
            title: Card title
            info_content: Optional info text to show in popup when info icon clicked
        """
        card = CardBorderFrame(parent, padx=1, pady=1)
        card.pack(fill="x", padx=0, pady=10)
        
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        header_bg = getattr(self.controller, '_card_header_bg', CARD_HEADER_BG_DARK)
        header_fg = CARD_HEADER_FG_DARK if self.controller.current_theme == "dark" else CARD_HEADER_FG_LIGHT
        tk.Label(
            header, 
            text=title, 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        # Add info icon if info_content provided
        if info_content:
            info_btn = self.controller.icons.create_icon_label(
                header,
                icon_name="info",
                command=lambda: InfoPopup(self.controller, title, info_content),
                scale=1.5,
                hover_color=HoverColor.BLUE
            )
            info_btn.pack(side="right", padx=(10, 0))
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        content = ttk.Frame(card, style="CardContent.TFrame", padding=10)
        content.pack(fill="x")
        return content

class TextBorderFrame(tk.Frame):
    """Custom Frame used as a border container for Text widgets."""
    pass


class CardBorderFrame(tk.Frame):
    """Custom Frame used as a border container for Card widgets.
    Uses tk.Frame (not ttk) so background color can be set directly.
    """
    pass
