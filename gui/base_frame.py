from tkinter.ttk import Frame
import tkinter as tk
from tkinter import ttk
import os
import subprocess
import platform

MAX_WIDTH = 700

# Text widget styling constants

# Text widget styling constants
# Text widget styling constants
TEXT_AREA_SPACING = 4
TEXT_AREA_PADX = 10
TEXT_AREA_PADY = 10


class TextBorderFrame(tk.Frame):
    """Custom Frame used as a border container for Text widgets."""
    pass


def create_text_area(parent, height: int = 6, **kwargs) -> tk.Text:
    """Create a consistently styled multi-line text area widget."""
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
        self.transient(parent)  # Stay on top of parent, minimize together
        self.resizable(False, False)
        
        # Set minimum size for the popup
        self.minsize(200, 100)
        
        # Content
        self.content_frame = ttk.Frame(self, padding=40)
        self.content_frame.pack(fill="both", expand=True)
        
        self.status_label = ttk.Label(self.content_frame, text=initial_status, font=self.parent.fonts.default_font)
        self.status_label.pack(pady=(0, 15))
        
        self.dots_label = ttk.Label(self.content_frame, text="", font=self.parent.fonts.default_font, foreground="gray")
        self.dots_label.pack()
        
        self.close_btn = ttk.Button(self.content_frame, text="Close", command=self.close)
        # Only shown on error
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
        # Make modal
        self.grab_set()
        self.focus_set()
        
        # Handle window close (X button) to ensure buttons are re-enabled
        self.protocol("WM_DELETE_WINDOW", self.close)
        
        # Animate dots
        self._dots_count = 0
        self._animate_dots()
    
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
            # Recurse into child widgets
            self._find_and_disable_buttons(child)
    
    def _enable_parent_buttons(self):
        """Re-enable previously disabled buttons."""
        for btn in self._disabled_buttons:
            try:
                btn.config(state='normal')
            except:
                pass
        self._disabled_buttons = []
    
    def _animate_dots(self):
        if self._is_error or not self.winfo_exists():
            return
        self._dots_count = (self._dots_count + 1) % 4
        self.dots_label.config(text="." * self._dots_count)
        self.after(500, self._animate_dots)
    
    def update_status(self, status: str):
        """Update status text. Call from main thread via parent.after(0, ...)"""
        if self.winfo_exists() and not self._is_error:
            self.status_label.config(text=status)
    
    def show_error(self, error_message: str):
        """Show error with close button in a scrollable, copyable text widget."""
        if not self.winfo_exists():
            return
        self._is_error = True
        
        # Hide status and dots labels
        self.status_label.pack_forget()
        self.dots_label.pack_forget()
        
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
        
        # Make window resizable and larger for error display
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
        
        # Center on parent - wider and taller
        width, height = 700, 500
        self.geometry(f"{width}x{height}")
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - width) // 2
        y = parent.winfo_y() + (parent.winfo_height() - height) // 2
        self.geometry(f"+{x}+{y}")
        
        # Get theme colors
        is_dark = parent.current_theme == "dark"
        header_bg = getattr(parent, '_card_header_bg', '#252525' if is_dark else '#f0f0f0')
        header_fg = "#ffffff" if is_dark else "#1c1c1c"
        
        # Main container
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(self, style="CardHeader.TFrame", padding=(15, 10))
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Centered title
        tk.Label(
            header_frame,
            text=popup_title,
            font=parent.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(expand=True)
        
        ttk.Separator(self, orient="horizontal").grid(row=0, column=0, sticky="sew")
        
        # Content area
        content_frame = ttk.Frame(self, padding=15)
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Text widget for content
        text = tk.Text(
            content_frame,
            wrap="word",
            font=parent.fonts.default_font,
            padx=10,
            pady=10,
            relief="flat",
            highlightthickness=0
        )
        text.grid(row=0, column=0, sticky="nsew")
        text.insert("1.0", content)
        text.config(state="disabled")
        
        # Apply theme colors to text
        if is_dark:
            text.configure(background="#1a1a1a", foreground="#ffffff")
        else:
            text.configure(background="#ffffff", foreground="#1c1c1c")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.config(yscrollcommand=scrollbar.set)
        
        # Footer with close button
        ttk.Separator(self, orient="horizontal").grid(row=2, column=0, sticky="ew")
        
        footer_frame = ttk.Frame(self, style="CardHeader.TFrame", padding=(15, 10))
        footer_frame.grid(row=3, column=0, sticky="ew")
        
        ttk.Button(footer_frame, text="Close", command=self.destroy).pack(side="right")
        
        # Grab focus
        self.grab_set()
        self.focus_set()


class BaseFrame(ttk.Frame):
    def __init__(self, parent, controller, title="Screen", has_next=True, next_text="Next", has_back=True, back_text="Back", has_regenerate=False, regenerate_text="Regenerate", header_file_path=None, info_content=None):
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
        
        # Info button on right side (pack before center container)
        if self.info_content:
            # Info icon is larger than other icons
            info_size = int(self.controller.fonts.base_size * 1.75)
            info_btn = self.controller.icons.create_icon_label(
                header_frame,
                icon_name="info",
                command=self._show_info_popup,
                size=info_size
            )
            info_btn.pack(side="right", padx=(10, 15))
        
        # Shared container for Title + Buttons to center them together
        center_container = ttk.Frame(header_frame, style="NavBar.TFrame")
        center_container.pack(expand=True)
        
        # Title - use tk.Label for reliable background color
        navbar_bg = getattr(self.controller, '_navbar_bg', '#1a1a1a')
        navbar_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        tk.Label(
            center_container, 
            text=self.title, 
            font=self.controller.fonts.header_font,
            bg=navbar_bg,
            fg=navbar_fg
        ).pack(side="left")
        
        # Action Buttons
        if self.header_file_path:
            actions_frame = ttk.Frame(center_container, style="NavBar.TFrame")
            actions_frame.pack(side="left", padx=(15, 0))
            
            # "Open in Editor" Button
            ttk.Button(
                actions_frame, 
                text="Open in Editor", 
                command=self._open_in_editor,
                style="Accent.TButton" 
            ).pack(side="left", padx=5)
            
            # "Show in Explorer" Button
            ttk.Button(
                actions_frame, 
                text="Show in Explorer", 
                command=self._show_in_explorer
            ).pack(side="left", padx=5)

            # "Reload" File Button
            ttk.Button(
                actions_frame, 
                text="Reload", 
                command=self.reload_content
            ).pack(side="left", padx=5)

        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        # Content container
        content_container = ttk.Frame(self)
        self.content_container = content_container # Save ref for updates
        content_container.grid(row=2, column=0, sticky="nsew")
        content_container.grid_columnconfigure(0, weight=1)
        
        # Calculate dynamic width based on font
        self.content_width = self.controller.fonts.measure_width(55) # Approx 55 chars wide
        content_container.grid_columnconfigure(1, weight=0, minsize=self.content_width)
        
        content_container.grid_columnconfigure(2, weight=1)
        content_container.grid_rowconfigure(0, weight=1)
        
        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background") or "#1c1c1c"
        
        self._canvas = tk.Canvas(content_container, highlightthickness=0, bg=bg_color)
        self._canvas.grid(row=0, column=1, sticky="nsew")
        
        self.scrollable_frame = ttk.Frame(self._canvas, padding=(10, 10, 10, 10))
        self._window_id = self._canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bindings
        self.scrollable_frame.bind("<Configure>", self._update_scrollregion)
        self._canvas.bind("<Configure>", self._update_window_width)
        
        # Mousewheel - bind when mouse enters this frame, unbind when leaves
        self.bind("<Enter>", self._bind_mousewheel)
        self.bind("<Leave>", self._unbind_mousewheel)
        
        self.create_content()
        
        # Nav bar
        if self.has_next or self.has_back or self.has_regenerate:
            ttk.Separator(self, orient="horizontal").grid(row=3, column=0, sticky="ew")
            nav_container = ttk.Frame(self, style="NavBar.TFrame")
            self.nav_container = nav_container # Save ref
            nav_container.grid(row=4, column=0, sticky="ew")
            nav_container.grid_columnconfigure(0, weight=1)
            nav_container.grid_columnconfigure(1, weight=0, minsize=self.content_width)
            nav_container.grid_columnconfigure(2, weight=1)
            
            nav_frame = ttk.Frame(nav_container, style="NavBar.TFrame", padding=(10, 12, 10, 12))
            nav_frame.grid(row=0, column=1, sticky="ew")
            
            style = ttk.Style()
            style.configure("Nav.TButton", font=self.controller.fonts.nav_button_font)
            
            if self.has_back:
                ttk.Button(nav_frame, text=self.back_text, command=self.on_back, style="Nav.TButton").pack(side="left")
            if self.has_next:
                ttk.Button(nav_frame, text=self.next_text, command=self.on_next, style="Nav.TButton").pack(side="right")
            if self.has_regenerate:
                ttk.Button(nav_frame, text=self.regenerate_text, command=self.on_regenerate, style="Nav.TButton").pack(side="right", padx=(0, 10))

        # Register for font updates
        self.controller.fonts.add_callback(self._update_layout)

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
    
    def _bind_mousewheel(self, event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self._canvas.bind_all("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self._canvas.bind_all("<Button-5>", self._on_mousewheel)    # Linux scroll down
    
    def _unbind_mousewheel(self, event):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        # Don't scroll if content fits
        if self.scrollable_frame.winfo_reqheight() <= self._canvas.winfo_height():
            return
        
        # Don't hijack scroll from widgets that scroll themselves
        widget = event.widget
        if widget.winfo_class() in ("Listbox", "Text", "TCombobox", "Treeview", "Canvas"):
            return
        
        # Platform-specific delta
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta > 0:
            delta = -1
        else:
            delta = 1
        
        self._canvas.yview_scroll(delta, "units")

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
            
        # Reset common state attributes if they exist to force re-fetches
        # This is a heuristic to help subclasses that use 'if getattr(self, x) is None' checks
        common_attrs = [
            'draft_text', 
            'plan_text', 
            'concept', 
            'hypotheses', 
            'current_hypothesis',
            '_results_loaded'
        ]
        for attr in common_attrs:
            if hasattr(self, attr):
                # We can't always delattr if it's defined in __init__, so setting to None is safer
                # But some checks use hasattr. Let's try deleting them if they are dynamic.
                try:
                    delattr(self, attr)
                except:
                    setattr(self, attr, None)

        # Re-create static content (info sections, etc.)
        self.create_content()
        
        # Trigger on_show to load dynamic content
        self.on_show()

    def create_card_frame(self, parent, title):
        """Helper to create a unified card-styled section with title and separator."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=1)
        card.pack(fill="x", padx=0, pady=10)
        
        header = ttk.Frame(card, style="CardHeader.TFrame", padding=(10, 6))
        header.pack(fill="x")
        
        # Use tk.Label instead of ttk.Label for reliable background color
        header_bg = getattr(self.controller, '_card_header_bg', '#252525')
        header_fg = "#ffffff" if self.controller.current_theme == "dark" else "#1c1c1c"
        tk.Label(
            header, 
            text=title, 
            font=self.controller.fonts.sub_header_font,
            bg=header_bg,
            fg=header_fg
        ).pack(side="left")
        
        ttk.Separator(card, orient="horizontal").pack(fill="x")
        
        content = ttk.Frame(card, padding=10)
        content.pack(fill="x")
        return content

