from tkinter.ttk import Frame


import tkinter as tk
from tkinter import ttk

MAX_WIDTH = 700

# Text widget styling constants

TEXT_AREA_SPACING = 4  # Line spacing (spacing between lines)
TEXT_AREA_PADX = 8
TEXT_AREA_PADY = 8


def create_text_area(parent, height: int = 6, **kwargs) -> tk.Text:
    """Create a consistently styled multi-line text area widget."""
    text = tk.Text(
        parent,
        height=height,
        wrap="word",
        padx=TEXT_AREA_PADX,
        pady=TEXT_AREA_PADY,
        spacing2=TEXT_AREA_SPACING,  # spacing between wrapped lines
        spacing3=TEXT_AREA_SPACING,  # spacing between paragraphs
        highlightthickness=0,
        borderwidth=0,
        relief="flat",
        **kwargs
    )
    return text


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


class BaseFrame(ttk.Frame):
    def __init__(self, parent, controller, title="Screen", has_next=True, next_text="Next", has_back=True, back_text="Back", has_regenerate=False, regenerate_text="Regenerate"):
        super().__init__(parent)
        self.controller = controller
        self.title = title
        self.has_next = has_next
        self.next_text = next_text
        self.has_back = has_back
        self.back_text = back_text
        self.has_regenerate = has_regenerate
        self.regenerate_text = regenerate_text
        
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=0)
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(self, padding=(10, 20))
        header_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(header_frame, text=self.title, font=self.controller.fonts.header_font).pack()
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
            nav_container = ttk.Frame(self)
            self.nav_container = nav_container # Save ref
            nav_container.grid(row=4, column=0, sticky="ew")
            nav_container.grid_columnconfigure(0, weight=1)
            nav_container.grid_columnconfigure(1, weight=0, minsize=self.content_width)
            nav_container.grid_columnconfigure(2, weight=1)
            
            nav_frame = ttk.Frame(nav_container, padding=(10, 20, 10, 20))
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

