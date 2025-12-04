from tkinter.ttk import Frame


import tkinter as tk
from tkinter import ttk

MAX_WIDTH = 700

# Text widget styling constants
TEXT_FONT = ("SF Pro", 14)
TEXT_SPACING = 4  # Line spacing (spacing between lines)
TEXT_PADX = 8
TEXT_PADY = 8


def create_styled_text(parent, height: int = 6, **kwargs) -> tk.Text:
    """Create a consistently styled Text widget."""
    text = tk.Text(
        parent,
        height=height,
        wrap="word",
        font=TEXT_FONT,
        padx=TEXT_PADX,
        pady=TEXT_PADY,
        spacing2=TEXT_SPACING,  # spacing between wrapped lines
        spacing3=TEXT_SPACING,  # spacing between paragraphs
        **kwargs
    )
    return text


def create_gray_button(parent, text: str, command, **kwargs) -> ttk.Label:
    """Create a gray-styled clickable label (for trash/remove buttons)."""
    label = ttk.Label(parent, text=text, foreground="gray", cursor="hand2", **kwargs)
    label.bind("<Button-1>", lambda e: command())
    return label


class ProgressPopup(tk.Toplevel):
    """
    Modal progress popup with status updates and error handling.
    
    Usage:
        popup = ProgressPopup(parent, "Generating...")
        # In background thread, use parent.after() for thread-safe updates:
        parent.after(0, lambda: popup.update_status("Processing..."))
        # On success:
        parent.after(0, popup.close)
        # On error:
        parent.after(0, lambda: popup.show_error("Something went wrong"))
    """
    
    def __init__(self, parent: tk.Tk, initial_status: str = "Processing..."):
        super().__init__(parent)
        self.parent = parent
        self._is_error = False
        self._is_closing = False
        
        # Remove window decorations and make it float above
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        
        # Semi-transparent overlay background
        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background") or "#1c1c1c"
        self.configure(bg=bg_color)
        
        # Make the toplevel cover the entire parent window
        self._update_geometry()
        # Bind to parent configure events to keep popup aligned
        parent.bind("<Configure>", self._on_parent_configure, add="+")
        
        # Create overlay frame (blocks clicks)
        overlay = tk.Frame(self, bg=bg_color)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Center dialog box
        dialog_frame = ttk.Frame(self, padding=30)
        dialog_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Status label
        self.status_label = ttk.Label(
            dialog_frame,
            text=initial_status,
            font=("SF Pro", 16),
            justify="center"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Loading indicator (simple dots animation)
        self.dots_label = ttk.Label(
            dialog_frame,
            text="",
            font=("SF Pro", 14),
            foreground="gray"
        )
        self.dots_label.pack()
        self._dots_count = 0
        self._animate_dots()
        
        # Error button (hidden initially)
        self.close_btn = ttk.Button(
            dialog_frame,
            text="Close",
            command=self.close
        )
        # Button is packed only when showing error
        
        # Make modal - grab all input
        self.grab_set()
        self.focus_set()
    
    def _update_geometry(self):
        """Update popup to cover parent window."""
        if not self.winfo_exists() or self._is_closing:
            return
        
        try:
            # Wait for parent window to be fully mapped before getting geometry
            self.parent.update_idletasks()
            
            x = self.parent.winfo_rootx()
            y = self.parent.winfo_rooty()
            w = self.parent.winfo_width()
            h = self.parent.winfo_height()
            
            # Only update if we have valid dimensions
            if w > 0 and h > 0:
                self.geometry(f"{w}x{h}+{x}+{y}")
                # Ensure popup stays on top
                self.attributes("-topmost", True)
                # Make sure it's still visible
                self.lift()
        except (tk.TclError, AttributeError):
            # Parent window might be destroyed or not yet mapped
            pass
    
    def _on_parent_configure(self, event):
        """Keep popup aligned with parent when parent moves/resizes."""
        # Only update if popup still exists and isn't closing
        if not self._is_closing and self.winfo_exists():
            try:
                # Use after_idle to ensure geometry update happens after the configure event
                self.after_idle(self._update_geometry)
            except (tk.TclError, AttributeError):
                # Popup might be destroyed during the update
                pass
    
    def _animate_dots(self):
        """Animate loading dots."""
        if self._is_error or not self.winfo_exists():
            return
        self._dots_count = (self._dots_count + 1) % 4
        self.dots_label.config(text="." * self._dots_count)
        self.after(500, self._animate_dots)
    
    def update_status(self, status: str):
        """Update the status text. Must be called from main thread (use after())."""
        if self.winfo_exists() and not self._is_error:
            self.status_label.config(text=status)
    
    def show_error(self, error_message: str):
        """Show error state with close button. Must be called from main thread."""
        if not self.winfo_exists():
            return
        self._is_error = True
        self.status_label.config(text=f"Error: {error_message}", foreground="red")
        self.dots_label.pack_forget()
        self.close_btn.pack(pady=(15, 0))
    
    def close(self):
        """Safe cleanup - always release grab before destroy."""
        # Mark as closing to prevent configure events from trying to update
        self._is_closing = True
        
        # Note: We don't unbind the configure event because Tkinter doesn't support
        # unbinding a specific handler. Instead, _on_parent_configure checks _is_closing
        # and does nothing if we're closing. This is safe because:
        # 1. The flag prevents updates during cleanup
        # 2. Once destroyed, winfo_exists() will return False anyway
        
        # Release grab safely
        try:
            self.grab_release()
        except:
            pass
        
        # Destroy the popup
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
        header_frame = ttk.Frame(self, padding="10")
        header_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(header_frame, text=self.title, font=("SF Pro", 22, "bold")).pack()
        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        # Content container
        content_container = ttk.Frame(self)
        content_container.grid(row=2, column=0, sticky="nsew")
        content_container.grid_columnconfigure(0, weight=1)
        content_container.grid_columnconfigure(1, weight=0, minsize=MAX_WIDTH)
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
            nav_container.grid(row=4, column=0, sticky="ew")
            nav_container.grid_columnconfigure(0, weight=1)
            nav_container.grid_columnconfigure(1, weight=0, minsize=MAX_WIDTH)
            nav_container.grid_columnconfigure(2, weight=1)
            
            nav_frame = ttk.Frame(nav_container, padding=(10, 10, 10, 10))
            nav_frame.grid(row=0, column=1, sticky="ew")
            
            if self.has_back:
                ttk.Button(nav_frame, text=self.back_text, command=self.on_back).pack(side="left")
            if self.has_next:
                ttk.Button(nav_frame, text=self.next_text, command=self.on_next).pack(side="right")
            if self.has_regenerate:
                ttk.Button(nav_frame, text=self.regenerate_text, command=self.on_regenerate).pack(side="right", padx=(0, 10))

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

