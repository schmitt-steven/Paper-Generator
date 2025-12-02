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

class BaseFrame(ttk.Frame):
    def __init__(self, parent, controller, title="Screen", has_next=True, next_text="Next", has_back=True, back_text="Back", has_skip=False, skip_text="Skip"):
        super().__init__(parent)
        self.controller = controller
        self.title = title
        self.has_next = has_next
        self.next_text = next_text
        self.has_back = has_back
        self.back_text = back_text
        self.has_skip = has_skip
        self.skip_text = skip_text
        
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
        if self.has_next or self.has_back or self.has_skip:
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
            if self.has_skip:
                ttk.Button(nav_frame, text=self.skip_text, command=self.on_skip).pack(side="right")

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

    def on_skip(self):
        self.controller.next_screen()

