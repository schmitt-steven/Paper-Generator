import tkinter as tk
from tkinter import ttk

class BaseFrame(ttk.Frame):
    def __init__(self, parent, controller, title="Screen", has_next=True, next_text="Next", has_back=True, back_text="Back"):
        super().__init__(parent)
        self.controller = controller
        self.title = title
        self.has_next = has_next
        self.next_text = next_text
        self.has_back = has_back
        self.back_text = back_text
        
        # Main layout configuration
        self.grid_rowconfigure(0, weight=1) # Content area
        self.grid_rowconfigure(1, weight=0) # Navigation bar
        self.grid_columnconfigure(0, weight=1)
        
        # Header / Title
        # Using a standard ttk.Frame for header. 
        # To add background colors with ttk, we would need to define styles.
        # For now, we stick to the native look which is cleaner.
        header_frame = ttk.Frame(self, padding="10")
        header_frame.grid(row=0, column=0, sticky="new")
        ttk.Label(header_frame, text=self.title, font=("Helvetica", 16, "bold")).pack()
        
        # Content Area (Placeholder)
        self.content_frame = ttk.Frame(self, padding="20")
        self.content_frame.grid(row=0, column=0, sticky="nsew", pady=(50, 0)) # Offset for header
        
        self.create_content()
        
        # Navigation Bar
        if self.has_next or self.has_back:
            nav_frame = ttk.Frame(self, padding="10")
            nav_frame.grid(row=1, column=0, sticky="ew")
            
            # Next Button
            if self.has_next:
                next_btn = ttk.Button(nav_frame, text=self.next_text, command=self.on_next)
                next_btn.pack(side="right")
            
            # Back Button
            if self.has_back:
                back_btn = ttk.Button(nav_frame, text=self.back_text, command=self.on_back)
                back_btn.pack(side="left")

    def create_content(self):
        """Override this method to add specific content for the screen."""
        ttk.Label(self.content_frame, text=f"Content for {self.title}").pack(expand=True)

    def on_next(self):
        """Default action for next button. Override if validation is needed."""
        self.controller.next_screen()

    def on_back(self):
        """Default action for back button. Override if validation is needed."""
        self.controller.previous_screen()

