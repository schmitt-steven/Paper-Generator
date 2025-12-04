#!/usr/bin/env python3
"""Minimal test script for ProgressPopup component."""

import tkinter as tk
from tkinter import ttk
import threading
import time
from pathlib import Path

# Add the gui directory to the path so we can import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from gui.base_frame import ProgressPopup


class TestWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Popup Test")
        self.root.geometry("800x600")
        
        # Create a simple frame with test buttons
        frame = ttk.Frame(root, padding="20")
        frame.pack(fill="both", expand=True)
        
        ttk.Label(
            frame,
            text="ProgressPopup Test",
            font=("SF Pro", 16, "bold")
        ).pack(pady=10)
        
        ttk.Label(
            frame,
            text="Click a button to test the popup:",
            font=("SF Pro", 12)
        ).pack(pady=10)
        
        # Test success path
        ttk.Button(
            frame,
            text="Test Success Path",
            command=self.test_success
        ).pack(pady=5)
        
        # Test error path
        ttk.Button(
            frame,
            text="Test Error Path",
            command=self.test_error
        ).pack(pady=5)
        
        # Test long-running task
        ttk.Button(
            frame,
            text="Test Long Task (5 steps)",
            command=self.test_long_task
        ).pack(pady=5)
    
    def test_success(self):
        """Test popup with successful completion."""
        popup = ProgressPopup(self.root, "Testing Success...")
        
        def task():
            try:
                self.root.after(0, lambda: popup.update_status("Step 1: Initializing..."))
                time.sleep(1)
                
                self.root.after(0, lambda: popup.update_status("Step 2: Processing data..."))
                time.sleep(1)
                
                self.root.after(0, lambda: popup.update_status("Step 3: Finalizing..."))
                time.sleep(0.5)
                
                # Success - close popup
                self.root.after(0, lambda: self._on_success(popup))
            except Exception as e:
                self.root.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def test_error(self):
        """Test popup with error."""
        popup = ProgressPopup(self.root, "Testing Error...")
        
        def task():
            try:
                self.root.after(0, lambda: popup.update_status("Step 1: Starting..."))
                time.sleep(1)
                
                self.root.after(0, lambda: popup.update_status("Step 2: About to fail..."))
                time.sleep(1)
                
                # Simulate an error
                raise ValueError("This is a test error message!")
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def test_long_task(self):
        """Test popup with a longer multi-step task."""
        popup = ProgressPopup(self.root, "Testing Long Task...")
        
        def task():
            try:
                steps = [
                    "Step 1: Loading configuration...",
                    "Step 2: Analyzing code files...",
                    "Step 3: Building paper concept...",
                    "Step 4: Searching literature...",
                    "Step 5: Generating hypotheses...",
                ]
                
                for i, step_msg in enumerate(steps, 1):
                    self.root.after(0, lambda msg=step_msg: popup.update_status(msg))
                    time.sleep(1.5)
                
                # Success
                self.root.after(0, lambda: self._on_success(popup))
            except Exception as e:
                self.root.after(0, lambda err=str(e): popup.show_error(err))
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def _on_success(self, popup: ProgressPopup):
        """Handle successful completion."""
        popup.close()
        print("Task completed successfully!")


def main():
    root = tk.Tk()
    app = TestWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

