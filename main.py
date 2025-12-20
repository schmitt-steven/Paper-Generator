"""
Entry point for Paper Generator application.
This script is used by PyInstaller to properly handle imports.
"""
import sys
import os

# Add the project root to path for proper imports
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    application_path = os.path.dirname(sys.executable)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, application_path)

# Now import and run the app
from gui.app import PaperGeneratorApp

if __name__ == "__main__":
    app = PaperGeneratorApp()
    app.mainloop()
