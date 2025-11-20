import json
import platform
import os
from pathlib import Path
from typing import Optional

class LMSJITSettings:
    """
    Context manager to temporarily disable LM Studio's 'Only Keep Last JIT Loaded Model' setting if enabled.
    This allows multiple models (e.g., LLM + Embedding) to be loaded at same time.
    """
    
    def __init__(self):
        self.settings_path = self._get_settings_path()
        self.original_value: Optional[bool] = None
        self.modified = False

    def _get_settings_path(self) -> Path:
        system = platform.system()
        home = Path.home()
        
        if system == "Darwin":  # macOS
            return home / "Library" / "Application Support" / "LM Studio" / "settings.json"
        elif system == "Windows":
            return home / "AppData" / "Roaming" / "LM Studio" / "settings.json"
        else:  # Linux and others
            return home / ".cache" / "lm-studio" / "settings.json"

    def _read_settings(self) -> dict:
        if not self.settings_path.exists():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _write_settings(self, data: dict):
        if not self.settings_path.exists():
            return
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[LMSJITSettings] Failed to write settings: {e}")

    def __enter__(self):
        if not self.settings_path.exists():
            print(f"[LMSJITSettings] Settings file not found at {self.settings_path}")
            return self

        data = self._read_settings()
        
        # Navigate to developer -> unloadPreviousJITModelOnLoad
        dev_settings = data.get("developer", {})
        
        # Store original value (default to True if not present)
        self.original_value = dev_settings.get("unloadPreviousJITModelOnLoad", True)
        
        # Only modify if it's currently True (enabled)
        if self.original_value is True:
            print("[LMSJITSettings] Temporarily disabling 'Only Keep Last JIT Loaded Model'...")
            if "developer" not in data:
                data["developer"] = {}
            
            data["developer"]["unloadPreviousJITModelOnLoad"] = False
            self._write_settings(data)
            self.modified = True
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.modified and self.original_value is not None:
            print("[LMSJITSettings] Restoring original JIT model loading setting...")
            data = self._read_settings()
            if "developer" not in data:
                data["developer"] = {}
            
            data["developer"]["unloadPreviousJITModelOnLoad"] = self.original_value
            self._write_settings(data)
