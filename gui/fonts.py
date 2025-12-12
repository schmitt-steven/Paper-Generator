import tkinter.font as tkfont
import sys

class FontManager:
    def __init__(self, root, base_size=16):
        self.root = root
        self.base_size = base_size
        
        # OS Detection and Configuration
        if sys.platform == "win32":
            self.font_family = "Bahnschrift"
            self.size_offset = -3 # Windows fonts tend to be larger at same point size ?? or just preference
        elif sys.platform == "darwin":
            self.font_family = "SF Pro"
            self.size_offset = 0
        else:
            self.font_family = "Helvetica" # Fallback for Linux
            self.size_offset = 0
            
        # Monospace font
        if sys.platform == "win32":
            self.mono_family = "Consolas"
        elif sys.platform == "darwin":
            self.mono_family = "Menlo"
        else:
            self.mono_family = "Courier New"

        self.fonts = {}
        self.callbacks = []
        self._init_fonts()

    def add_callback(self, callback):
        """Add a callback to be invoked when font size changes."""
        self.callbacks.append(callback)

    def _init_fonts(self):
        """Initialize NamedFonts."""
        # We hold references to these NamedFonts so they don't get garbage collected
        # and so we can update them later.
        
        # Default Font (Labels, Buttons, etc.)
        self.default_font = tkfont.Font(
            root=self.root, 
            name="AppDefaultFont", 
            family=self.font_family, 
            size=self._calc_size(0)
        )
        
        # Header Font (Screen Titles)
        self.header_font = tkfont.Font(
            root=self.root, 
            name="AppHeaderFont", 
            family=self.font_family, 
            size=self._calc_size(6), 
            weight="bold"
        )
        
        # Sub Header Font (Section Headers)
        self.sub_header_font = tkfont.Font(
            root=self.root, 
            name="AppSubHeaderFont", 
            family=self.font_family, 
            size=self._calc_size(2), 
            weight="bold"
        )
        
        # Text Area Font (Main Content)
        self.text_area_font = tkfont.Font(
            root=self.root, 
            name="AppTextAreaFont", 
            family=self.font_family, 
            size=self._calc_size(-2)
        )
        
        # Text Field Font (Entry widgets)
        self.text_field_font = tkfont.Font(
            root=self.root, 
            name="AppTextFieldFont", 
            family=self.font_family, 
            size=self._calc_size(0) # Same as default
        )

        # Nav Button Font
        self.nav_button_font = tkfont.Font(
            root=self.root, 
            name="AppNavButtonFont", 
            family=self.font_family, 
            size=self._calc_size(0)
        )

        # Code Font (Monospace)
        self.code_font = tkfont.Font(
            root=self.root,
            name="AppCodeFont",
            family=self.mono_family,
            size=self._calc_size(-4)
        )

        # Small Font (Captions, secondary text)
        self.small_font = tkfont.Font(
            root=self.root,
            name="AppSmallFont",
            family=self.font_family,
            size=self._calc_size(-4)
        )
        
    def _calc_size(self, relative_size):
        """Calculate actual font size based on base size, offset, and relative diff."""
        # Ensure minimum size of 8 to remain readable
        return max(8, self.base_size + self.size_offset + relative_size)

    def update_base_size(self, new_size):
        """Update the base size and refresh all registered fonts."""
        self.base_size = int(new_size)
        
        self.default_font.configure(size=self._calc_size(0))
        self.header_font.configure(size=self._calc_size(6))
        self.sub_header_font.configure(size=self._calc_size(2))
        self.text_area_font.configure(size=self._calc_size(-2))
        self.text_field_font.configure(size=self._calc_size(-6))
        self.text_field_font.configure(size=self._calc_size(-6))
        self.nav_button_font.configure(size=self._calc_size(0))
        self.code_font.configure(size=self._calc_size(-4))
        self.small_font.configure(size=self._calc_size(-4))
        
        for callback in self.callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in font change callback: {e}")

    def measure_width(self, num_chars: int) -> int:
        """Measure the width of N characters in the default font."""
        return self.default_font.measure("M") * num_chars
