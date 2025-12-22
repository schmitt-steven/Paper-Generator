# Theme-aware icon management using Pillow
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
from typing import Optional


class IconManager:
    """
    Manages theme-aware icons using Pillow.
    Icons are stored as template PNGs (white on transparent) and colorized at runtime.
    """
    
    # Multiplier to convert font size to icon size
    SIZE_MULTIPLIER = 1.25
    
    def __init__(self, app):
        self.app = app
        self.icons_dir = Path(__file__).parent / "icons"
        self._cache: dict[tuple[str, str, int], ImageTk.PhotoImage] = {}
        
        # Register for font/theme changes
        self.app.fonts.add_callback(self._clear_cache)
    
    @property
    def default_size(self) -> int:
        """Calculate icon size based on current font size."""
        base_size = self.app.fonts.base_size
        return int(base_size * self.SIZE_MULTIPLIER)
    
    def _clear_cache(self):
        """Clear cached icons when theme/font changes."""
        self._cache.clear()
    
    def get_icon(self, name: str, size: int = None) -> 'ImageTk.PhotoImage | None':
        """
        Get a theme-aware icon by name.
        
        Args:
            name: Icon name (without extension), e.g., 'x', 'upload'
            size: Icon size in pixels (defaults to font-scaled size)
            
        Returns:
            PhotoImage ready for use in tkinter widgets
        """
        # Use font-scaled default if size not specified
        if size is None:
            size = self.default_size
            
        # Determine color based on current theme
        color = "#ffffff" if self.app.current_theme == "dark" else "#1c1c1c"
        
        # Check cache
        cache_key = (name, color, size)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load and colorize icon
        icon_path = self.icons_dir / f"{name}.png"
        if not icon_path.exists():
            print(f"[Icons] Warning: Icon not found: {icon_path}")
            return None
        
        try:
            # Load the template image (should be white on transparent)
            img = Image.open(icon_path).convert("RGBA")
            
            # Resize if needed
            if img.size != (size, size):
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # Colorize: replace white pixels with target color
            img = self._colorize(img, color)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self._cache[cache_key] = photo
            return photo
            
        except Exception as e:
            print(f"[Icons] Error loading icon {name}: {e}")
            return None
    
    def _colorize(self, img: Image.Image, color: str) -> Image.Image:
        """
        Colorize a template image (white on transparent) to the target color.
        """
        # Parse hex color
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        # Get pixel data
        data = img.getdata()
        new_data = []
        
        for pixel in data:
            # Keep original alpha, replace RGB with target color
            # Use the original pixel's luminance to determine the new color intensity
            if len(pixel) == 4:
                orig_r, orig_g, orig_b, a = pixel
                # Use max of RGB as intensity factor
                intensity = max(orig_r, orig_g, orig_b) / 255.0
                new_data.append((
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity),
                    a
                ))
            else:
                new_data.append(pixel)
        
        img.putdata(new_data)
        return img
    
    def create_icon_label(self, parent, icon_name: str, command=None, 
                          size: int = None) -> tk.Label:
        """
        Create a clickable label with an icon.
        
        Args:
            parent: Parent widget
            icon_name: Name of the icon
            command: Callback function when clicked
            size: Icon size in pixels (defaults to font-scaled size)
            
        Returns:
            tk.Label configured as an icon button
        """
        icon = self.get_icon(icon_name, size)
        
        # Determine background based on parent style
        bg = self._get_parent_bg(parent)
        
        label = tk.Label(parent, image=icon, bg=bg, cursor="hand2")
        label._icon_name = icon_name  # Store for theme updates
        label._icon_size = size
        label._icon_ref = icon  # Keep reference to prevent garbage collection
        
        if command:
            label.bind("<Button-1>", lambda e: command())
        
        return label
    
    def _get_parent_bg(self, parent) -> str:
        """Get the background color of the parent widget, walking up hierarchy if needed."""
        # Walk up the parent hierarchy to find a styled frame
        widget = parent
        while widget is not None:
            try:
                if isinstance(widget, ttk.Frame):
                    style = str(widget.cget('style'))
                    if 'NavBar' in style:
                        return getattr(self.app, '_navbar_bg', self._default_bg())
                    elif 'CardHeader' in style:
                        return getattr(self.app, '_card_header_bg', self._default_bg())
            except:
                pass
            
            # Move to parent widget
            try:
                widget = widget.master
            except:
                break
        
        # Fallback: try to get background from original parent
        try:
            if hasattr(parent, 'cget'):
                return parent.cget('background')
        except:
            pass
        
        return self._default_bg()
    
    def _default_bg(self) -> str:
        """Get the default background color from the ttk theme."""
        try:
            style = ttk.Style()
            bg = style.lookup("TFrame", "background")
            if bg:
                return bg
        except:
            pass
        # Fallback
        if self.app.current_theme == "dark":
            return "#1c1c1c"
        else:
            return "#fafafa"
    
    def update_icon_labels(self, widget=None):
        """
        Recursively update all icon labels to match current theme.
        Call this after theme changes.
        """
        if widget is None:
            widget = self.app
        
        # Check if this is an icon label we created
        if isinstance(widget, tk.Label) and hasattr(widget, '_icon_name'):
            icon_name = widget._icon_name
            size = getattr(widget, '_icon_size', None) or self.default_size
            new_icon = self.get_icon(icon_name, size)
            if new_icon:
                widget.configure(image=new_icon)
                widget._icon_ref = new_icon  # Update reference
                
                # Update background too
                bg = self._get_parent_bg(widget.master)
                try:
                    widget.configure(bg=bg)
                except:
                    pass
        
        # Recurse
        for child in widget.winfo_children():
            self.update_icon_labels(child)

