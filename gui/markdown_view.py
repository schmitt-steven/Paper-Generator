import tkinter as tk
from tkinter import ttk
import markdown
import webbrowser
from tkinterweb import HtmlFrame
import sv_ttk
import re
import sys
from .theme_colors import (
    TEXT_BG_DARK, TEXT_BG_LIGHT,
    TEXT_BG_DARK_ALT, TEXT_BG_LIGHT_ALT,
    TEXT_FG_DARK, TEXT_FG_LIGHT,
    LINK_COLOR_DARK, LINK_COLOR_LIGHT,
    BORDER_DARK, BORDER_LIGHT,
    CANVAS_BG_DARK, CANVAS_BG_LIGHT,
    CODE_BG_DARK, CODE_BG_LIGHT,
    CODE_FG_DARK, CODE_FG_LIGHT,
    TABLE_BORDER_DARK, TABLE_BORDER_LIGHT,
    TABLE_HEADER_BG_DARK, TABLE_HEADER_BG_LIGHT,
    HEADER_BORDER_DARK, HEADER_BORDER_LIGHT
)

class MarkdownView(HtmlFrame):
    """A widget that renders Markdown using tkinterweb's HtmlFrame."""
    def __init__(self, parent, font_manager=None, theme_mode="dark", **kwargs):
        # Extract arguments not supported by ttk.Frame init
        padx = kwargs.pop('padx', 0)
        pady = kwargs.pop('pady', 0)
        
        # Initialize HtmlFrame
        # messages_enabled=False prevents debug prints to stdout
        # on_link_click handles external links
        super().__init__(parent, messages_enabled=False, horizontal_scrollbar="auto", on_link_click=self._on_link_click, **kwargs)
        
        self.font_manager = font_manager
        self.theme_mode = theme_mode
        
        if theme_mode == "dark":
            self.base_bg = TEXT_BG_DARK
            self.base_fg = TEXT_FG_DARK
        else:
            self.base_bg = TEXT_BG_LIGHT_ALT
            self.base_fg = TEXT_FG_LIGHT

        self._current_markdown = None
        
        # Register for font updates if manager is provided
        if self.font_manager:
            self.font_manager.add_callback(self._update_font)
            
        # Bind to virtual event for theme changes
        self.bind("<<ThemeChanged>>", self._on_theme_changed, add="+")

    def _update_font(self):
        """Callback for when system font size changes."""
        if self._current_markdown:
            # Re-render content with new font sizes
            self.set_markdown(self._current_markdown)

    def destroy(self):
        """Cleanup before destruction."""
        if self.font_manager:
            self.font_manager.remove_callback(self._update_font)
        super().destroy()

    def _on_theme_changed(self, event=None):
        """Handle theme change event."""
        self.theme_mode = sv_ttk.get_theme()
        
        if self.theme_mode == "dark":
            self.base_bg = TEXT_BG_DARK_ALT
            self.base_fg = TEXT_FG_DARK
        else:
            self.base_bg = TEXT_BG_LIGHT_ALT
            self.base_fg = TEXT_FG_LIGHT
            
        if self._current_markdown:
            self.set_markdown(self._current_markdown)

    def _on_link_click(self, url):
        """Open links in default browser instead of inside the widget."""
        webbrowser.open(url)

    def set_markdown(self, markdown_text):
        """Convert markdown to HTML and render it."""
        self._current_markdown = markdown_text
        
        if not markdown_text:
            self.load_html("")
            return

        markdown_text = re.sub(r'^\s*\*\*(.+?)\*\*[:]?\s*$', r'### \1', markdown_text, flags=re.MULTILINE)

        # Convert markdown to html
        html_content = markdown.markdown(
            markdown_text, 
            extensions=['fenced_code', 'tables', 'sane_lists', 'extra']
        )
        
        # Determine fonts
        if sys.platform == "win32":
            font_family = "Bahnschrift"
            mono_family = "Consolas"
        elif sys.platform == "darwin":
            font_family = "SF Pro"
            mono_family = "Menlo"
        else:
            font_family = "Helvetica"
            mono_family = "Courier"
        
        # Determine colors based on theme
        is_dark = self.theme_mode == "dark"
        
        link_color = LINK_COLOR_DARK if is_dark else LINK_COLOR_LIGHT
        code_bg = CODE_BG_DARK if is_dark else CODE_BG_LIGHT
        code_fg = CODE_FG_DARK if is_dark else CODE_FG_LIGHT
        border_color = BORDER_DARK if is_dark else BORDER_LIGHT
        header_border = HEADER_BORDER_DARK if is_dark else HEADER_BORDER_LIGHT
        table_border = TABLE_BORDER_DARK if is_dark else TABLE_BORDER_LIGHT
        th_bg = TABLE_HEADER_BG_DARK if is_dark else TABLE_HEADER_BG_LIGHT
        
        font_size = 12
        if self.font_manager:
            font_family = self.font_manager.font_family
            mono_family = self.font_manager.mono_family
            
            font_size = self.font_manager.default_font.cget("size")
            if font_size < 0: font_size = -font_size
            
        css = f"""
        <style>
            body {{
                font-family: '{font_family}', sans-serif;
                font-size: {font_size}px;
                background-color: {self.base_bg};
                color: {self.base_fg};
                margin: 0;
                padding: 15px;
            }}
            
            /* Headers - Scaled relative to body font size */
            h1, h2, h3, h4, h5, h6 {{
                color: {self.base_fg};
                font-weight: bold;
                margin-top: 0.6em;
                margin-bottom: 0.3em;
                margin-left: 0;
                padding-left: 0;
            }}
            h1 {{ font-size: 1.2em; border-bottom: 1px solid {header_border}; padding-bottom: 5px; }}
            h2 {{ font-size: 1.15em; margin-top: 0.8em; }}
            h3 {{ font-size: 1.1em; }}
            h4 {{ font-size: 1.05em; text-decoration: underline; }}
            
            /* Remove top margin for first header to avoid extra spacing at top */
            h1:first-child, h2:first-child, h3:first-child, h4:first-child {{
                margin-top: 0;
            }}
            
            /* Text */
            p {{ 
                margin-bottom: 0.6em; 
                line-height: 1.3;
                margin-top: 0;
            }}
            
            /* Links */
            a {{ color: {link_color}; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            
            /* Lists */
            ul {{
                margin-top: 0.2em;
                margin-bottom: 0.6em;
                margin-left: 0; 
                padding-left: 1.5em; 
                list-style-position: outside;
                list-style-type: disc !important;
            }}
            
            ol {{
                margin-top: 0.2em;
                margin-bottom: 0.6em;
                margin-left: 0; 
                padding-left: 2.0em; 
                list-style-position: outside;
                list-style-type: decimal !important;
            }}
            
            li {{
                margin-bottom: 0.2em;
                margin-left: 0;
            }}
            
            /* Code */
            code {{
                font-family: '{mono_family}', monospace;
                background-color: {code_bg};
                color: {code_fg};
                padding: 2px 4px;
                border-radius: 3px;
                font-size: 0.9em;
            }}
            
            pre {{
                background-color: {code_bg};
                color: {code_fg};
                padding: 12px;
                border-radius: 6px;
                overflow-x: auto;
                margin-bottom: 1em;
            }}
            pre code {{
                padding: 0;
                background-color: transparent;
                border-radius: 0;
            }}
            
            /* Blockquotes */
            blockquote {{
                border-left: 4px solid {border_color};
                margin-left: 0;
                padding-left: 1em;
                opacity: 0.8;
            }}
            
            /* Tables */
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1em;
            }}
            th, td {{
                border: 1px solid {table_border};
                padding: 6px 10px;
                text-align: left;
            }}
            th {{
                background-color: {th_bg};
                font-weight: bold;
            }}
            
            /* Images */
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 10px 0;
            }}
        </style>
        """
        
        final_html = f"<html><head>{css}</head><body>{html_content}</body></html>"
        
        # Load into tkinterweb HtmlFrame
        self.load_html(final_html)
