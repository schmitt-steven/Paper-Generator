"""Metadata management for LaTeX document generation."""

from dataclasses import dataclass
from typing import List, Dict

from settings import Settings


@dataclass
class LaTeXMetadata:
    """Metadata for LaTeX document generation (IEEEtran format)."""

    title: str
    authors: list[dict[str, str]]  # List of author dictionaries

    @classmethod
    def from_settings(cls, generated_title: str) -> "LaTeXMetadata":
        """Create LaTeXMetadata from settings.
        
        Args:
            generated_title: Title from PaperDraft (respects Settings.LATEX_TITLE if set)
        
        Returns:
            LaTeXMetadata with all fields for IEEEtran template
        """
        return cls(
            title=generated_title,
            authors=Settings.LATEX_AUTHORS,
        )

