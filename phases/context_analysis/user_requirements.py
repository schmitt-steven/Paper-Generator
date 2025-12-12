from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re
from pathlib import Path


@dataclass
class UserRequirements:
    """Structured user requirements for paper generation"""
    topic: str
    hypothesis: str
    abstract: str
    introduction: str
    related_work: str
    methods: str
    results: str
    discussion: str
    conclusion: str
    acknowledgements: Optional[str] = None

    @staticmethod
    def load_user_requirements(file_path: str) -> UserRequirements:
        """Load and parse user_requirements.md file into UserRequirements object."""
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"User requirements file not found: {file_path}")

        content = path.read_text(encoding='utf-8')

        # Initialize all sections as empty strings or None
        sections = {
            'topic': "",
            'hypothesis': "",
            'abstract': "",
            'introduction': "",
            'related_work': "",
            'methods': "",
            'results': "",
            'discussion': "",
            'conclusion': "",
            'acknowledgements': None
        }

        # Split content by lines for easier processing
        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            # Check for section headers
            if line.startswith('## General Information'):
                # Just a grouping header, reset current section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = None
                section_content = []
            elif line.startswith('### Topic'):
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = 'topic'
                section_content = []
            elif line.startswith('### Hypothesis'):
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = 'hypothesis'
                section_content = []
            elif line.startswith('## Section Specifications'):
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = None  # Stop collecting until we hit a ### header
            elif line.startswith('### '):
                section_name = line[4:].strip()  # Remove '### '
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = section_name.lower().replace(' ', '_')
                section_content = []
            elif current_section is not None:
                section_content.append(line)

        # Save last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content).strip()

        #  Acknowledgements - if empty or just whitespace, set to None
        if not sections['acknowledgements'] or sections['acknowledgements'].strip() == "":
            sections['acknowledgements'] = None

        return UserRequirements(**sections)
