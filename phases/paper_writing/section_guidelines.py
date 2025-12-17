from pathlib import Path
import re
from phases.paper_writing.data_models import Section

class SectionGuidelinesLoader:
    """Manages loading and saving of per-section writing guidelines."""
    
    FILE_PATH = Path("user_files/section_guidelines.md")

    @classmethod
    def load_guidelines(cls) -> dict[Section, str]:
        """Load guidelines from markdown file. Returns dict of Section -> text."""
        if not cls.FILE_PATH.exists():
            return {}
            
        try:
            content = cls.FILE_PATH.read_text(encoding="utf-8")
            guidelines = {}
            
            # Split by markdown headers lvl 2 (e.g., "## Abstract")
            # Pattern matches ## Header Name, captures name, then content until next header
            parts = re.split(r'^##\s+(.+)$', content, flags=re.MULTILINE)
            
            # parts[0] is preamble (ignore)
            # parts[1] is header1, parts[2] is content1, parts[3] is header2, etc.
            
            for i in range(1, len(parts), 2):
                section_name = parts[i].strip().upper()
                section_text = parts[i+1].strip()
                
                # Map section name to Section enum
                try:
                    # Handle multi-word sections (RELATED WORK -> RELATED_WORK)
                    enum_name = section_name.replace(" ", "_")
                    section_enum = Section[enum_name]
                    guidelines[section_enum] = section_text
                except KeyError:
                    # Try reverse lookup by value
                    found = False
                    for s in Section:
                        if s.value.upper() == section_name:
                            guidelines[s] = section_text
                            found = True
                            break
                    if not found:
                        print(f"Warning: Unknown section in guidelines file: {section_name}")
            
            return guidelines
            
        except Exception as e:
            print(f"Error loading section guidelines: {e}")
            return {}

    @classmethod
    def save_guidelines(cls, guidelines: dict[Section, str]) -> None:
        """Save guidelines to markdown file."""
        cls.FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        lines = ["# Section Guidelines"]
        
        # Sort by standard paper order if possible, or just iteration order
        ordered_sections = [
            Section.ABSTRACT, 
            Section.INTRODUCTION, 
            Section.RELATED_WORK,
            Section.METHODS, 
            Section.RESULTS, 
            Section.DISCUSSION, 
            Section.CONCLUSION, 
            Section.ACKNOWLEDGEMENTS
        ]
        
        # Add any others not in the ordered list
        for s in guidelines:
            if s not in ordered_sections:
                ordered_sections.append(s)
                
        for section in ordered_sections:
            if section in guidelines:
                # Header formatting: "Abstract", "Related Work" (Title Case from value is decent enough)
                header_name = section.value.title()
                lines.append(f"## {header_name}")
                lines.append("")
                lines.append(guidelines[section].strip())
                lines.append("")
                lines.append("")
        
        try:
            cls.FILE_PATH.write_text("\n".join(lines), encoding="utf-8")
            print(f"Saved section guidelines to {cls.FILE_PATH}")
        except Exception as e:
            print(f"Error saving section guidelines: {e}")
