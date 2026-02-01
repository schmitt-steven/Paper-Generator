from pathlib import Path
import re
from phases.paper_writing.data_models import Section

class SectionGuidelinesLoader:
    """Manages loading and saving of per-section style guidelines."""
    
    FILE_PATH = Path("user_files/style_guidelines.md")

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
            print(f"Error loading style guidelines: {e}")
            return {}

    @classmethod
    def save_guidelines(cls, guidelines: dict[Section, str]) -> None:
        """Save guidelines to markdown file."""
        cls.FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        lines = ["# Style Guidelines"]
        
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
                
        for i, section in enumerate(ordered_sections):
            if section in guidelines:
                header_name = section.value.title()
                if i > 0:
                    lines.append("")  # Blank line before ## (except first section)
                lines.append(f"## {header_name}")
                lines.append(guidelines[section].strip())
        
        try:
            cls.FILE_PATH.write_text("\n".join(lines), encoding="utf-8")
            print(f"Saved style guidelines to {cls.FILE_PATH}")
        except Exception as e:
            print(f"Error saving style guidelines: {e}")

    @classmethod
    def get_guidelines(cls, section_type: Section, experiment=None) -> str:
        """Get guidelines for a section from the loaded file."""
        import textwrap
        
        # Load from file
        custom_guidelines = cls.load_guidelines()
        guideline = custom_guidelines.get(section_type, "")
        
        # If results section and experiment has plots, append plot instructions
        if section_type == Section.RESULTS and experiment and experiment.plots:
             guideline += "\n\n" + cls._get_results_plot_instructions(experiment)
             
        return guideline

    @staticmethod
    def _get_results_plot_instructions(experiment) -> str:
        """Get instructions for integrating plots into the Results section."""
        import textwrap
        
        plots_block = ""
        for idx, plot in enumerate(experiment.plots, 1):
            filename = plot.filename
            # Simplify path logic
            if filename.startswith("output/"):
                filename = filename[len("output/"):]
            
            plots_block += f"Figure {idx}:\n  Filename: {filename}\n  Caption: {plot.caption}\n\n"

        if not plots_block:
            return ""

        return textwrap.dedent(f"""
            [FIGURE INTEGRATION]
            The following figures were generated from the experiment. You MUST integrate all of them into your Results section.

            {plots_block.strip()}

            For each figure:
            1. Reference it naturally in the text (e.g., "As shown in Figure 1..." or "Figure 2 demonstrates...")
            2. Include the markdown image syntax: ![Brief alt text](relative_path_to_image.png)
            3. CRITICAL: Use RELATIVE paths from the paper_draft.md location (which is in the output/ directory).
               - If filename is "experiments/plots/file.pdf", use exactly that (no "output/" prefix)
            4. Add a visible caption line immediately below: *Figure N: Full caption text*
            5. Use the exact caption text provided above for each figure
        """)
