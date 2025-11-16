"""LLM-based markdown to LaTeX conversion."""

import textwrap
import lmstudio as lms
from phases.paper_writing.data_models import Section


class MarkdownToLaTeX:
    """Converts markdown text to LaTeX format using LLM."""

    @staticmethod
    def convert_section_to_latex(md_text: str, section_type: Section, llm) -> str:
        """Convert markdown section text to LaTeX using LLM."""

        prompt = MarkdownToLaTeX._build_conversion_prompt(md_text)
        
        try:
            response = llm.act(prompt, config={"temperature": 0.2})
            # Extract text from response
            if hasattr(response, "content"):
                latex_text = response.content
            elif isinstance(response, str):
                latex_text = response
            else:
                latex_text = str(response)
            
            latex_text = MarkdownToLaTeX._clean_latex_output(latex_text)
            
            return latex_text.strip()
        except Exception as e:
            print(f"[MarkdownToLaTeX] Error converting section {section_type.value}: {e}")

    @staticmethod
    def _build_conversion_prompt(md_text: str) -> str:
        """Build the LLM prompt for markdown to LaTeX conversion."""

        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer and LaTeX formatter.

            [TASK]
            Convert the following markdown text to LaTeX format.

            [CONVERSION RULES]
            - Citations: Convert (smith2024quantum) to \\cite{{smith2024quantum}}
            - Multiple citations: Convert (smith2024; jones2023) to \\cite{{smith2024,jones2023}}
            - Abbreviations: Identify abbreviations and convert them properly
            - On FIRST occurrence: Write "Full Form (ABBR)" format, e.g., "Artificial Intelligence (AI)"
            - On SUBSEQUENT occurrences: Use \\ac{{ABBR}} format with uppercase abbreviation, e.g., \\ac{{AI}}
            - The full form will be automatically extracted from the first occurrence
            - Figures: Convert ![alt text](filename.png) followed by *Figure N: Caption text* to:
            \\begin{{figure}}[ht]
            \\centering
            \\includegraphics{{filename.png}}
            \\caption{{Caption text}}
            \\label{{fig:filename}}
            \\end{{figure}}
            - Code blocks: Convert ```python ... ``` to \\begin{{lstlisting}}[language=Python]...\\end{{lstlisting}}
            - Math: Preserve $...$ for inline math and $$...$$ for display math (or convert to \\[...\\])
            - Headers: Convert # Title to \\subsection{{Title}}, ## Subtitle to \\subsubsection{{Subtitle}} (Note: Main sections use \\section and are added by the generator)
            - Bold/italic: Convert **text** to \\textbf{{text}}, *text* to \\textit{{text}}
            - Lists: Convert markdown lists to LaTeX \\begin{{itemize}}...\\end{{itemize}} or \\begin{{enumerate}}...\\end{{enumerate}}
            - Escape LaTeX special characters: _, &, %, {{, }} must be escaped as \\_, \\&, \\%, \\{{, \\}}
            - Paragraphs: Preserve paragraph breaks (double newlines)
            - Do NOT alter text content, only convert formatting
            - Follow academic LaTeX conventions

            [INPUT MARKDOWN]
            {md_text}

            [OUTPUT REQUIREMENTS]
            - Output ONLY the LaTeX-formatted text
            - Do NOT include any explanations or comments
            - Do NOT wrap in \\section{{}} as the main section header is added automatically
            - Use \\subsection{{}} and \\subsubsection{{}} for any subsections if needed
            - Ensure all citations are properly formatted as \\cite{{key}}
            - Ensure all figures have proper \\begin{{figure}} environments with \\caption and \\label
            - Ensure all subsequent abbreviations use \\ac{{ABBR}} format with uppercase abbreviations

            Convert the markdown to LaTeX now:""")

    @staticmethod
    def _clean_latex_output(latex_text: str) -> str:
        """Clean up LaTeX output by removing remaining markdown syntax if present."""

        latex_text = latex_text.strip()
        if latex_text.startswith("```"):
            # Remove markdown code fences if LLM added them
            lines = latex_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            latex_text = "\n".join(lines)
        return latex_text


