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
            response = llm.respond(prompt, config={"temperature": 0.0})
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
            return ""

    @staticmethod
    def _build_conversion_prompt(md_text: str) -> str:
        """Build the LLM prompt for markdown to LaTeX conversion."""

        return textwrap.dedent(f"""\
            [ROLE]
            You are an expert academic writer and LaTeX formatter.

            [TASK]
            Convert the following markdown text to LaTeX format.

            [CONVERSION RULES]
            - Citations: Convert ALL citations in the format [citationKey] to \\cite{{citationKey}}
            - CRITICAL: Preserve citation keys EXACTLY as they appear. Do NOT modify, shorten, or change citation keys (e.g., if markdown has [Diekhoff2024RecursiveBQ], LaTeX must use \\cite{{Diekhoff2024RecursiveBQ}}, NOT \\cite{{diekhoff2024}})
            - Multiple citations: Convert citations like [key1, key2] or [key1; key2] to \\cite{{key1,key2}} (preserve exact keys, convert semicolons to commas, remove spaces after commas/semicolons)
              - Example: [Memarian2021SelfSupervisedOR; Park2025FromST] -> \\cite{{Memarian2021SelfSupervisedOR,Park2025FromST}}
              - Example: [Ghasemi2024ACS; Lo2022GoalSpacePW] -> \\cite{{Ghasemi2024ACS,Lo2022GoalSpacePW}}
            - Abbreviations: Identify abbreviations and convert them properly
                - On FIRST occurrence in the document: Write "Full Form (ABBR)" format, e.g., "Recursive Backwards Q-Learning (RBQL)"
                - On ALL SUBSEQUENT occurrences: Use \\ac{{ABBR}} format with UPPERCASE abbreviation, e.g., \\ac{{RBQL}}
                - CRITICAL: Always use UPPERCASE for abbreviation keys in \\ac{{}}, e.g., \\ac{{RBQL}}, \\ac{{EBU}}, NOT \\ac{{rbql}} or \\ac{{Rbql}}
                - For abbreviations with hyphens like "Dyna-Q", use \\ac{{DYNA-Q}} (uppercase with hyphen)
                - Do NOT repeat "Full Form (ABBR)" format after the first occurrence - always use \\ac{{ABBR}} for subsequent uses
                - Do NOT add abbreviations to text that doesn't have them - if text says "Dyna-Q", only convert if it was previously defined as "Dyna-Q (DYNA-Q)"
            - The full form will be automatically extracted from the first occurrence
            - Figures: Convert ![alt text](path/to/filename.png) followed by *Figure N: Caption text* to:
            \\begin{{figure*}}[ht]
            \\centering
            \\includegraphics[width=\\textwidth]{{images/filename.png}}
            \\caption{{Caption text}}
            \\label{{fig:filename}}
            \\end{{figure*}}
            - CRITICAL: Always use figure* (with asterisk) to span full page width!
            - Use width=\\textwidth to ensure images fit within page boundaries
            - For images: Extract the ACTUAL filename from the markdown path and use it EXACTLY.
              - If markdown has: ![text](experiments/plots/convergence_comparison.png)
              - Use: \\includegraphics{{images/convergence_comparison.png}} (extract "convergence_comparison.png" from the path)
              - Do NOT generate generic names like "figure1.png" or "image1.png"
              - Do NOT use the full path from markdown, only extract the filename (the part after the last "/")
              - Always use images/ directory prefix, but keep the original filename
              - CRITICAL: NEVER use the same image filename twice. Each figure must use a UNIQUE filename from the markdown.
                If you see multiple figures, each must use its own distinct filename from the markdown path.
            - CRITICAL: ONLY convert images that are explicitly present in the markdown with ![alt](path) syntax.
              - Do NOT add figures based on text references like "Figure 1" or "as shown in Figure X"
              - Do NOT create figure environments unless there is an actual ![alt](path) image markdown in the text
              - Text references to figures (e.g., "Figure 1 shows...") should remain as text, NOT converted to figure environments
            - Example: ![Alt](experiments/plots/my_plot.png) -> \\includegraphics{{images/my_plot.png}} (NOT images/figure1.png)
            - Code blocks: Convert ```python ... ``` to \\begin{{lstlisting}}[language=Python]...\\end{{lstlisting}}
            - Math: Preserve $...$ for inline math and $$...$$ for display math (or convert to \\[...\\])
            - Greek letters: Convert Unicode Greek letters to LaTeX math mode:
              - α (alpha) -> $\\alpha$ or \\alpha (in math mode)
              - β (beta) -> $\\beta$ or \\beta (in math mode)
              - γ (gamma) -> $\\gamma$ or \\gamma (in math mode)
              - δ (delta) -> $\\delta$ or \\delta (in math mode)
              - ε (epsilon) -> $\\varepsilon$ or \\varepsilon (in math mode)
              - θ (theta) -> $\\theta$ or \\theta (in math mode)
              - λ (lambda) -> $\\lambda$ or \\lambda (in math mode)
              - μ (mu) -> $\\mu$ or \\mu (in math mode)
              - π (pi) -> $\\pi$ or \\pi (in math mode)
              - σ (sigma) -> $\\sigma$ or \\sigma (in math mode)
              - φ (phi) -> $\\phi$ or \\phi (in math mode)
              - ω (omega) -> $\\omega$ or \\omega (in math mode)
              - Always wrap Greek letters in math mode: if you see "α=1", convert to "$\\alpha=1$"
            - Headers: Convert # Title to \\subsection{{Title}}, ## Subtitle to \\subsubsection{{Subtitle}} (Note: Main sections use \\section and are added by the generator)
            - Bold/italic: Convert **text** to \\textbf{{text}}, *text* to \\textit{{text}}
            - Lists: Convert markdown lists to LaTeX \\begin{{itemize}}...\\end{{itemize}} or \\begin{{enumerate}}...\\end{{enumerate}}
            - Escape LaTeX special characters: _, &, %, {{, }} must be escaped as \\_, \\&, \\%, \\{{, \\}}
            - Paragraphs: Preserve paragraph breaks (double newlines)
            - Never alter text content, only convert formatting
            - Follow academic LaTeX conventions

            [INPUT MARKDOWN]
            {md_text}

            [OUTPUT REQUIREMENTS]
            1. Output ONLY the LaTeX-formatted text
            2. Do NOT include any explanations or comments
            3. Do NOT wrap in \\section{{}} as the main section header is added automatically
            4. Use \\subsection{{}} and \\subsubsection{{}} for any subsections if needed
            5. Ensure all citations are properly formatted as \\cite{{key}} with EXACT citation keys preserved (e.g., \\cite{{Diekhoff2024RecursiveBQ}}, not \\cite{{diekhoff2024}})
            6. Ensure all figures have proper \\begin{{figure}} environments with \\caption and \\label
            7. Ensure all subsequent abbreviations use \\ac{{ABBR}} format with uppercase abbreviations

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


