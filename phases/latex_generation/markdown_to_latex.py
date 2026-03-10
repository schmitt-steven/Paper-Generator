"""LLM-based markdown to LaTeX conversion."""

import re
import textwrap
import lmstudio as lms
from phases.paper_writing.data_models import Section
from utils.llm_utils import remove_thinking_blocks


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
                latex_text = remove_thinking_blocks(response.content)
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

            - Figures: Convert ![alt text](path/to/filename.png) followed by *Figure N: Caption text* to:
            \\begin{{figure*}}[ht]
            \\centering
            \\includegraphics[width=\\textwidth]{{images/filename.png}}
            \\caption{{Caption text}}
            \\Description{{alt text}}
            \\label{{fig:filename}}
            \\end{{figure*}}
            - CRITICAL: Always use figure* (with asterisk) to span full page width!
            - CRITICAL: Always include \\Description{{}} command for accessibility (required by JAIR/ACM templates)
              - Use the alt text from the markdown image as the description
              - If alt text is generic (e.g., "figure", "image"), write a brief description of what the figure shows based on context
              - Description should be plain text, under 2000 characters, describing what someone who cannot see the image needs to know
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

            - Tables: Convert markdown tables to LaTeX using booktabs style:
              - Markdown table format:
                | Header1 | Header2 | Header3 |
                |---------|---------|---------|
                | Cell1   | Cell2   | Cell3   |
              - Convert to LaTeX:
                \\begin{{table}}[ht]
                \\caption{{Table caption from context or *Table N: Caption* if present}}
                \\label{{tab:descriptive_name}}
                \\centering
                \\begin{{tabular}}{{@{{}}lll@{{}}}}
                \\toprule
                Header1 & Header2 & Header3 \\\\
                \\midrule
                Cell1 & Cell2 & Cell3 \\\\
                \\bottomrule
                \\end{{tabular}}
                \\end{{table}}
              - Use @{{}} to remove extra horizontal padding
              - Use l (left), c (center), or r (right) alignment based on content type (text=left, numbers=right)
              - ALWAYS use \\toprule, \\midrule, \\bottomrule (booktabs style) - never use \\hline
              - Place \\caption ABOVE the tabular (required by many journals)
              - For wide tables, use table* environment to span full page width

            - Code blocks: Convert ```python ... ``` to \\begin{{lstlisting}}[language=Python]...\\end{{lstlisting}}
            - Inline code: Convert `code` to \\texttt{{code}} (ensure special chars inside are escaped)
            - Math: Preserve $...$ for inline math and $$...$$ for display math (or convert to \\[...\\])
            - Headers: Convert # Title to \\subsection{{Title}}, ## Subtitle to \\subsubsection{{Subtitle}} (Note: Main sections use \\section and are added by the generator)
            - Bold/italic: Convert **text** to \\textbf{{text}}, *text* to \\textit{{text}}
            - Lists: Convert markdown lists to LaTeX \\begin{{itemize}}...\\end{{itemize}} or \\begin{{enumerate}}...\\end{{enumerate}}
            - Paragraphs: Preserve paragraph breaks (double newlines)
            - NEVER alter text content, only convert formatting
            - Follow academic LaTeX conventions

            - Hyperlinks: Convert markdown links [text](url) to \\href{{url}}{{text}} (requires hyperref package, already included in templates)
              - If the link text equals the URL, use \\url{{url}} instead
              - Do NOT confuse with citations [citationKey] which have no parentheses URL part

            - Smart quotes and typography:
              - " (left double quote) and " (right double quote) -> ``text''
              - ' (left single quote) and ' (right single quote) -> `text'
              - Straight quotes "text" -> ``text''
              - … (ellipsis) -> \\ldots
              - — (em-dash) -> ---
              - – (en-dash) -> --
              - Non-breaking space (\\u00A0) -> ~

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
            
            - Escape LaTeX special characters:
              - You MUST escape the following characters in normal text (unless in a code block or URL macro):
                - \\ (backslash) -> \\textbackslash
                - _ (underscore) -> \\_
                - % (percent) -> \\%
                - $ (dollar) -> \\$ (unless determining math mode)
                - # (hash) -> \\#
                - & (ampersand) -> \\&
                - {{ (curly brace) -> \\{{
                - }} (curly brace) -> \\}}
                - ~ (tilde) -> \\textasciitilde
                - ^ (caret) -> \\textasciicircum
              - Forward slash /: Leave as is in text (e.g., "input/output" remains "input/output"). Do not escape / as it is not a special character in LaTeX text.
              - URLs and File Paths: If a path or URL appears in text (not in a listing), consider deciding if it needs \\path{{...}} or \\url{{...}} to handle breaking, but at minimum ensure special chars are escaped if written as plain text.

            [INPUT MARKDOWN]
            {md_text}

            [OUTPUT REQUIREMENTS]
            1. Output ONLY the LaTeX-formatted text
            2. Do NOT include any explanations or comments
            3. Do NOT wrap in \\section{{}} as the main section header is added automatically
            4. Use \\subsection{{}} and \\subsubsection{{}} for any subsections if needed
            5. Ensure all citations are properly formatted as \\cite{{key}} with EXACT citation keys preserved (e.g., \\cite{{Diekhoff2024RecursiveBQ}}, not \\cite{{diekhoff2024}})
            6. Ensure all figures have proper \\begin{{figure}} environments with \\caption and \\label
            7. Check and escape ALL special characters (e.g., underscores in filenames or variable names that are not in code/math mode).

            Convert the markdown to LaTeX now:""")

    # Unicode to LaTeX conversion mapping for Greek letters and common math symbols
    UNICODE_TO_LATEX = {
        # Greek lowercase
        'α': r'$\alpha$', 'β': r'$\beta$', 'γ': r'$\gamma$', 'δ': r'$\delta$',
        'ε': r'$\varepsilon$', 'ζ': r'$\zeta$', 'η': r'$\eta$', 'θ': r'$\theta$',
        'ι': r'$\iota$', 'κ': r'$\kappa$', 'λ': r'$\lambda$', 'μ': r'$\mu$',
        'ν': r'$\nu$', 'ξ': r'$\xi$', 'π': r'$\pi$', 'ρ': r'$\rho$',
        'σ': r'$\sigma$', 'τ': r'$\tau$', 'υ': r'$\upsilon$', 'φ': r'$\phi$',
        'χ': r'$\chi$', 'ψ': r'$\psi$', 'ω': r'$\omega$',
        # Greek uppercase
        'Α': r'$A$', 'Β': r'$B$', 'Γ': r'$\Gamma$', 'Δ': r'$\Delta$',
        'Ε': r'$E$', 'Ζ': r'$Z$', 'Η': r'$H$', 'Θ': r'$\Theta$',
        'Ι': r'$I$', 'Κ': r'$K$', 'Λ': r'$\Lambda$', 'Μ': r'$M$',
        'Ν': r'$N$', 'Ξ': r'$\Xi$', 'Π': r'$\Pi$', 'Ρ': r'$P$',
        'Σ': r'$\Sigma$', 'Τ': r'$T$', 'Υ': r'$\Upsilon$', 'Φ': r'$\Phi$',
        'Χ': r'$X$', 'Ψ': r'$\Psi$', 'Ω': r'$\Omega$',
        # Common math symbols
        '∈': r'$\in$', '∉': r'$\notin$', '⊂': r'$\subset$', '⊃': r'$\supset$',
        '⊆': r'$\subseteq$', '⊇': r'$\supseteq$', '∪': r'$\cup$', '∩': r'$\cap$',
        '∅': r'$\emptyset$', '∞': r'$\infty$', '≤': r'$\leq$', '≥': r'$\geq$',
        '≠': r'$\neq$', '≈': r'$\approx$', '±': r'$\pm$', '×': r'$\times$',
        '÷': r'$\div$', '·': r'$\cdot$', '∑': r'$\sum$', '∏': r'$\prod$',
        '∫': r'$\int$', '∂': r'$\partial$', '∇': r'$\nabla$', '√': r'$\sqrt{}$',
        '→': r'$\rightarrow$', '←': r'$\leftarrow$', '↔': r'$\leftrightarrow$',
        '⇒': r'$\Rightarrow$', '⇐': r'$\Leftarrow$', '⇔': r'$\Leftrightarrow$',
        '∀': r'$\forall$', '∃': r'$\exists$', '¬': r'$\neg$', '∧': r'$\land$',
        '∨': r'$\lor$', '⊕': r'$\oplus$', '⊗': r'$\otimes$',
        # Superscripts and subscripts (common ones)
        '²': r'$^2$', '³': r'$^3$', '⁴': r'$^4$', '⁵': r'$^5$',
        '⁶': r'$^6$', '⁷': r'$^7$', '⁸': r'$^8$', '⁹': r'$^9$',
        '₀': r'$_0$', '₁': r'$_1$', '₂': r'$_2$', '₃': r'$_3$',
        '₄': r'$_4$', '₅': r'$_5$', '₆': r'$_6$', '₇': r'$_7$',
        '₈': r'$_8$', '₉': r'$_9$',
    }

    @staticmethod
    def _clean_latex_output(latex_text: str) -> str:
        """Clean up LaTeX output by removing remaining markdown syntax and converting any remaining Unicode characters to LaTeX equivalents."""

        latex_text = latex_text.strip()
        if latex_text.startswith("```"):
            # Remove markdown code fences if LLM added them
            lines = latex_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            latex_text = "\n".join(lines)
        
        # Convert Unicode characters to LaTeX equivalents
        latex_text = MarkdownToLaTeX._convert_unicode_to_latex(latex_text)

        # Escape special characters that the LLM missed
        latex_text = MarkdownToLaTeX._escape_special_chars_outside_math(latex_text)

        return latex_text

    @staticmethod
    def _convert_unicode_to_latex(text: str) -> str:
        """Convert Unicode Greek letters and math symbols to LaTeX equivalents."""
        for unicode_char, latex_equiv in MarkdownToLaTeX.UNICODE_TO_LATEX.items():
            text = text.replace(unicode_char, latex_equiv)
        return text

    # Characters that must be escaped in LaTeX text mode (not in math/commands).
    # Maps bare char -> escaped form. Order matters: backslash is not included
    # since escaping bare backslashes in LLM output is too risky (likely already a command).
    _SPECIAL_CHARS = {
        '_': r'\_',
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
    }

    @staticmethod
    def _escape_special_chars_outside_math(text: str) -> str:
        """Escape LaTeX special characters that appear in text mode.

        Splits the input into protected regions (math mode, LaTeX commands
        with braced arguments, and verbatim/lstlisting environments) vs.
        plain text regions.  Only plain text regions are checked for
        unescaped special characters.
        """
        # Pattern to identify regions where special chars are valid and should be left alone:
        #  - $$ ... $$  (display math)
        #  - $ ... $    (inline math)
        #  - \command{...} (commands with braced args — one level of braces)
        #  - \begin{lstlisting}...\end{lstlisting}
        #  - \begin{verbatim}...\end{verbatim}
        #  - \url{...}  (already caught by command pattern but listed for clarity)
        protected_pattern = re.compile(
            r'\\begin\{(?:lstlisting|verbatim)\}.*?\\end\{(?:lstlisting|verbatim)\}'  # verbatim envs
            r'|'
            r'\$\$.*?\$\$'          # display math
            r'|'
            r'\$[^$]+?\$'           # inline math
            r'|'
            r'\\[a-zA-Z]+\*?\{[^}]*\}'  # \command{...} (single-level braces)
            r'|'
            r'\\[a-zA-Z]+'          # bare commands like \textbf (no braces)
            r'|'
            r'\\.',                 # escaped char like \_ \& \% \#
            re.DOTALL
        )

        parts = []
        last_end = 0

        for match in protected_pattern.finditer(text):
            start, end = match.span()
            # Process the unprotected text before this match
            if start > last_end:
                parts.append(
                    MarkdownToLaTeX._escape_text_region(text[last_end:start])
                )
            # Keep the protected region as-is
            parts.append(match.group())
            last_end = end

        # Process any remaining unprotected text after the last match
        if last_end < len(text):
            parts.append(
                MarkdownToLaTeX._escape_text_region(text[last_end:])
            )

        return ''.join(parts)

    @staticmethod
    def _escape_text_region(text: str) -> str:
        """Escape special characters in a plain-text region."""
        for char, escaped in MarkdownToLaTeX._SPECIAL_CHARS.items():
            text = text.replace(char, escaped)
        return text


