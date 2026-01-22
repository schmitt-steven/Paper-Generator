import os
import re
import json
import ast
import textwrap
from pathlib import Path
from dataclasses import dataclass
from typing import cast
from pydantic import BaseModel
from settings import Settings
from utils.lazy_model_loader import LazyModelMixin
from utils.llm_utils import remove_thinking_blocks


@dataclass
class UserCode:
    """Stores analysis results for a code file"""
    file_path: str
    file_name: str
    file_content: str
    summary: str = ""
    keywords: list[str] = None
    method: str = ""
    contribution: str = ""
    important_snippets: list['CodeSnippet'] | None = None
    signatures: list[str] | None = None
    
    def __post_init__(self):
        if self.important_snippets is None:
            self.important_snippets = []
        if self.keywords is None:
            self.keywords = []
        if self.signatures is None:
            self.signatures = []


class CodeSnippet(BaseModel):
    """Represents an important code snippet extracted from a file"""
    label: str
    code: str


class UserCodeAnalysisResult(BaseModel):
    """Structured response format used by the LLM"""
    summary: str
    keywords: list[str]
    method: str
    contribution: str


class SnippetExtractionResult(BaseModel):
    """Structured response format for snippet extraction"""
    snippets: list[CodeSnippet]
    

class CodeAnalyzer(LazyModelMixin):
    """Encapsulates code file loading and LLM-based analysis methods."""

    # Supported languages
    LANGUAGE_MAP = {
        '.py': 'python',
        ".ipynb": 'jupyter notebook',
        '.js': 'javascript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.ts': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
    }

    def __init__(self, model_name: str = "qwen/qwen3-coder-30b"):
        self.model_name = model_name
        self._model = None  # Lazy-loaded via LazyModelMixin

    @staticmethod
    def load_code_files(folder_path: str, extensions: list[str] | None = None) -> list[UserCode]:
        if extensions is None:
            extensions = list(CodeAnalyzer.LANGUAGE_MAP.keys())
        code_files = []
        folder = Path(folder_path)

        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    code_files.append(UserCode(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_content=content
                    ))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")


        print(f"Loaded {len(code_files)} code file(s)")

        return code_files

    def analyze_code_file(self, code_analysis: UserCode) -> UserCode:
        """Analyze a code file using a single structured LLM call."""
        print(f"Analyzing {code_analysis.file_name}...")
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"

        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are a Senior Research Engineer. Extract the research essence from this code.

            [TASK]
            Analyze the code and output a JSON object.

            [GUIDELINES]
            - **Ignore Boilerplate:** Skip logging, argparse, standard IO.
            - **Be Precise:** Don't say "optimization"; say "AdamW with weight decay".

            [OUTPUT_FORMAT]
            Respond ONLY with valid JSON using exactly these keys:
            {{
                "summary": "2-3 sentences explaining what this file does technically.",
                "keywords": ["list", "of", "algorithms", "libraries", "math_terms", "architectures"],
                "method": "How is it implemented? (e.g., 'Uses a custom collate function with resizing' or 'Standard Transformer Encoder block').",
                "contribution": "What role does this play in the research? (e.g., 'Core training loop' or 'Data augmentation pipeline')."
            }}

            {title_section}[CODE FILE] 
            {code_analysis.file_name}

            [CODE CONTENT]
            ```python
            {code_analysis.file_content}
            ```"""
        )

        result = self.model.respond(
            prompt, 
            response_format=UserCodeAnalysisResult
        )
        # result.parsed is a dict, not the Pydantic model instance
        parsed_dict = result.parsed
        parsed = UserCodeAnalysisResult(**parsed_dict)
        code_analysis.summary = parsed.summary
        code_analysis.keywords = parsed.keywords
        code_analysis.method = parsed.method
        code_analysis.contribution = parsed.contribution

        # Debug output
        print(f"  Keywords: {', '.join(code_analysis.keywords)}")
        print(f"Completed analyzing {code_analysis.file_name}")
        return code_analysis

    def extract_important_snippets(self, code_analysis: UserCode) -> UserCode:
        """Extract important code snippets from a file."""
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"
                
        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are a Technical Editor. Your job is to select code blocks for a paper's "Methodology" section.

            {title_section}[INPUT CONTEXT]
            File: {code_analysis.file_name}
            Core Logic Identified: {code_analysis.method}
            Keywords: {', '.join(code_analysis.keywords)}

            [TASK]
            Extract 1-3 code snippets that mathematically or algorithmically implement the "Core Logic" above.

            [RULES]
            1. **Verbatim:** Copy code exactly. Do not summarize.
            2. **Pure Logic:** Exclude imports, print statements, and error handling.
            3. **Limit:** Maximum 20 lines per snippet. If it's longer, extract the central loop or equation.

            [CODE CONTENT]
            ```python
            {code_analysis.file_content}
            ```
            
            [OUTPUT FORMAT]
            Return a JSON object:
            {{"snippets": [{{"label": "Short Name", "code": "verbatim code here"}}]}}""")

        result = self.model.respond(prompt)
        
        # Parse JSON from content string (structured response has issues with verbatim code)
        content = remove_thinking_blocks(result.content)
        json_match = re.search(r'```(?:json)?\s*(\{{.*?\}})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        parsed_dict = json.loads(content)
        extraction_result = SnippetExtractionResult(**parsed_dict)
        code_analysis.important_snippets = extraction_result.snippets

        print(f"Extracted {len(code_analysis.important_snippets)} code snippet(s)")
        return code_analysis

    def extract_signatures(self, code_analysis: UserCode) -> UserCode:
        """Extract function and class signatures using AST."""
        if not code_analysis.file_content:
            return code_analysis
            
        try:
            tree = ast.parse(code_analysis.file_content)
            signatures = []
            
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    # Extract function signature
                    args = [arg.arg for arg in node.args.args]
                    sig = f"def {node.name}({', '.join(args)})"
                    if node.returns:
                        # Simple attempt to get return type annotation as string
                        try:
                            ret_type = ast.unparse(node.returns)
                            sig += f" -> {ret_type}"
                        except Exception:
                            pass
                    signatures.append(f"Function: {sig}")
                    
                elif isinstance(node, ast.ClassDef):
                    signatures.append(f"Class: {node.name}")
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            args = [arg.arg for arg in item.args.args]
                            # Only include self if it's there
                            args_str = ', '.join(args)
                            sig = f"  - method: {item.name}({args_str})"
                            if item.returns:
                                try:
                                    ret_type = ast.unparse(item.returns)
                                    sig += f" -> {ret_type}"
                                except Exception:
                                    pass
                            signatures.append(sig)
            
            code_analysis.signatures = signatures
            print(f"Extracted {len(signatures)} signatures from {code_analysis.file_name}")
            
        except Exception as e:
            print(f"Error parsing AST for {code_analysis.file_name}: {e}")
            
        return code_analysis

    def analyze_all_files(self, code_files: list[UserCode]) -> list[UserCode]:
        """
        Analyze all code files and extract important code snippets.
        """
        analyzed_files = []
        for code_file in code_files:
            analyzed = self.analyze_code_file(code_file)
            analyzed = self.extract_important_snippets(analyzed)
            analyzed = self.extract_signatures(analyzed)
            analyzed_files.append(analyzed)
        
        print(f"Code analysis complete: analyzed {len(analyzed_files)} file(s)")
        return analyzed_files
    

    @staticmethod
    def get_analysis_report(analyzed_files: list[UserCode]) -> str:
        report = []
        
        for analysis in analyzed_files:
            report.extend([
                f"## File: {analysis.file_name}",
                f"\n**Summary:** {analysis.summary}"
            ])
            
            if analysis.keywords:
                report.append(f"\n**Keywords:** {', '.join(analysis.keywords)}")
            
            if analysis.method:
                report.append(f"\n**Method:** {analysis.method}")
                
            if analysis.contribution:
                report.append(f"\n**Contribution:** {analysis.contribution}")
            
            if analysis.important_snippets:
                report.append(f"\n**Code Snippets ({len(analysis.important_snippets)}):**")
                for snippet in analysis.important_snippets:
                    report.extend([
                        f"\n### {snippet.label}",
                        f"```python\n{snippet.code}\n```"
                    ])
            
            report.append("\n---\n")

        return "\n".join(report)
