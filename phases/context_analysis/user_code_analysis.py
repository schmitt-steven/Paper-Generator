import os
import re
import json
import textwrap
from pathlib import Path
from dataclasses import dataclass
from typing import cast
from pydantic import BaseModel
from utils.lazy_model_loader import LazyModelMixin
from utils.llm_utils import remove_thinking_blocks


@dataclass
class UserCode:
    """Stores analysis results for a code file"""
    file_path: str
    file_name: str
    file_content: str
    summary: str = ""
    novel_concepts: str = ""
    research_relevance: str = ""
    important_snippets: list['CodeSnippet'] | None  = None
    
    def __post_init__(self):
        if self.important_snippets is None:
            self.important_snippets = []


class CodeSnippet(BaseModel):
    """Represents an important code snippet extracted from a file"""
    code: str
    explanation: str
    importance_reasoning: str


class UserCodeAnalysisResult(BaseModel):
    """Structured response format used by the LLM"""
    summary: str
    novel_concepts: str
    research_relevance: str


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

        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert in code analysis and scientific literature.

            [TASK]
            Analyze the following code file and provide a structured analysis.

            [OUTPUT_FORMAT]
            You MUST respond with valid JSON containing exactly these three fields:
            {{
                "summary": "Technical description of what the code does and how it works in a few sentences.",
                "novel_concepts": "What makes this code truly novel or innovative? If nothing is novel, use an empty string.",
                "research_relevance": "Based on the novel concepts above, what is the research potential? Otherwise, explain specific academic value in a few sentences. If no research potential, use an empty string."
            }}

            [CODE FILE] 
            {code_analysis.file_name}

            [CODE CONTENT]
            ```
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
        code_analysis.novel_concepts = parsed.novel_concepts
        code_analysis.research_relevance = parsed.research_relevance

        print(f"Completed analyzing {code_analysis.file_name}")
        return code_analysis

    def extract_important_snippets(self, code_analysis: UserCode) -> UserCode:
        """Extract important code snippets from a file that has novel concepts."""
                
        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an expert in code analysis and scientific research.

            [TASK]
            Extract the most important code snippets from this file that demonstrate the novel concepts.

            [CODE SUMMARY]
            {code_analysis.summary}

            [CODE NOVEL CONCEPTS]
            {code_analysis.novel_concepts}

            [CODE RESEARCH RELEVANCE]
            {code_analysis.research_relevance}

            [CODE FILE]
            {code_analysis.file_name}

            [CODE FILE CONTENT]
            ```
            {code_analysis.file_content}
            ```

            [INSTRUCTIONS]
            1. Extract only the most important code snippet(s) that best demonstrate the novel/research-worthy aspects
            2. Copy the code EXACTLY as it appears (verbatim) - preserve all indentation, newlines, and formatting
            4. For each snippet provide:
            - code: The exact code (function, class, or relevant block) with original formatting preserved
            - explanation: What this code does (2-3 sentences)
            - importance_reasoning: Why this specific code is important for research (1-2 sentences)
            5. Prioritize:
            - Core algorithmic implementations
            - Novel data structures or patterns
            - Key architectural decisions
            6. If the file is very long, focus on the most critical snippets
            
            [OUTPUT FORMAT]
            Return your response as a JSON object with this exact structure:
            {{
            "snippets": [
                {{
                "code": "the actual code here",
                "explanation": "what it does",
                "importance_reasoning": "why it matters"
                }}
            ]
            }}"""
        )

        result = self.model.respond(prompt)
        
        # Parse JSON from content string (response schema doesn't work well for code)
        content = remove_thinking_blocks(result.content)
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        parsed_dict = json.loads(content)
        extraction_result = SnippetExtractionResult(**parsed_dict)
        code_analysis.important_snippets = extraction_result.snippets

        print(f"Extracted {len(code_analysis.important_snippets)} code snippet(s)")
        return code_analysis

    def analyze_all_files(self, code_files: list[UserCode]) -> list[UserCode]:
        """
        Analyze all code files and extract important code snippets from files
        that have novel concepts.
        """
        analyzed_files = []
        for code_file in code_files:
            analyzed = self.analyze_code_file(code_file)
            
            # Extract snippets only if novel concepts were found
            if analyzed.novel_concepts.strip():
                analyzed = self.extract_important_snippets(analyzed)
            
            analyzed_files.append(analyzed)
        
        print(f"Code analysis complete: analyzed {len(analyzed_files)} file(s)")
        return analyzed_files
    

    @staticmethod
    def get_analysis_report(analyzed_files: list[UserCode]) -> str:
        report = ["=== Code Analysis Report ===\n"]
        
        for analysis in analyzed_files:
            report.extend([
                f"File: {analysis.file_name}",
                f"Summary:\n{analysis.summary}"
            ])
            if analysis.novel_concepts:
                report.append(f"Novel Concepts:\n{analysis.novel_concepts}")
                
            if analysis.research_relevance:
                report.append(f"Research Relevance:\n{analysis.research_relevance}")
            
            if analysis.important_snippets:
                report.append(f"\nImportant Code Snippets ({len(analysis.important_snippets)}):")
                for i, snippet in enumerate(analysis.important_snippets, 1):
                    report.extend([
                        f"\n  Snippet {i}:",
                        f"  Importance: {snippet.importance_reasoning}",
                        f"  Explanation: {snippet.explanation}",
                        f"  Code:\n```\n{snippet.code}\n```"
                    ])
            
            report.append("==== End of Code Analysis Report ====" + "\n")

        return "\n".join(report)


if __name__ == "__main__":
    code_analyzer = CodeAnalyzer(model_name="qwen/qwen3-coder-30b")
    code_files = code_analyzer.load_code_files("user_files")
    analyzed_files = code_analyzer.analyze_all_files(code_files)
    print(code_analyzer.get_analysis_report(analyzed_files))