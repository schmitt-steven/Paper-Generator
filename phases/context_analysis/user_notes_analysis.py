import textwrap
import lmstudio as lms
from pathlib import Path
from dataclasses import dataclass
from typing import List
from lmstudio import BaseModel



@dataclass
class UserNotes:
    """Stores analysis results from a document"""
    file_path: str
    file_name: str
    file_content: str
    summary: str = ""
    key_findings: str = ""
    methodologies: str = ""
    technical_details: str = ""
    data_and_results: str = ""
    related_work: str = ""


class NotesAnalysisResult(BaseModel):
    """Structured response format used by the LLM"""
    summary: str
    key_findings: str
    methodologies: str
    technical_details: str
    data_and_results: str
    related_work: str


class NotesAnalyzer:
    """Analyzes documents to extract research-relevant information"""

    SUPPORTED_EXTENSIONS = {
        '.txt': 'plain text',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.rst': 'reStructuredText',
        '.tex': 'LaTeX',
        '.org': 'org-mode',
    }

    def __init__(self, model_name: str = "qwen/qwen3-coder-30b"):
        self.model = lms.llm(model_name)

    def load_user_notes(self, folder_path: str, extensions: List[str] = None) -> List[UserNotes]:
        """Load documents from the specified folder."""
        if extensions is None:
            extensions = list(self.SUPPORTED_EXTENSIONS.keys())
        
        files = []
        folder = Path(folder_path)

        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    files.append(UserNotes(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_content=content
                    ))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        print(f"Loaded {len(files)} user note(s)")
        return files

    def analyze_user_note(self, user_notes: UserNotes) -> UserNotes:
        """Analyze a document to extract research-relevant information using structured LLM analysis."""

        prompt = textwrap.dedent(f"""\
            You are an expert research analyst helping to analyze documents for automated research paper generation.
            <task>
            Analyze the following document and extract relevant information in a structured format.
            First, provide a brief summary covering the main topic and key points of the entire document.
            Put this summary in the field named `summary` in the structured response.

            Then for each field below, COPY EXACT VERBATIM passages from the document (do NOT summarize or paraphrase).
            IMPORTANT: Extract ALL relevant information from the document. If information doesn't fit into any field, place it in the CLOSEST matching field.
            </task>

            <analysis_framework>
            The structured response must contain the following fields:
            1. key_findings: Copy exact quotes about main discoveries, results, conclusions, or insights.
            - What is being researched? is there a research question or hypothesis?
            - What problems does it solve?
            - What are the important takeaways?
            - If no significant findings, use an empty string.

            2. methodologies: Copy exact descriptions of approaches, algorithms, techniques, or experimental methods.
            - How were things implemented or tested?
            - What frameworks or tools were used?
            - If no methodologies, use an empty string."

            3. technical_details: Copy exact implementation specifics, parameters, formulas, or specifications.
            - Important parameters, configurations, or design choices
            - Mathematical formulations if present
            - If minimal technical content, use an empty string.

            4. data_and_results: Copy exact datasets used, experimental results, performance metrics, or quantitative findings.
            - What data was analyzed?
            - What were the outcomes or measurements?
            - If no data/results, use an empty string.

            5. related_work: Copy exact references, related work, or external sources mentioned.
            - Author names, paper titles, or sources referenced
            - If no citations, use an empty string.
            </analysis_framework>

            <document_information>
            Document: {user_notes.file_name}
            Content:
            ```
            {user_notes.file_content}
            ```
            </document_information>
        """)

        result = self.model.respond(
            prompt,
            response_format=NotesAnalysisResult
        ).parsed
        
        analysis = NotesAnalysisResult(**result)        
        user_notes.summary = analysis.summary
        user_notes.key_findings = analysis.key_findings
        user_notes.methodologies = analysis.methodologies
        user_notes.technical_details = analysis.technical_details
        user_notes.data_and_results = analysis.data_and_results
        user_notes.related_work = analysis.related_work

        print(f"Completed analyzing {user_notes.file_name}")
        return user_notes

    def analyze_all_user_notes(self, user_notes: List[UserNotes]) -> List[UserNotes]:
        """Analyze all loaded documents."""
        analyzed_documents = []
        for document in user_notes:
            analyzed = self.analyze_user_note(document)
            analyzed_documents.append(analyzed)
        return analyzed_documents

    @staticmethod
    def get_analysis_report(analyzed_notes: List[UserNotes]) -> str:
        report = "=== User Notes Analysis Report ===\n"
        for doc in analyzed_notes:
            report += f"File: {doc.file_path}\n"
            if doc.summary:
                report += f"Summary:\n{doc.summary}\n"
            if doc.key_findings:
                report += f"Key Findings:\n{doc.key_findings}\n"
            if doc.methodologies:
                report += f"Methodologies:\n{doc.methodologies}\n"
            if doc.technical_details:
                report += f"Technical Details:\n{doc.technical_details}\n"
            if doc.data_and_results:
                report += f"Data & Results:\n{doc.data_and_results}\n"
            if doc.related_work:
                report += f"Relevant Citations:\n{doc.related_work}\n"
        return report


if __name__ == "__main__":
    # Example usage for research paper generation framework
    analyzer = NotesAnalyzer(model_name="qwen/qwen3-coder-30b")
    
    # Load all documents from the research folder
    notes = analyzer.load_user_notes("user_files")
    
    # Analyze all documents
    analyzed_notes = analyzer.analyze_all_user_notes(notes)
    
    print(analyzer.get_analysis_report(analyzed_notes))