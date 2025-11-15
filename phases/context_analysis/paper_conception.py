from dataclasses import dataclass
import textwrap
import lmstudio as lms
from typing import List
from phases.context_analysis.user_code_analysis import CodeAnalyzer, UserCode, CodeSnippet
from phases.context_analysis.user_notes_analysis import NotesAnalyzer, UserNotes 
from utils.file_utils import save_markdown_to_file 


@dataclass
class PaperConcept():
    """Stores paper concept details"""
    description: str = ""
    code_snippets: str = ""  # Markdown-formatted code snippets section (as text)
    open_questions: str = ""

class PaperConception:

    def __init__(self, model_name, user_code: List[UserCode], user_notes: List[UserNotes]):
        self.model = lms.llm(model_name)
        self.user_code = user_code
        self.user_notes = user_notes

    def _format_code_snippets_section(self) -> str:
        """Extract and format code snippets prominently for the LLM."""
        snippets_text = []
        
        for code_file in self.user_code:
            if code_file.important_snippets:
                snippets_text.append(f"\n## From: {code_file.file_name}")
                snippets_text.append(f"\n**Novel Concepts:** {code_file.novel_concepts}\n")
                
                for i, snippet in enumerate(code_file.important_snippets, 1):
                    snippets_text.extend([
                        f"\n### Snippet {i}\n",
                        f"**Why Important:**  \n{snippet.importance_reasoning}\n",
                        f"**What It Does:**  \n{snippet.explanation}\n",
                        f"**Code:**\n```python\n{snippet.code}\n```\n"
                    ])
        
        return "\n".join(snippets_text) if snippets_text else "[No code snippets extracted]"

    def generate_core_information(self) -> PaperConcept:
        
        code_snippets_section = self._format_code_snippets_section()
        
        prompt = textwrap.dedent(f"""\
            You are a critical research advisor with expertise in academic rigor, novelty assessment, and peer review standards.

            TASK:
            Generate a rigorous paper concept that identifies core ideas, gaps, and research direction.
            Be CRITICAL and DEMANDING. Do not accept vague claims or weak differentiation.

            CRITICAL SECTIONS (in order):
            1. Paper Specifications
            2. Research Topic
            3. Research Field
            4. Problem Statement
            5. Motivation
            6. Novelty & Differentiation
            7. Methodology & Implementation (High-Level)
            8. Expected Contribution
            
            Note: We are at the CONCEPT stage - no need for detailed proofs, experimental designs, or final paper titles yet.

            STRICT REQUIREMENTS FOR EACH SECTION:

            1. PAPER SPECIFICATIONS
            - Extract all metadata (type, length, audience, style, figures/tables)
            - If missing specific items, state: "[Missing: specific item name - needed for X reason]"

            2. RESEARCH TOPIC
            - Briefly describe the general topic/area of research (1-2 sentences)
            - This is NOT the final paper title, just the subject matter

            3. RESEARCH FIELD
            - Identify the primary field and relevant subfields
            - State standard terminology if applicable

            4. PROBLEM STATEMENT
            - Must be SPECIFIC, not generic (e.g., "scalability issues" or "efficiency problems" are too vague)
            - Must quantify inefficiencies or failure modes with concrete examples
            - Must clearly scope the problem domain and constraints
            - If vague or missing, state: "[Missing: quantifiable problem definition - current description too broad]"

            5. MOTIVATION
            - Why is this problem important to solve?
            - What are the implications or applications?

            6. NOVELTY & DIFFERENTIATION
            **CRITICAL: This is where most papers fail.**
            - Explicitly compare to existing methods in the field
            - State: "This differs from [Method X] because [specific technical difference]"
            - If code/notes don't differentiate from existing work, state: "[Missing: differentiation from existing methods - must explain specific advantages]"
            - Do NOT claim novelty without clear differentiation from prior art

            7. METHODOLOGY & IMPLEMENTATION (High-Level)
            - Describe the approach at a high level
            - Reference CODE SNIPPETS below for key implementation insights
            - Identify if mathematical formulation is present or missing
            - If critical details missing, state: "[Missing: X - needed for Y]"

            8. EXPECTED CONTRIBUTION
            - Must be concrete and measurable
            - Avoid vague claims like "improves efficiency"
            - State specific advantages with conditions (e.g., "faster convergence in sparse reward settings")

            CRITICAL INSTRUCTIONS:
            - **Be BRUTAL**: If information is vague, mark it as insufficient
            - **Demand precision**: Generic claims → demand specific examples and scope
            - **Require differentiation**: Always compare to existing methods in the field
            - **Specify gaps**: Don't just write "[Missing information]" - write "[Missing: X because Y]"
            - **NO INVENTED DATA**: Do NOT make up percentages, specific metrics, or quantitative results
            - Use qualitative comparisons: "faster", "more accurate", "scales better" instead of "20% faster"
            - When writing Novelty, ask: "How is this different from existing state-of-the-art methods?"
            - Extract domain/field from the notes/code and tailor analysis accordingly
            - Focus on CONCEPT quality, not full paper details

            OUTPUT FORMAT:
            - Use ## for section headings (e.g., "## 1. Paper Specifications")
            - Use ### for subsections if needed
            - Do NOT use horizontal rules (---) between sections
            - Use bullet points (-) for lists
            - Keep formatting clean and consistent

            ═══════════════════════════════════════════════════════════════
            USER NOTES ANALYSIS
            ═══════════════════════════════════════════════════════════════
            {NotesAnalyzer.get_analysis_report(self.user_notes)}

            ═══════════════════════════════════════════════════════════════
            FULL CODE ANALYSIS
            ═══════════════════════════════════════════════════════════════
            {CodeAnalyzer.get_analysis_report(self.user_code)}

            ═══════════════════════════════════════════════════════════════
            CODE SNIPPETS (Priority Information - Use These in Methodology)
            ═══════════════════════════════════════════════════════════════
            {code_snippets_section}
        """)

        result = self.model.respond(prompt)
        
        # Convert result to string (for plain text responses without response_format)
        description_text = str(result.content) if hasattr(result, 'content') else str(result)
        
        # Format code snippets as markdown string
        code_snippets_text = self._format_code_snippets_section()
        
        return PaperConcept(description=description_text, code_snippets=code_snippets_text)

    def identify_open_questions(self, concept: PaperConcept) -> PaperConcept:
        """
        Analyze the paper concept and identify what information is needed to write
        a high-quality academic paper. These questions will guide literature search.
        """
        
        prompt = textwrap.dedent(f"""\
            You are a strategic research advisor who prioritizes questions for maximum research impact.

            TASK:
            Generate a FOCUSED list of literature search questions to understand the research landscape and strengthen differentiation.
            Prioritize questions that address critical gaps in understanding the field and prior work.

            ANALYSIS APPROACH:
            1. Identify the MOST CRITICAL gaps in understanding the field and related work
            2. Focus on: (a) existing methods/prior art, (b) how this work differs, (c) key concepts to understand
            3. Questions should guide literature search to establish novelty and context

            QUESTION PRIORITIES:

            **Priority 1: Related Work & Prior Art**
            - What existing methods in this field address similar problems?
            - What are the standard/state-of-the-art approaches?
            - What are their key strengths and limitations?
            Focus: 4-6 questions to map the research landscape

            **Priority 2: Differentiation & Positioning**
            - How does this approach differ technically from each major baseline?
            - What are the specific advantages/disadvantages vs. existing methods?
            - Where does this fit in the taxonomy of approaches?
            Focus: 2-4 questions to establish clear differentiation

            **Priority 3: Key Concepts & Background**
            - What theoretical frameworks or mathematical tools are relevant?
            - What domain-specific knowledge is needed to understand the approach?
            - What terminology and definitions are standard in this field?
            Focus: 2-3 questions on foundational understanding

            CRITICAL INSTRUCTIONS:
            - Maximum 10 questions total - quality over quantity
            - Group questions by priority (label each group)
            - Be SPECIFIC (e.g., "How does Method X differ from Method Y in aspect Z?" not "What is Method Y?")
            - Focus on what's needed to establish novelty and write a strong related work section
            - Every question should have clear literature search targets
            - Adapt questions to the specific research domain identified in the paper concept

            PAPER CONCEPT TO ANALYZE:
            {concept.description}

            CODE SNIPPETS AVAILABLE:
            {self._format_code_snippets_section()}

            OUTPUT FORMAT:
            1. [question]
            2. [question]
            ...
        """)

        result = self.model.respond(prompt)
        
        # Convert result to string (for plain text responses without response_format)
        questions_text = str(result.content) if hasattr(result, 'content') else str(result)
        concept.open_questions = questions_text
        
        print(f"Generated open questions for literature search")
        return concept

    def build_paper_concept(self) -> PaperConcept:
        """Build the complete paper concept by generating core information and identifying open questions."""
        print("Generating paper concept...")
        
        concept = self.generate_core_information()
        concept = self.identify_open_questions(concept)
        
        # Automatically save
        self.save_paper_concept(concept, filename="paper_concept.md", output_dir="output")
        
        return concept

    def save_paper_concept(self, concept: PaperConcept, filename: str = "paper_concept.md", output_dir: str = "output") -> str:
        """
        Save the paper concept to a markdown file with open questions and code snippets.
        
        Args:
            concept: The PaperConcept to save
            filename: Name of the output file (default: "paper_concept.md")
            output_dir: Directory to save the file (default: "output")
            
        Returns:
            str: Path to the saved file
        """
        content_parts = []
        
        # Paper Concept/Outline section
        content_parts.extend([
            "# Paper Concept\n",
            concept.description
        ])
        
        # Open Questions section
        if concept.open_questions:
            content_parts.extend([
                "\n\n",
                "# Open Questions for Literature Search\n",
                concept.open_questions
            ])
        
        # Important Code Snippets section
        if concept.code_snippets:
            content_parts.extend([
                "\n\n",
                "# Important Code Snippets\n",
                concept.code_snippets
            ])
        
        full_content = "\n".join(content_parts)        
        file_path = save_markdown_to_file(full_content, filename, output_dir)
        print(f"Paper concept saved to: {file_path}")
        
        return file_path

    @staticmethod
    def load_paper_concept(file_path: str) -> PaperConcept:
        """
        Load a paper concept from a saved markdown file.
        Allows users to review and edit the concept before continuing.
        
        Args:
            file_path: Path to the saved paper concept markdown file
            
        Returns:
            PaperConcept object with description, open_questions, and code_snippets (as markdown text) loaded from file
            
        Note:
            Users can edit all sections directly in the markdown file:
            - Paper concept description (main research content)
            - Open questions for literature search
            - Code snippets section (entire section as editable markdown text)
            
            This simplified approach makes loading/editing much easier since code snippets
            are stored as markdown text rather than parsed into structured objects.
        """
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Paper concept file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the markdown content
        description = ""
        open_questions = ""
        code_snippets_section = ""
        
        # Split by main section headers (# at start of line)
        sections = content.split('\n# ')
        
        for section in sections:
            # Handle the first section which may start with # (no \n before it)
            section = section.lstrip('# ')
            
            if section.startswith('Paper Concept'):
                # Extract everything after "Paper Concept" until the next section
                desc_content = section.split('\n', 1)[1] if '\n' in section else ""
                description = desc_content.strip()
                
            elif section.startswith('Open Questions'):
                # Extract everything after "Open Questions for Literature Search" until the next section
                questions_content = section.split('\n', 1)[1] if '\n' in section else ""
                open_questions = questions_content.strip()
                
            elif section.startswith('Important Code Snippets'):
                # Extract code snippets section
                snippets_content = section.split('\n', 1)[1] if '\n' in section else ""
                code_snippets_section = snippets_content.strip()
        
        # Clean up description - remove the "Important Code Snippets" section if it got included
        if '# Important Code Snippets' in description:
            description = description.split('# Important Code Snippets')[0].strip()
        
        print(f"Loaded paper concept from: {file_path}")
        print(f"  - Description: {len(description)} characters")
        print(f"  - Open Questions: {len(open_questions)} characters")
        print(f"  - Code Snippets: {len(code_snippets_section)} characters")
        
        return PaperConcept(
            description=description,
            open_questions=open_questions,
            code_snippets=code_snippets_section
        )
    
    @staticmethod
    def print_paper_concept(concept: PaperConcept):
        print("=== Paper Concept ===")
        print(f"Description:\n{concept.description}")
        print(f"\nCode Snippets ({len(concept.code_snippets)} chars):")
        print(concept.code_snippets[:500] + "..." if len(concept.code_snippets) > 500 else concept.code_snippets)
        print(f"\nOpen Questions:\n{concept.open_questions}")


if __name__ == "__main__":
    # Mock analyzed code
    mock_code = UserCode(
        file_path="test.py",
        file_name="test.py",
        file_content="# test code",
        summary="Implements recursive backward Q-learning algorithm",
        novel_concepts="Novel backward propagation through state graph",
        research_relevance="Improves sample efficiency in sparse reward environments"
    )
    
    # Mock analyzed notes
    mock_notes = UserNotes(
        file_path="notes.md",
        file_name="notes.md", 
        file_content="# Research notes",
        summary="Proposes Recursive Backwards Q-Learning (RBQ)",
        key_findings="Standard Q-learning is slow in sparse reward settings",
        methodologies="Backward pass through trajectories after each episode",
        technical_details="Uses BFS for reward propagation",
        data_and_results="",
        related_work=""
    )
    
    # Test paper conception
    conception = PaperConception(
        model_name="qwen/qwen3-coder-30b",
        user_code=[mock_code],
        user_notes=[mock_notes]
    )
    
    concept = conception.build_paper_concept()
    conception.print_paper_concept(concept)
    
    # Save to markdown file
    file_path = conception.save_paper_concept(concept, filename="test_paper_concept.md")
    
    # Example: Load the concept back from file
    print("\n" + "="*80)
    print("Testing load_paper_concept...")
    print("="*80)
    loaded_concept = PaperConception.load_paper_concept(file_path)
    print("\nLoaded concept summary:")
    print(f"Description starts with: {loaded_concept.description[:100]}...")
    print(f"Open questions: {loaded_concept.open_questions[:100] if loaded_concept.open_questions else 'None'}...")
    print(f"Code snippets: {len(loaded_concept.code_snippets)} characters")