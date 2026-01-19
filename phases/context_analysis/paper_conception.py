from dataclasses import dataclass
import textwrap
from typing import List

from phases.context_analysis.user_code_analysis import (
    CodeAnalyzer,
    CodeSnippet,
    UserCode,
)
from phases.context_analysis.user_requirements import UserRequirements
from settings import Settings
from utils.file_utils import load_markdown, save_markdown
from utils.lazy_model_loader import LazyModelMixin
from utils.llm_utils import remove_thinking_blocks


@dataclass
class PaperConcept():
    """Stores paper concept details"""
    description: str = ""
    code_snippets: str = ""  # Markdown-formatted code snippets section (as text)
    open_questions: str = ""

class PaperConception(LazyModelMixin):

    def __init__(self, model_name, user_code: list[UserCode], user_requirements: UserRequirements):
        self.model_name = model_name
        self._model = None  # Lazy-loaded via LazyModelMixin
        self.user_code = user_code
        self.user_requirements = user_requirements

    def _format_user_requirements_section(self) -> str:
        """Format user requirements into a readable section for the LLM prompt."""
        req = self.user_requirements
        
        # Collect all non-empty fields
        sections = []
        
        if req.topic and req.topic.strip():
            sections.append(f"Topic: {req.topic}")
        if req.hypothesis and req.hypothesis.strip():
            sections.append(f"Hypothesis: {req.hypothesis}")
        if req.abstract and req.abstract.strip():
            sections.append(f"Abstract: {req.abstract}")
        if req.introduction and req.introduction.strip():
            sections.append(f"Introduction: {req.introduction}")
        if req.related_work and req.related_work.strip():
            sections.append(f"Related Work: {req.related_work}")
        if req.methods and req.methods.strip():
            sections.append(f"Methods: {req.methods}")
        if req.results and req.results.strip():
            sections.append(f"Results: {req.results}")
        if req.discussion and req.discussion.strip():
            sections.append(f"Discussion: {req.discussion}")
        if req.conclusion and req.conclusion.strip():
            sections.append(f"Conclusion: {req.conclusion}")
        if req.acknowledgements and req.acknowledgements.strip():
            sections.append(f"Acknowledgements: {req.acknowledgements}")
        
        # Return empty string if no requirements provided, otherwise format as a block
        if not sections:
            return ""
        
        return "[User Requirements]\n" + "\n".join(sections)

    def generate_core_information(self) -> PaperConcept:
        
        code_analysis_report = CodeAnalyzer.get_analysis_report(self.user_code)
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"
                
        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are an Expert in advanced scientific research.
            Your task is to distill user notes and code into a **canonical research definition**.
            
            [OBJECTIVE]
            Create a structured "Paper Concept" that serves as the semantic anchor for this research. 
            This output will be embedded to find similar papers. Therefore, it must use precise, standard terminology and avoid conversational fillers.

            [INPUT DATA]
            1. User Hypothesis/Notes
            2. Code Analysis (Ground Truth for implementation details)
            3. Code Snippets (Source of algorithmic logic)

            [STRICT WRITING RULES]
            1. **No Meta-Commentary:** Do NOT write "The user wants," "The code shows," or "I will generate."
            2. **Fact-Based Tone:** Write as if the research already exists. (e.g., "This method leverages X to solve Y.")
            3. **Semantic Density:** Use specific field terminology found in the code (e.g., "Cross-Entropy Loss," "Monte Carlo Tree Search") rather than generic terms (e.g., "Standard Loss," "Search Algorithm").
            4. **Inference:** If the notes are vague, strictly infer the methodology from the logic present in the **Code Snippets**.

            [OUTPUT FORMAT]
            ## 1. Taxonomic Classification
            *Keywords that define the search space.*
            - **Primary Domain:** (e.g., Natural Language Processing)
            - **Specific Task:** (e.g., Low-Resource Machine Translation)
            - **Methodological Class:** (e.g., Transformer-based Sequence-to-Sequence learning)

            ## 2. Abstract & Core Contribution
            *A dense, 4-5 sentence summary. This is the primary vector for embeddings.*
            - **Structure:** [Current Challenge in SOTA] -> [Proposed Method] -> [Mechanism of Action] -> [Expected Outcome].
            - **Requirement:** Mention specific algorithms or architectures identified in the code snippets.

            ## 3. Problem Definition
            *The specific gap this paper fills.*
            - **The Bottleneck:** What specific limitation prevents current methods from succeeding in this context? (e.g., "Vanishing gradients in deep networks," "High computational cost of attention mechanisms").
            - **The Constraint:** Under what conditions does the problem exist?

            ## 4. Technical Approach
            *The "How" - strictly derived from code/notes.*
            - **Architecture:** Define the structural logic (e.g., "A dual-encoder framework...").
            - **Key differentiator:** How does this implementation differ from the standard approach? (e.g., "Replaces standard Softmax with Sparsemax to...").
            
            [USER REQUIREMENTS]
            {self._format_user_requirements_section()}

            {title_section}[CODE ANALYSIS]
            {code_analysis_report}"""
        )

        result = self.model.respond(prompt)
        
        description_text = remove_thinking_blocks(result.content)
        
        return PaperConcept(description=description_text, code_snippets=code_analysis_report)

    def identify_open_questions(self, concept: PaperConcept) -> PaperConcept:
        """
        Analyze the paper concept and identify what information is needed to write
        a high-quality academic paper. These questions will guide literature search.
        """
        
        # Paper title if provided by user
        title_section = ""
        if Settings.LATEX_TITLE and Settings.LATEX_TITLE.strip():
            title_section = f"[PAPER TITLE]\n{Settings.LATEX_TITLE}\n\n"
        
        prompt = textwrap.dedent(f"""\
            [ROLE]
            You are a strategic research advisor who prioritizes questions for maximum research impact.

            [TASK]
            Generate a FOCUSED list of literature search questions to understand the research landscape and strengthen differentiation.
            Prioritize questions that address critical gaps in understanding the field and prior work.

            [ANALYSIS APPROACH]
            1. Identify the MOST CRITICAL gaps in understanding the field and related work
            2. Focus on: (a) existing methods/prior art, (b) how this work differs, (c) key concepts to understand
            3. Questions should guide literature search to establish novelty and context

            [QUESTION PRIORITIES]
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

            [CRITICAL INSTRUCTIONS]
            - Maximum 10 questions total - quality over quantity
            - Group questions by priority (label each group)
            - Be SPECIFIC (e.g., "How does Method X differ from Method Y in aspect Z?" not "What is Method Y?")
            - Focus on what's needed to establish novelty and write a strong related work section
            - Every question should have clear literature search targets
            - Adapt questions to the specific research domain identified in the paper concept

            {title_section}[PAPER CONCEPT TO ANALYZE]
            {concept.description}

            [CODE ANALYSIS]
            {CodeAnalyzer.get_analysis_report(self.user_code)}

            [OUTPUT FORMAT]
            1. question
            2. question
            ...
        """)

        result = self.model.respond(prompt)
        
        questions_text = remove_thinking_blocks(result.content)
        concept.open_questions = questions_text
        
        print(f"Generated open questions for literature search")
        return concept

    def build_paper_concept(self) -> PaperConcept:
        """Build the complete paper concept by generating core information and identifying open questions."""
        print("Generating paper concept...")
        
        concept = self.generate_core_information()
        concept = self.identify_open_questions(concept)

        # Automatically save
        PaperConception.save_paper_concept(concept, filename="paper_concept.md", output_dir="output")

        return concept

    @staticmethod
    def save_paper_concept(concept: PaperConcept, filename: str = "paper_concept.md", output_dir: str = "output") -> str:
        """Save the paper concept to a markdown file with open questions and code snippets. """

        content_parts = []
        
        content_parts.extend([
            "# Paper Concept\n",
            concept.description
        ])
        
        if concept.open_questions:
            content_parts.extend([
                "\n\n",
                "# Open Questions for Literature Search\n",
                concept.open_questions
            ])
        
        if concept.code_snippets:
            content_parts.extend([
                "\n\n",
                "# Important Code Snippets\n",
                concept.code_snippets
            ])
        
        full_content = "\n".join(content_parts)
        file_path = save_markdown(full_content, filename, output_dir)
        print(f"Paper concept saved to: {file_path}")
        
        return file_path

    @staticmethod
    def load_paper_concept(file_path: str) -> PaperConcept:
        """
        Load a paper concept from a saved markdown file.
        Allows users to review and edit the concept before continuing.

        Users can edit all sections directly in the markdown file, as long as the 3 large headers are preserved:
        - Paper concept description
        - Open questions for literature search
        - Code snippets section
        """
        from pathlib import Path

        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Paper concept file not found: {file_path}")

        content = load_markdown(path_obj.name, str(path_obj.parent))
        
        # Parse the markdown content
        description = ""
        open_questions = ""
        code_snippets_section = ""
        
        # Split by main section headers (# at start of line)
        sections = content.split('\n# ')
        
        for section in sections:
            section = section.lstrip('# ')
            
            if section.startswith('Paper Concept'):
                desc_content = section.split('\n', 1)[1] if '\n' in section else ""
                description = desc_content.strip()
                
            elif section.startswith('Open Questions'):
                questions_content = section.split('\n', 1)[1] if '\n' in section else ""
                open_questions = questions_content.strip()
                
            elif section.startswith('Important Code Snippets'):
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
