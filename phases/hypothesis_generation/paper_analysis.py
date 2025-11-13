import re
import json
import os
import lmstudio as lms
from typing import List
from phases.literature_review.arxiv_api import Paper
from phases.hypothesis_generation.hypothesis_models import PaperFindings, FindingsExtractionResult
from utils.file_utils import preprocess_markdown


class PaperAnalyzer:
    """Analyzes papers to extract key findings using section-aware extraction"""
    
    def __init__(self, model_name: str):
        self.model = lms.llm(model_name)

    def extract_findings(self, papers: List[Paper]) -> List[PaperFindings]:
        """
        Extract findings from a list of papers sequentially.
        Shows progress bar.
        """
        findings = []
        total = len(papers)

        print(f"\nExtracting findings from {total} papers...")
        for i, paper in enumerate(papers, 1):
            # Extract clean ID for display (e.g., "http://arxiv.org/abs/2301.12345v1" -> "2301.12345v1")
            paper_id = paper.id.split('/')[-1] if '/' in paper.id else paper.id
            print(f"  [{i}/{total}] Processing {paper_id}...")
            finding = self.extract_paper_findings(paper)
            findings.append(finding)
        return findings
    
    def extract_paper_findings(self, paper: Paper) -> PaperFindings:
        """
        Extract key findings from a single paper using LLM.
        
        Returns structured PaperFindings with methods, datasets, results, limitations.
        """
        # Extract clean ID (e.g., "http://arxiv.org/abs/2301.12345v1" -> "2301.12345v1")
        paper_id = paper.id.split('/')[-1] if '/' in paper.id else paper.id
        
        key_sections = self.extract_key_sections(paper)
        
        if not key_sections.strip():
            # Return empty findings if no content
            return PaperFindings(
                paper_id=paper_id,
                title=paper.title,
                methods_used=[],
                test_setup="",
                main_limitations=""
            )
        
        prompt = f"""Extract key information from this research paper.
        Paper Title: {paper.title}

        Paper Content:
        {key_sections[:10000]}

        Extract ONLY the following (be concise and accurate):
        1. methods_used: List of method names mentioned (e.g., ["Q-learning", "DQN"])
        2. test_setup: Single string describing HOW the method was tested/evaluated (e.g., "simulation environments", "gridworld navigation", "Atari games", "real-world robotics", "theoretical analysis", "benchmark datasets")
        3. main_limitations: Single string describing the key limitation(s) mentioned (combine multiple limitations into one coherent string)

        If a field has no information, use empty list for lists or empty string for strings. DO NOT HALLUCINATE OR MAKE UP DATA."""

        try:
            result = self.model.respond(
                prompt, 
                response_format=FindingsExtractionResult,
                config={"temperature": 0.3}
            )
            
            # LLM always returns dict, never an object
            extraction_result = result.parsed
            
            # Convert to full PaperFindings 
            findings = PaperFindings(
                paper_id=paper_id,
                title=paper.title,
                methods_used=extraction_result.get("methods_used", []),
                test_setup=extraction_result.get("test_setup", ""),
                main_limitations=extraction_result.get("main_limitations", "")
            )
            return findings
        except Exception as e:
            print(f"Failed to extract findings from {paper_id}: {e}")
            return PaperFindings(
                paper_id=paper_id,
                title=paper.title,
                methods_used=[],
                test_setup="",
                main_limitations=""
            )

    def extract_key_sections(self, paper: Paper) -> str:
        """
        Extract key sections from paper for analysis.
        
        Extracts and combines:
        - Abstract (from paper.summary)
        - Methods (~2000 chars)
        - Results (~1500 chars)
        - Limitations (~1000 chars)
        
        Returns: Combined text (~5000 chars) with most relevant content for analysis
        """
        # Use abstract from paper summary
        abstract = paper.summary if paper.summary else ""
        
        # Try to get markdown from paper object
        if paper.markdown_text:
            markdown = paper.markdown_text
        else:
            # If no markdown, just return the abstract
            return abstract
        
        if not markdown:
            return abstract
        
        # Preprocess markdown to remove gibberish that was generated during conversion from PDF to Markdown
        markdown = preprocess_markdown(markdown)
        
        # Extract methods section
        method_keywords = ["method", "approach", "architecture", "model", "framework", "algorithm"]
        methods = self.extract_section_by_keywords(markdown, method_keywords, max_chars=2000)
        
        # Extract results section
        result_keywords = ["result", "experiment", "evaluation", "performance", "finding"]
        results = self.extract_section_by_keywords(markdown, result_keywords, max_chars=1500)
        
        # Extract limitations section
        limitation_keywords = ["limitation", "future work", "discussion", "conclusion"]
        limitations = self.extract_section_by_keywords(markdown, limitation_keywords, max_chars=1000)
        
        # Combine sections
        combined_sections = f"{abstract}\n\n{methods}\n\n{results}\n\n{limitations}"
        
        return combined_sections
    
    def extract_section_by_keywords(self, markdown_text: str, keywords: List[str], max_chars: int = 2000) -> str:
        """
        Extract a section from markdown text using comprehensive pattern matching.
        
        Supports different formatting styles:
        - Markdown headers: ## Methods, ### 3. Methodology
        - Bold numbered: **1. Introduction**, **3. Methods**
        - Separate bold: **1** **Introduction**, **2** **Methods**
        - Plain bold: **Introduction**, **Abstract**
        - Roman numerals + caps: I. INTRODUCTION, IV. M ETHODOLOGY
        - Numbers + spaced caps: 1 I NTRODUCTION, 2 M ETHODOLOGY
        - Italic subsections: _1.1_ _Background_, _A. Framework_
        - Period-only: . Intro duction (rare)
        
        Falls back to keyword density search if no headers found.
        """
        # Build flexible regex patterns (case-insensitive)
        # Handle keywords with spaces (e.g., "RELATED WORK" or "R ELATED W ORK")
        keyword_pattern = "|".join(re.escape(k) for k in keywords)
        keyword_pattern_spaced = "|".join(r"\s*".join(re.escape(c) for c in k) for k in keywords)
        
        patterns = [
            # Markdown headers with optional numbering and "our" (e.g., "## 3 Methods", "# Methods")
            rf"#{1,3}\s*\d*\.?\s*(?:our\s+)?(?:{keyword_pattern})s?",
            # Numbers + spaced all caps (e.g., "1 I NTRODUCTION", "2 M ETHODOLOGY")
            rf"^\d+\.?\s+(?:{keyword_pattern_spaced})",
            # Roman numerals + spaced all caps (e.g., "IV. M ETHODOLOGY", "II. RELATED WORK")
            rf"^[IVX]+\.\s+(?:{keyword_pattern_spaced})",
            # Bold numbered with period (e.g., "**1. Introduction**")
            rf"^\*\*\d+\.?\s+(?:{keyword_pattern})s?\*\*",
            # Separate bold number + bold title (e.g., "**1** **Introduction**")
            rf"^\*\*\d+\.?\d*\s*\*\*\s*\*\*(?:{keyword_pattern})s?\*\*",
            # Plain bold without numbers (e.g., "**Introduction**", "**Abstract**")
            rf"^\*\*(?:{keyword_pattern})s?\*\*",
            # Italic subsections with numbering (e.g., "_1.1_ _Existing surveys_" or "_A. Framework_")
            rf"^_(?:\d+\.\d+|[A-Z])\.\s+(?:{keyword_pattern})s?_",
            # Period-only numbered (rare, e.g., ". Intro duction")
            rf"^\.\s+(?:{keyword_pattern_spaced})",
            # Numbered sections in text (e.g., "\n3. Methodology")
            rf"\n\d+\.?\s+(?:{keyword_pattern})s?"
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = list(re.finditer(pattern, markdown_text, re.IGNORECASE | re.MULTILINE))
            if matches:
                # Get the first match
                match = matches[0]
                start = match.end()
                
                # Find next header (any of the patterns)
                next_header_pattern = r"(?:#{1,3}\s|\*\*|\n\d+\.)"
                next_match = re.search(next_header_pattern, markdown_text[start:start+max_chars])
                
                if next_match:
                    end = start + next_match.start()
                else:
                    end = start + max_chars
                
                return markdown_text[start:end].strip()
        
        # Fallback: keyword density search
        return self._extract_by_keyword_density(markdown_text, keywords, max_chars)
    
    def _extract_by_keyword_density(self, markdown_text: str, keywords: List[str], max_chars: int) -> str:
        """
        Extract section by finding the window with highest keyword density.
        Uses sliding windows of 500 chars.
        """
        window_size = 500
        best_window = ""
        best_density = 0
        
        for i in range(0, len(markdown_text) - window_size, 100):
            window = markdown_text[i:i+window_size]
            # Count keyword occurrences (case-insensitive)
            density = sum(window.lower().count(kw.lower()) for kw in keywords)
            
            if density > best_density:
                best_density = density
                best_window = window
        
        # Extend to max_chars if possible
        if best_window and len(best_window) < max_chars:
            start_idx = markdown_text.find(best_window)
            if start_idx != -1:
                best_window = markdown_text[start_idx:start_idx + max_chars]
        
        return best_window.strip() if best_window else markdown_text[:max_chars]
    
    
    @staticmethod
    def save_findings(findings: List[PaperFindings], filepath: str = "output/paper_findings.json"):
        """Save paper findings to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        findings_data = [
            {
                "paper_id": f.paper_id,
                "title": f.title,
                "methods_used": f.methods_used,
                "test_setup": f.test_setup,
                "main_limitations": f.main_limitations
            }
            for f in findings
        ]
        
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(findings_data, file, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(findings)} paper findings to {filepath}")
    
    @staticmethod
    def load_findings(filepath: str) -> List[PaperFindings]:
        """Load paper findings from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        findings = [PaperFindings(**item) for item in data]
        print(f"Loaded {len(findings)} paper findings from {filepath}")
        return findings

    

