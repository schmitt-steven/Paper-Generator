import re
import json
from pathlib import Path
from typing import Any


def save_markdown(content: str, filename: str, output_dir: str = None) -> str:
    """Save markdown content to a file."""

    if output_dir:
        target_dir = Path(output_dir)
    else:
        # Use current working directory
        target_dir = Path.cwd()

    # Create dir if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return str(file_path)


def load_markdown(filename: str, input_dir: str = None) -> str:
    """Load markdown content from a file."""

    if input_dir:
        source_dir = Path(input_dir)
    else:
        # Use current working directory
        source_dir = Path.cwd()

    file_path = source_dir / filename

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def save_json(data: dict | list | Any, filename: str, output_dir: str = None, indent: int = 2, ensure_ascii: bool = False) -> str:
    """Save JSON data to a file."""

    if output_dir:
        target_dir = Path(output_dir)
    else:
        # Use current working directory
        target_dir = Path.cwd()

    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    return str(file_path)


def load_json(filename: str, input_dir: str = None) -> dict | list | Any:
    """Load JSON data from a file."""

    if input_dir:
        source_dir = Path(input_dir)
    else:
        # Use current working directory
        source_dir = Path.cwd()

    file_path = source_dir / filename

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def preprocess_markdown(markdown: str) -> str:
    """Remove common PDF-to-markdown conversion artifacts and gibberish."""

    if not markdown:
        return ""
    
    # Remove excessive whitespace (3+ newlines → 2)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # Remove broken table artifacts (lines with only pipes and dashes)
    markdown = re.sub(r'^[\|\-\s]+$', '', markdown, flags=re.MULTILINE)
    
    # Remove isolated reference markers (e.g., "[1]" "[2]" on their own or in clusters)
    markdown = re.sub(r'(?:\[\d+\]\s*){3,}', '', markdown)  # Clusters of 3+ references
    
    # Remove garbled LaTeX (unmatched backslashes and brackets)
    markdown = re.sub(r'\\+[a-z]{0,2}(?![a-zA-Z])', '', markdown)  # Isolated backslashes
    
    # Remove repeated special characters (e.g., "- - - - -" or "* * * *")
    markdown = re.sub(r'([\-\*_=])\s*\1{4,}', '', markdown)
    
    # Remove single characters on their own lines (probably OCR artifacts)
    markdown = re.sub(r'^\s*[a-zA-Z0-9]\s*$', '', markdown, flags=re.MULTILINE)
    
    # Remove likely page numbers (isolated small numbers, usually page numbers)
    # Only remove if it's a small number (page numbers) and not part of content
    markdown = re.sub(r'^\s*\d{1,3}\s*$', '', markdown, flags=re.MULTILINE)
    
    # Remove header/footer patterns: only "Page X" (without colon/description)
    # Preserve figure/table captions which usually have colons and descriptions
    markdown = re.sub(r'(?i)^\s*page\s+\d+\s*$', '', markdown, flags=re.MULTILINE)
    
    # Clean up spacing around punctuation
    markdown = re.sub(r'\s+([.,;:!?])', r'\1', markdown)
    
    # Remove excessive spaces
    markdown = re.sub(r' {2,}', ' ', markdown)
    
    return markdown.strip()


def remove_references_section(text: str) -> str:
    """Remove references, acknowledgments, and bibliography sections from text.
    
    Should be called after PDF conversion to clean the stored markdown.
    """
    if not text:
        return ""
    
    # Pattern matches common reference section headers (case-insensitive, with optional markdown formatting)
    # Matches: REFERENCES, **REFERENCES**, 6. REFERENCES, **6** **REFERENCES**, **7. References**
    # Regex:
    # ^\s*(?:#+\s*)?           : Start of line, optional hash headers
    # (?:\*\*)?                : Optional bold start (global or number)
    # (?:[\d\.]+\s+)?          : Optional numbering (e.g. "6. " or "6 ")
    # (?:\*\*)?                : Optional bold end (if number was bolded separately)
    # (?:\s+)?                 : Optional space
    # (?:\*\*)?                : Optional bold start (if title bolded separately)
    # (?:REFERENCES?|ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS?|BIBLIOGRAPHY) : Keyword
    # (?:[:\.]|(?:\s.*))?      : Optional colon/period or trailing text
    # (?:\*\*)?                : Optional bold end
    # \s*$                     : End of line
    
    # Simplified flexible regex:
    # Allows for optional bolding wrapping the whole thing or parts
    pattern = r'^\s*(?:#+\s*)?(?:\*\*)?(?:[\d\.]+\s+)?(?:\*\*)?(?:\s+)?(?:\*\*)?(?:REFERENCES?|ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS?|BIBLIOGRAPHY)(?:[:\.]|(?:\s.*))?(?:\*\*)?\s*$'
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(pattern, line, re.IGNORECASE):
            # Truncate at this line
            return '\n'.join(lines[:i]).strip()
    
    return text


def extract_conclusion(text: str) -> str:
    """Extract the Conclusion section from the markdown text."""
    import re
    
    if not text:
        return ""
        
    # Pattern to find Conclusion section headers
    # Handles various formats in real papers:
    # - **6** **Conclusion**
    # - **Conclusions**
    # - **6. Closing Remarks** 
    # - IV. CONCLUSION 
    # - **Conclusion** 
    # - 7 CONCLUSION 
    # - ### **6 Conclusion** 
    # - 6. Closing Remarks
    # - DISCUSSION AND CONCLUSION
    #
    # Regex:
    # ^\s*                           : Start of line, optional whitespace
    # (?:#+\s*)?                     : Optional markdown headers (###)
    # (?:\*\*)?                      : Optional bold start
    # (?:                            : Section number group:
    #   (?:[IVXivx]+\.?\s*)|         :   Roman numerals (IV., V, etc.) OR
    #   (?:\d+\.?\s*)                :   Arabic numerals (6., 7, etc.)
    # )?
    # (?:\*\*)?\s*                   : Optional bold end for number + spaces
    # (?:\*\*)?                      : Optional bold start for title
    # (?:DISCUSSION\s+AND\s+)?       : Optional "Discussion and" prefix
    # (?:CONCLUSIONS?|CONCLUDING\s+REMARKS?|CLOSING\s+REMARKS?)  : Main keywords
    # (?:[:\.]|(?:\s[^\n]*))?        : Optional colon/period or trailing text
    # (?:\*\*)?                      : Optional bold end
    pattern = r'^\s*(?:#+\s*)?(?:\*\*)?(?:(?:[IVXivx]+\.?\s*)|(?:\d+\.?\s*))?(?:\*\*)?\s*(?:\*\*)?(?:DISCUSSION\s+AND\s+)?(?:CONCLUSIONS?|CONCLUDING\s+REMARKS?|CLOSING\s+REMARKS?)(?:[:\.\s][^\n]*)?(?:\*\*)?\s*$'
    
    lines = text.split('\n')
    start_index = -1
    
    # 1. Find start of Conclusion section
    for i, line in enumerate(lines):
        # Skip if it looks like a TOC entry (contains ".......")
        if "....." in line:
            continue
            
        if re.match(pattern, line, re.IGNORECASE):
            start_index = i
            # Don't break immediately, we want the last occurrence
            
    if start_index == -1:
        return ""
        
    # 2. Extract content
    conclusion_lines = []
    # Skip the header itself
    for line in lines[start_index + 1:]:
        # Stop at the next section header (start with # or **, or bold numbering)
        if re.match(r'^\s*#+\s+[A-Z]', line):
             break
        
        # Stop at likely next header (e.g. **References**)
        # Using a simple heuristic for top-level headers in these docs
        if re.match(r'^\s*(?:\*\*)?(?:[\d\.]+\s+)?(?:\*\*)?[A-Z][A-Za-z\s]+(?:\*\*)?\s*$', line) and len(line) < 100:
             # But don't stop for just any bold line, ensure it looks header-y
             # For now, strict '#' or strict 'References' check via remove_references_section logic is safer
             # If we hit References, stop
             if "REFERENCES" in line.upper() or "BIBLIOGRAPHY" in line.upper():
                  break
            
        conclusion_lines.append(line)
        
    return '\n'.join(conclusion_lines).strip()
