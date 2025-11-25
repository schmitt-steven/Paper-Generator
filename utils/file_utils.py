import re
import json
from pathlib import Path
from typing import Any


def save_markdown(content: str, filename: str, output_dir: str = None) -> str:
    """Save markdown content to a file."""

    if output_dir:
        target_dir = Path(output_dir)
    else:
        # Use the current working directory
        target_dir = Path.cwd()

    # Create directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create the full file path
    file_path = target_dir / filename

    # Write the content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return str(file_path)


def load_markdown(filename: str, input_dir: str = None) -> str:
    """Load markdown content from a file."""

    if input_dir:
        source_dir = Path(input_dir)
    else:
        # Use the current working directory
        source_dir = Path.cwd()

    # Create the full file path
    file_path = source_dir / filename

    # Read the content from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def save_json(data: dict | list | Any, filename: str, output_dir: str = None, indent: int = 2, ensure_ascii: bool = False) -> str:
    """Save JSON data to a file."""

    if output_dir:
        target_dir = Path(output_dir)
    else:
        # Use the current working directory
        target_dir = Path.cwd()

    # Create directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create the full file path
    file_path = target_dir / filename

    # Write the JSON data to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    return str(file_path)


def load_json(filename: str, input_dir: str = None) -> dict | list | Any:
    """Load JSON data from a file."""

    if input_dir:
        source_dir = Path(input_dir)
    else:
        # Use the current working directory
        source_dir = Path.cwd()

    # Create the full file path
    file_path = source_dir / filename

    # Read the JSON data from the file
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
    
    # Remove single characters on their own lines (likely OCR artifacts)
    markdown = re.sub(r'^\s*[a-zA-Z0-9]\s*$', '', markdown, flags=re.MULTILINE)
    
    # Remove likely page numbers (isolated small numbers, typically 1-999 for page numbers)
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

