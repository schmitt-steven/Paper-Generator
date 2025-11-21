"""Citation extraction and bibliography generation."""

import re
import textwrap
import logging
from typing import List, Set, Dict
from phases.paper_search.arxiv_api import Paper
from phases.paper_writing.data_models import PaperDraft

logger = logging.getLogger(__name__)


def extract_citation_keys_from_markdown(md_text: str) -> Set[str]:
    """
    Extract citation keys from markdown text (before LaTeX conversion).
    
    Handles citations in square brackets format:
    - [smith2024quantum]
    - [smith2024, jones2023] or [smith2024; jones2023]
    - [1], [29] (numeric - will generate placeholder entries)
    
    Args:
        md_text: Markdown text with citations in square brackets
    
    Returns:
        Set of unique citation keys (including numeric ones for placeholder generation)
    """
    # Pattern to match [key1] or [key1, key2] or [key1; key2]
    # Handles both comma and semicolon separators
    pattern = r'\[([a-zA-Z0-9]+(?:\s*[,;]\s*[a-zA-Z0-9]+)*)\]'
    matches = re.findall(pattern, md_text)
    
    citation_keys = set()
    for match in matches:
        # Split by comma or semicolon and strip whitespace
        keys = [k.strip() for k in re.split(r'[,;]', match)]
        citation_keys.update(keys)
    
    return citation_keys


def extract_citation_keys_from_latex(latex_text: str) -> Set[str]:
    """
    Extract all citation keys from LaTeX text.
    
    Handles both single and multiple citations:
    - \\cite{key1}
    - \\cite{key1,key2,key3}
    
    Args:
        latex_text: LaTeX-formatted text
    
    Returns:
        Set of unique citation keys
    """
    # Pattern to match \cite{key1,key2,key3} or \cite{key1}
    pattern = r'\\cite\{([^}]+)\}'
    matches = re.findall(pattern, latex_text)
    
    citation_keys = set()
    for match in matches:
        # Split by comma and strip whitespace
        keys = [k.strip() for k in match.split(",")]
        citation_keys.update(keys)
    
    return citation_keys


def extract_all_citations(paper_draft: PaperDraft, is_latex: bool = False) -> Set[str]:
    """
    Extract all citation keys from all sections of a PaperDraft.
    
    Args:
        paper_draft: PaperDraft with sections (markdown or LaTeX)
        is_latex: If True, extract from LaTeX format; if False, extract from markdown format
    
    Returns:
        Set of unique citation keys
    """
    all_keys = set()
    
    # Extract from each section
    sections = [
        paper_draft.abstract,
        paper_draft.introduction,
        paper_draft.related_work,
        paper_draft.methods,
        paper_draft.results,
        paper_draft.discussion,
        paper_draft.conclusion,
    ]
    
    extract_func = extract_citation_keys_from_latex if is_latex else extract_citation_keys_from_markdown
    
    for section_text in sections:
        if section_text:
            keys = extract_func(section_text)
            all_keys.update(keys)
    
    return all_keys


def create_paper_mapping(indexed_papers: List[Paper]) -> Dict[str, Paper]:
    """
    Create a mapping from citation_key to Paper object.
    
    Maps papers by both their current citation_key and their BibTeX key (if available).
    This handles cases where papers have short keys (e.g., "lee2018") but the paper
    draft uses full BibTeX keys (e.g., "Lee2018SampleEfficientDR").
    
    Args:
        indexed_papers: List of Paper objects with citation_key set
    
    Returns:
        Dictionary mapping citation_key -> Paper (includes both short and BibTeX keys)
    """
    mapping = {}
    for paper in indexed_papers:
        # Map by current citation_key
        if paper.citation_key:
            mapping[paper.citation_key] = paper
        
        # Also map by BibTeX key if BibTeX is available
        # Extract key from BibTeX entry: @article{Key, or @inproceedings{Key,
        if paper.bibtex:
            bibtex_match = re.search(r'@\w+\{([^,]+)', paper.bibtex)
            if bibtex_match:
                bibtex_key = bibtex_match.group(1).strip()
                if bibtex_key and bibtex_key != paper.citation_key:
                    mapping[bibtex_key] = paper
    
    return mapping


def generate_bibtex_entry(paper: Paper) -> str:
    """
    Generate a BibTeX entry for a Paper object.
    
    Uses paper.bibtex if available, otherwise generates minimal entry.
    
    Args:
        paper: Paper object
    
    Returns:
        BibTeX entry as string
    """
    if paper.bibtex:
        return paper.bibtex
    
    # Generate minimal BibTeX entry
    # Extract year from published date
    year = paper.published[:4] if paper.published and len(paper.published) >= 4 else "n.d."
    
    # Format authors
    if paper.authors:
        authors = " and ".join(paper.authors)
    else:
        authors = "Unknown"
    
    # Use citation_key as entry key
    entry_key = paper.citation_key or "unknown"
    
    # Determine entry type (default to article)
    entry_type = "article"
    if paper.journal_ref:
        entry_type = "article"
    elif "arxiv" in (paper.primary_category or "").lower():
        entry_type = "article"
    
    # Build BibTeX entry
    bibtex_lines = [
        f"@{entry_type}{{{entry_key},",
        f"  author = {{{authors}}},",
        f"  title = {{{paper.title}}},",
        f"  year = {{{year}}},",
    ]
    
    if paper.journal_ref:
        bibtex_lines.append(f"  journal = {{{paper.journal_ref}}},")
    
    if paper.doi:
        bibtex_lines.append(f"  doi = {{{paper.doi}}},")
    
    if paper.published:
        bibtex_lines.append(f"  date = {{{paper.published}}},")
    
    # Remove trailing comma from last line
    bibtex_lines[-1] = bibtex_lines[-1].rstrip(",")
    bibtex_lines.append("}")
    
    return "\n".join(bibtex_lines)


def generate_literature_bib(
    paper_draft: PaperDraft,
    indexed_papers: List[Paper],
    is_latex: bool = False,
) -> str:
    """
    Generate literature.bib file content from PaperDraft citations.
    
    Args:
        paper_draft: PaperDraft with sections (markdown or LaTeX)
        indexed_papers: List of Paper objects to map citations to
        is_latex: If True, extract from LaTeX format; if False, extract from markdown format
    
    Returns:
        Complete BibTeX file content as string
    """
    # Extract all citation keys
    citation_keys = extract_all_citations(paper_draft, is_latex=is_latex)
    
    # Create mapping
    paper_mapping = create_paper_mapping(indexed_papers)
    
    # Generate BibTeX entries
    bibtex_entries = []
    missing_keys = []
    
    for key in sorted(citation_keys):
        if key in paper_mapping:
            paper = paper_mapping[key]
            bibtex_entry = generate_bibtex_entry(paper)
            bibtex_entries.append(bibtex_entry)
        else:
            missing_keys.append(key)
            logger.warning(f"[Bibliography] Missing citation key: {key}")
            # Generate placeholder entry with complete required fields
            placeholder = textwrap.dedent(f"""\
                @misc{{{key},
                  author = {{Unknown}},
                  title = {{Missing reference for {key}}},
                  year = {{n.d.}},
                  note = {{Citation key not found in indexed papers}},
                }}""")
            bibtex_entries.append(placeholder)
    
    #if missing_keys:
    #    logger.warning(f"[Bibliography] {len(missing_keys)} citation keys not found in indexed papers: {missing_keys}")
    
    return "\n\n".join(bibtex_entries)

