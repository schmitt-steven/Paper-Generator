import re
import textwrap
import html
from typing import List, Dict
from phases.literature_search.paper import Paper
from phases.paper_writing.data_models import PaperDraft

def extract_citation_keys_from_markdown(md_text: str) -> set[str]:
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
        for key in keys:
            # Citation keys must contain at least one digit (e.g. year like Smith2024).
            # This filters out plain English words used in square-bracket notation
            # for variable names, lists, etc. (e.g. [Area, Density, Population]).
            if re.search(r'\d', key):
                citation_keys.add(key)

    return citation_keys



def extract_all_citations(paper_draft: PaperDraft) -> set[str]:
    """
    Extract all citation keys from all sections of a PaperDraft.
    
    Args:
        paper_draft: PaperDraft with sections in markdown format
    
    Returns:
        Set of unique citation keys
    """
    all_keys = set()
    
    sections = [
        paper_draft.abstract,
        paper_draft.introduction,
        paper_draft.related_work,
        paper_draft.methods,
        paper_draft.results,
        paper_draft.discussion,
        paper_draft.conclusion,
    ]
    
    for section_text in sections:
        if section_text:
            keys = extract_citation_keys_from_markdown(section_text)
            all_keys.update(keys)
    
    return all_keys


def create_paper_mapping(indexed_papers: list[Paper]) -> dict[str, Paper]:
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
        # Unescape HTML entities (e.g. &amp; → &) then escape & for LaTeX
        bibtex = html.unescape(paper.bibtex).replace('&', r'\&')
        # Replace the BibTeX entry key with paper.citation_key to ensure
        # the key in the .bib file matches \cite{} commands in the LaTeX.
        # The raw bibtex may have a different key (e.g., with Unicode accents
        # like "Petráš2025..." vs the normalized citation_key "Petras2025...").
        if paper.citation_key:
            return re.sub(
                r'(@\w+\{)[^,]+',
                lambda m: m.group(1) + paper.citation_key,
                bibtex,
                count=1,
            )
        return bibtex
    
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
    
    # Determine entry type
    entry_type = "article"  # Default fallback
    
    # Try to extract from BibTeX if available
    if paper.bibtex:
        bibtex_type_match = re.search(r'@(\w+)\{', paper.bibtex)
        if bibtex_type_match:
            entry_type = bibtex_type_match.group(1)
    
    # If no BibTeX or couldn't extract, infer from venue name
    if entry_type == "article" and paper.venue:
        venue_lower = paper.venue.lower()
        # Conference indicators
        conference_keywords = ["conference", "proceedings", "workshop", "symposium", 
                               "iclr", "neurips", "icml", "aaai", "ijcai", "acl", 
                               "emnlp", "cvpr", "iccv", "eccv", "sigir", "kdd"]
        # Journal indicators  
        journal_keywords = ["journal", "transactions", "review", "magazine"]
        
        if any(keyword in venue_lower for keyword in conference_keywords):
            entry_type = "inproceedings"
        elif any(keyword in venue_lower for keyword in journal_keywords):
            entry_type = "article"
    
    # Build BibTeX entry
    bibtex_lines = [
        f"@{entry_type}{{{entry_key},",
        f"  author = {{{authors}}},",
        f"  title = {{{paper.title}}},",
        f"  year = {{{year}}},",
    ]
    
    # Add venue field - choose field name based on entry type
    if paper.venue:
        if entry_type == "inproceedings":
            bibtex_lines.append(f"  booktitle = {{{paper.venue}}},")
        else:  # article or other
            bibtex_lines.append(f"  journal = {{{paper.venue}}},")
    
    if paper.doi:
        bibtex_lines.append(f"  doi = {{{paper.doi}}},")
    
    if paper.published:
        bibtex_lines.append(f"  date = {{{paper.published}}},")
    
    # Remove trailing comma from last line
    bibtex_lines[-1] = bibtex_lines[-1].rstrip(",")
    bibtex_lines.append("}")
    
    return "\n".join(bibtex_lines)


def _sanitize_bibtex_unicode(text: str) -> str:
    """Strip non-Latin Unicode characters that crash pdflatex/biber (Cyrillic, CJK, etc.)
    and replace smart quotes with ASCII equivalents."""
    # Smart quotes → ASCII
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    text = text.replace('\u201C', '"').replace('\u201D', '"')  # " "
    # Strip anything outside Basic Latin + Latin Extended (U+0000–U+024F)
    text = re.sub('[^\u0000-\u024F]', '', text)
    return text


def generate_literature_bib(
    paper_draft: PaperDraft,
    indexed_papers: list[Paper],
) -> str:
    """
    Generate bibliography.bib file content from PaperDraft citations.
    
    Args:
        paper_draft: PaperDraft with sections in markdown format
        indexed_papers: List of Paper objects to map citations to
    
    Returns:
        Complete BibTeX file content as string
    """
    citation_keys = extract_all_citations(paper_draft)
    
    # Create mapping
    paper_mapping = create_paper_mapping(indexed_papers)
    
    # Generate BibTeX entries
    bibtex_entries = []
    missing_keys = []
    
    for key in sorted(citation_keys):
        if key in paper_mapping:
            paper = paper_mapping[key]
            bibtex_entry = _sanitize_bibtex_unicode(generate_bibtex_entry(paper))
            bibtex_entries.append(bibtex_entry)
        else:
            missing_keys.append(key)
            print(f"[Bibliography] Missing citation key: {key}")
            # Generate placeholder entry with complete required fields
            placeholder = textwrap.dedent(f"""\
                @misc{{{key},
                  author = {{Unknown}},
                  title = {{Missing reference for {key}}},
                  year = {{n.d.}},
                  note = {{Citation key not found in indexed papers}},
                }}""")
            bibtex_entries.append(placeholder)
    
    return "\n\n".join(bibtex_entries)

