"""
Evidence persistence and management utilities.

Handles saving, loading, adding, and removing evidence chunks organized by section.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from dataclasses import asdict

from phases.paper_writing.data_models import Evidence, PaperChunk, Section
from phases.paper_search.paper import Paper, RankingScores


EVIDENCE_FILE = "output/evidence.json"


def _paper_to_dict(paper: Paper) -> dict:
    """Convert Paper to JSON-serializable dict."""
    return {
        "id": paper.id,
        "title": paper.title,
        "published": paper.published,
        "authors": paper.authors,
        "summary": paper.summary,
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "fields_of_study": paper.fields_of_study,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "bibtex": paper.bibtex,
        "markdown_text": paper.markdown_text,
        "ranking": asdict(paper.ranking) if paper.ranking else None,
        "is_open_access": paper.is_open_access,
        "user_provided": paper.user_provided,
        "pdf_path": paper.pdf_path,
    }


def _paper_from_dict(d: dict) -> Paper:
    """Reconstruct Paper from dict."""
    ranking = None
    if d.get("ranking"):
        ranking = RankingScores(**d["ranking"])
    
    return Paper(
        id=d["id"],
        title=d["title"],
        published=d["published"],
        authors=d["authors"],
        summary=d["summary"],
        pdf_url=d.get("pdf_url"),
        doi=d.get("doi"),
        fields_of_study=d.get("fields_of_study", []),
        venue=d.get("venue"),
        citation_count=d.get("citation_count"),
        bibtex=d.get("bibtex"),
        markdown_text=d.get("markdown_text"),
        ranking=ranking,
        is_open_access=d.get("is_open_access", False),
        user_provided=d.get("user_provided", False),
        pdf_path=d.get("pdf_path"),
    )


def _chunk_to_dict(chunk: PaperChunk) -> dict:
    """Convert PaperChunk to JSON-serializable dict (without embedding for space)."""
    return {
        "chunk_id": chunk.chunk_id,
        "paper": _paper_to_dict(chunk.paper),
        "chunk_text": chunk.chunk_text,
        "chunk_index": chunk.chunk_index,
        # Skip embedding to save space - not needed for display
    }


def _chunk_from_dict(d: dict) -> PaperChunk:
    """Reconstruct PaperChunk from dict."""
    return PaperChunk(
        chunk_id=d["chunk_id"],
        paper=_paper_from_dict(d["paper"]),
        chunk_text=d["chunk_text"],
        chunk_index=d["chunk_index"],
        embedding=[],  # Not stored
    )


def _evidence_to_dict(evidence: Evidence) -> dict:
    """Convert Evidence to JSON-serializable dict."""
    return {
        "chunk": _chunk_to_dict(evidence.chunk),
        "summary": evidence.summary,
        "vector_score": evidence.vector_score,
        "llm_score": evidence.llm_score,
        "combined_score": evidence.combined_score,
        "source_query": evidence.source_query,
    }


def _evidence_from_dict(d: dict) -> Evidence:
    """Reconstruct Evidence from dict."""
    return Evidence(
        chunk=_chunk_from_dict(d["chunk"]),
        summary=d["summary"],
        vector_score=d["vector_score"],
        llm_score=d["llm_score"],
        combined_score=d["combined_score"],
        source_query=d["source_query"],
    )


def save_evidence(
    evidence_by_section: Dict[Section, Sequence[Evidence]],
    filepath: str = EVIDENCE_FILE,
) -> Path:
    """
    Save evidence dictionary to JSON file.
    
    Args:
        evidence_by_section: Dict mapping Section enum to list of Evidence objects
        filepath: Output file path
        
    Returns:
        Path to saved file
    """
    # Convert to JSON-serializable format
    data = {}
    for section, evidence_list in evidence_by_section.items():
        section_key = section.value  # Use enum value as key
        data[section_key] = [_evidence_to_dict(e) for e in evidence_list]
    
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_path


def load_evidence(filepath: str = EVIDENCE_FILE) -> Dict[Section, List[Evidence]]:
    """
    Load evidence dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Dict mapping Section enum to list of Evidence objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Evidence file not found: {filepath}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert back to typed objects
    result: Dict[Section, List[Evidence]] = {}
    for section_key, evidence_list in data.items():
        # Find matching Section enum
        section = None
        for s in Section:
            if s.value == section_key:
                section = s
                break
        
        if section is None:
            continue
            
        result[section] = [_evidence_from_dict(e) for e in evidence_list]
    
    return result


def add_evidence(
    evidence_by_section: Dict[Section, List[Evidence]],
    section: Section,
    evidence: Evidence,
) -> None:
    """
    Add a new evidence chunk to a section.
    
    Args:
        evidence_by_section: The evidence dictionary to modify
        section: Target section
        evidence: Evidence to add
    """
    if section not in evidence_by_section:
        evidence_by_section[section] = []
    evidence_by_section[section].insert(0, evidence)  # Add at top


def remove_evidence(
    evidence_by_section: Dict[Section, List[Evidence]],
    section: Section,
    chunk_id: str,
) -> bool:
    """
    Remove an evidence chunk from a section by chunk ID.
    
    Args:
        evidence_by_section: The evidence dictionary to modify
        section: Target section
        chunk_id: ID of the chunk to remove
        
    Returns:
        True if removed, False if not found
    """
    if section not in evidence_by_section:
        return False
    
    original_len = len(evidence_by_section[section])
    evidence_by_section[section] = [
        e for e in evidence_by_section[section] 
        if e.chunk.chunk_id != chunk_id
    ]
    
    return len(evidence_by_section[section]) < original_len


def get_evidence_stats(
    evidence_by_section: Dict[Section, Sequence[Evidence]],
) -> tuple[int, int]:
    """
    Get statistics about the evidence collection.
    
    Returns:
        Tuple of (total_chunks, unique_papers)
    """
    total_chunks = sum(len(ev) for ev in evidence_by_section.values())
    
    paper_ids = set()
    for evidence_list in evidence_by_section.values():
        for evidence in evidence_list:
            paper_ids.add(evidence.chunk.paper.id)
    
    return total_chunks, len(paper_ids)
