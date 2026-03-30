"""
Executes all six phases, captures logs, timing data and metadata into a single JSON log file for evaluation.

Usage:
    python -m tests.demonstration.demo_runner
"""

import json
import os
import sys
import time

# Add project root to path so bare imports like `settings` resolve when running
# this script directly (python3 demo_runner.py) instead of via `python -m`.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import traceback
import hashlib
from datetime import datetime
from io import StringIO
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict


# ── Pipeline parameters ─────────────────────────────────────────────────────
# Parameters that influences the demo output.
# Parameters marked "passable" are forwarded to the backend; the rest are
# hardcoded in the source and recorded here for reproducibility.
# This excludes the paramters in settings.py

PIPELINE_PARAMS = {
    # Phase 2 — Literature Search
    "search_query_count": 15,      
    "max_results_per_query": 20,       
    "fields_of_study": "Computer Science",  
    "ranking_weights": {                     
        "relevance": 0.8,
        "citations": 0.1,
        "recency": 0.1,
    },
    "filter_target_count": 40,       
    "filter_min_relevance": 0.5,     

    # Phase 4 — Experimentation
    "experiment_max_fix_attempts": 5,        
    "experiment_max_validation_attempts": 3,  

    # Phase 5 — Paper Writing  (passable — forwarded to write_paper)
    "chunks_per_query": 5,            
    "max_chunks_per_paper": 2,        
}


# ── Log capture ──────────────────────────────────────────────────────────────

@dataclass
class PhaseLog:
    """Saves all metadata for a phase."""
    phase_name: str
    phase_number: int
    status: str = "pending"          # pending | running | passed | failed
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    status_messages: list[str] = field(default_factory=list)
    error: str = ""
    error_traceback: str = ""
    output_files: list[dict] = field(default_factory=list)   # [{path, size_bytes, sha256}]
    extra: dict = field(default_factory=dict)                 # phase-specific metadata


@dataclass
class DemoLog:
    """Summary log for the entire demonstration run."""
    run_id: str = ""
    start_time: str = ""
    end_time: str = ""
    total_duration_seconds: float = 0.0
    settings_snapshot: dict = field(default_factory=dict)
    pipeline_params: dict = field(default_factory=dict)
    phases: list[dict] = field(default_factory=list)
    overall_status: str = "pending"  # pending | passed | failed
    summary: dict = field(default_factory=dict)
    stdout_log: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_meta(path: str | Path) -> dict:
    """Return size and SHA-256 for a file."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    size = p.stat().st_size
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": str(p), "size_bytes": size, "sha256": sha}


def _snapshot_settings() -> dict:
    """Capture all current Settings values."""
    from settings import Settings
    snap = {}
    for attr in dir(Settings):
        if attr.startswith("_") or callable(getattr(Settings, attr)):
            continue
        val = getattr(Settings, attr)
        # Make enum serialisable
        if hasattr(val, "value"):
            val = val.value
        snap[attr] = val
    return snap


def _collect_output_files(directory: str, extensions: set[str] | None = None) -> list[dict]:
    """Collect metadata for all files in a directory (non-recursive by default)."""
    results = []
    d = Path(directory)
    if not d.exists():
        return results
    for f in sorted(d.rglob("*")):
        if f.is_file():
            if extensions and f.suffix.lower() not in extensions:
                continue
            results.append(_file_meta(f))
    return results


@contextmanager
def capture_stdout():
    """Context manager that captures stdout while still printing to terminal."""
    buffer = StringIO()
    original = sys.stdout

    class Tee:
        def write(self, data):
            buffer.write(data)
            original.write(data)

        def flush(self):
            buffer.flush()
            original.flush()

    sys.stdout = Tee()
    try:
        yield buffer
    finally:
        sys.stdout = original


# ── Phase runners ────────────────────────────────────────────────────────────

def run_phase(phase_log: PhaseLog, func, *args, **kwargs):
    """Execute a phase function, populate the PhaseLog, and return the result."""
    phase_log.status = "running"
    phase_log.start_time = datetime.now().isoformat()
    print(f"\n{'='*70}")
    print(f"  PHASE {phase_log.phase_number}: {phase_log.phase_name}")
    print(f"  Started at {phase_log.start_time}")
    print(f"{'='*70}\n")

    result = None
    try:
        result = func(phase_log, *args, **kwargs)
        phase_log.status = "passed"
    except Exception as e:
        phase_log.status = "failed"
        phase_log.error = str(e)
        phase_log.error_traceback = traceback.format_exc()
        print(f"\n  !! PHASE FAILED: {e}\n")

    phase_log.end_time = datetime.now().isoformat()
    phase_log.duration_seconds = round(
        (datetime.fromisoformat(phase_log.end_time) - datetime.fromisoformat(phase_log.start_time)).total_seconds(), 2
    )
    print(f"\n  Phase {phase_log.phase_number} finished in {phase_log.duration_seconds}s — {phase_log.status.upper()}")
    return result


# ── Individual phases ────────────────────────────────────────────────────────

def phase_1_context_analysis(log: PhaseLog):
    """Phase 1: Generate research context from user files."""
    from phases.context_analysis.research_context_generator import ResearchContextGenerator

    def cb(msg):
        log.status_messages.append(msg)
        print(f"  [{log.phase_name}] {msg}")

    context = ResearchContextGenerator.generate_new_context(progress_callback=cb)

    # Record outputs
    log.output_files = [_file_meta("output/research_context.md")]
    log.extra["description_length"] = len(context.description)
    log.extra["open_questions_length"] = len(context.open_questions)
    log.extra["code_snippets_length"] = len(context.code_snippets)
    return context


def phase_2_literature_search(log: PhaseLog):
    """Phase 2: Search, rank, filter, download, and convert papers.

    Runs the same steps as LiteratureSearch.run_automated_search but captures
    intermediate paper counts at each stage for the summary.
    """
    from phases.context_analysis.research_context_generator import ResearchContextGenerator
    from phases.paper_search.literature_search import LiteratureSearch
    from phases.paper_search.paper_ranking import PaperRanker
    from phases.paper_search.paper_filter import PaperFilter
    from phases.paper_search.citation_gap_finder import CitationGapFinder
    from utils.open_access_finder import find_open_access_pdfs
    from utils.pdf_downloader import PDFDownloader
    from utils.pdf_converter import PDFConverter
    from settings import Settings

    def cb(msg):
        log.status_messages.append(msg)
        print(f"  [{log.phase_name}] {msg}")

    research_context = ResearchContextGenerator.load_research_context("output/research_context.md")
    lit = LiteratureSearch(model_name=Settings.LITERATURE_SEARCH_MODEL)

    # Step 1: Build queries and search
    cb("Building search queries")
    search_queries = lit.build_search_queries(research_context)
    log.extra["search_queries_generated"] = len(search_queries)

    cb(f"Searching with {len(search_queries)} queries")
    raw_papers = lit.search_papers(search_queries, max_results_per_query=20)
    log.extra["papers_raw_total"] = len(raw_papers)

    # Step 2: Rank
    cb("Ranking papers")
    ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
    ranked_papers = ranker.rank_papers(
        papers=raw_papers, context=research_context.description,
        weights={"relevance": 0.8, "citations": 0.1, "recency": 0.1},
    )
    log.extra["papers_after_ranking"] = len(ranked_papers)

    # Step 3: Filter
    cb("Filtering papers")
    enhanced_context = f"{research_context.description}\n\nOpen Research Questions:\n{research_context.open_questions}"
    filtered_papers = PaperFilter.filter_papers(
        papers=ranked_papers, research_context=enhanced_context,
        model_name=lit.model_name, target_count=40, min_relevance=0.5,
    )
    log.extra["papers_after_filter"] = len(filtered_papers)

    # Step 4: Citation gap analysis
    cb("Analyzing citation gaps")
    gap_finder = CitationGapFinder()
    suggestions = gap_finder.identify_missing_papers(
        papers=filtered_papers, research_context=enhanced_context,
        model_name=lit.model_name,
    )
    log.extra["gap_suggestions"] = len(suggestions) if suggestions else 0

    foundational_count = 0
    if suggestions:
        cb(f"Searching for {len(suggestions)} foundational papers")
        existing_ids = {p.id for p in filtered_papers}
        foundational_papers = gap_finder.search_suggested_papers(suggestions, existing_ids)
        if foundational_papers:
            cb("Ranking foundational papers")
            foundational_papers = ranker.rank_papers(
                papers=foundational_papers, context=research_context.description,
            )
            filtered_papers.extend(foundational_papers)
            filtered_papers.sort(
                key=lambda p: p.ranking.relevance_score if p.ranking else 0, reverse=True,
            )
            foundational_count = len(foundational_papers)
    log.extra["foundational_papers_added"] = foundational_count

    # Step 5: Open access check
    papers_without_urls = [p for p in filtered_papers if not p.pdf_url]
    if papers_without_urls:
        cb(f"Finding open access PDFs for {len(papers_without_urls)} papers")
        find_open_access_pdfs(papers_without_urls)

    papers = filtered_papers
    log.extra["papers_final"] = len(papers)
    log.extra["open_access_count"] = sum(1 for p in papers if p.is_open_access)

    # Step 6: Download and convert
    cb("Downloading open-access PDFs")
    downloadable = [p for p in papers if p.is_open_access and p.pdf_url]
    if downloadable:
        ok, fail = PDFDownloader.download_papers_as_pdfs(downloadable, base_folder="output/literature/")
        log.extra["pdfs_downloaded"] = ok
        log.extra["pdfs_failed"] = fail
    else:
        log.extra["pdfs_downloaded"] = 0
        log.extra["pdfs_failed"] = 0

    cb("Converting PDFs to markdown")
    converter = PDFConverter()
    converter.convert_all_papers(downloadable, base_folder="output/literature/")

    # Save
    cb("Saving papers")
    LiteratureSearch.save_papers(papers, filename="papers.json", output_dir="output")
    log.output_files = [_file_meta("output/papers.json")]
    return papers


def phase_3_hypothesis_generation(log: PhaseLog):
    """Phase 3: Generate hypothesis from research context + paper specification."""
    from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder

    def cb(msg):
        log.status_messages.append(msg)
        print(f"  [{log.phase_name}] {msg}")

    hypothesis = HypothesisBuilder.generate_new_hypothesis(status_callback=cb)

    log.output_files = [_file_meta("output/hypothesis.md")]
    log.extra["description"] = hypothesis.description[:500]
    log.extra["rationale"] = hypothesis.rationale[:500]
    log.extra["success_criteria"] = hypothesis.success_criteria[:500]
    return hypothesis


def phase_4_experimentation(log: PhaseLog):
    """Phase 4: Generate experiment plan, code, execute, evaluate."""
    from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder, Hypothesis
    from phases.experimentation.experiment_runner import ExperimentRunner

    def cb(msg):
        log.status_messages.append(msg)
        print(f"  [{log.phase_name}] {msg}")

    # Load hypothesis
    hypothesis = HypothesisBuilder.load_hypothesis("output/hypothesis.md")
    if hypothesis is None:
        raise ValueError("No hypothesis found — cannot run experimentation phase.")

    # Generate experiment plan
    cb("Generating experiment plan")
    ExperimentRunner.generate_new_experiment_plan(hypothesis, status_callback=cb)
    log.output_files.append(_file_meta("output/experiments/experiment_plan.md"))

    # Run experiment
    cb("Running experiment")
    result = ExperimentRunner.run_new_experiment(status_callback=cb)

    log.output_files.append(_file_meta("output/experiments/experiment_result.json"))

    # Collect plot files
    plots_dir = Path("output/experiments/plots")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*"))
        log.extra["plot_count"] = len(plot_files)
        for pf in plot_files:
            log.output_files.append(_file_meta(pf))
    else:
        log.extra["plot_count"] = 0

    log.extra["experiment_executed"] = True
    log.extra["fix_attempts"] = result.fix_attempts
    log.extra["validation_attempts"] = result.validation_attempts
    log.extra["execution_time"] = result.execution_time
    log.extra["verdict"] = result.hypothesis_evaluation.verdict if result.hypothesis_evaluation else None
    log.extra["verdict_reasoning"] = result.hypothesis_evaluation.reasoning if result.hypothesis_evaluation else None

    exp_code_path = Path("output/experiments/experiment.py")
    if exp_code_path.exists():
        lines = exp_code_path.read_text(encoding="utf-8").splitlines()
        log.extra["experiment_code_lines"] = len(lines)
        log.extra["experiment_code_non_blank_lines"] = sum(1 for l in lines if l.strip())

    return result


def _build_writing_timing(events: list[tuple[float, str]]) -> dict:
    """
    Parse status callbacks into per-section and per-step timing.

    Returns a dict like:
    {
        "indexing_seconds": 312.5,
        "sections": {
            "Methods": {
                "total_seconds": 180.0,
                "steps": {
                    "Drafting": 45.2,
                    "Critiquing": 30.1,
                    ...
                }
            }, ...
        },
        "step_totals": {
            "Drafting": 310.0,
            "Critiquing": 200.0,
            ...
        }
    }
    """
    import re

    result: dict = {"indexing_seconds": 0.0, "sections": {}, "step_totals": {}}
    step_keywords = ["Drafting", "Critiquing", "Searching evidence", "Rewriting"]

    for i, (ts, msg) in enumerate(events):
        next_ts = events[i + 1][0] if i + 1 < len(events) else ts
        duration = round(next_ts - ts, 2)

        # Indexing
        if "embeddings" in msg.lower():
            result["indexing_seconds"] = duration
            continue

        # Evidence chunk count: "Evidence chunks for Methods: 12"
        m = re.match(r'Evidence chunks for (.+): (\d+)$', msg)
        if m:
            section, count = m.group(1), int(m.group(2))
            if section not in result["sections"]:
                result["sections"][section] = {"total_seconds": 0.0, "steps": {}}
            result["sections"][section]["evidence_chunks"] = count
            continue

        # Rewrite delta: "Section rewrite delta for Methods: 1234 -> 1567"
        m = re.match(r'Section rewrite delta for (.+): (\d+) -> (\d+)$', msg)
        if m:
            section, before, after = m.group(1), int(m.group(2)), int(m.group(3))
            if section not in result["sections"]:
                result["sections"][section] = {"total_seconds": 0.0, "steps": {}}
            result["sections"][section]["chars_before_rewrite"] = before
            result["sections"][section]["chars_after_rewrite"] = after
            result["sections"][section]["chars_delta"] = after - before
            continue

        # Per-section steps: "Drafting Methods section", "Searching evidence for Methods"
        for kw in step_keywords:
            if msg.startswith(kw):
                # Extract section name: "Drafting X section" or "Searching evidence for X"
                m = re.search(r'(?:section|for)\s*$', msg)
                section = msg.replace(kw, "").replace(" section", "").replace(" for ", "").strip()
                if not section:
                    continue

                if section not in result["sections"]:
                    result["sections"][section] = {"total_seconds": 0.0, "steps": {}}

                result["sections"][section]["steps"][kw] = duration
                result["sections"][section]["total_seconds"] = round(
                    result["sections"][section]["total_seconds"] + duration, 2
                )
                result["step_totals"][kw] = round(
                    result["step_totals"].get(kw, 0.0) + duration, 2
                )
                break

    return result


def phase_5_paper_writing(log: PhaseLog):
    """Phase 5: Write paper sections with critique-rewrite loop."""
    from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
    from phases.context_analysis.research_context_generator import ResearchContextGenerator
    from phases.context_analysis.paper_specification import PaperSpecification
    from phases.experimentation.experiment_runner import ExperimentRunner
    from phases.paper_search.literature_search import LiteratureSearch

    # Timestamped events for per-step timing
    timed_events: list[tuple[float, str]] = []

    def cb(msg):
        timed_events.append((time.monotonic(), msg))
        log.status_messages.append(msg)
        print(f"  [{log.phase_name}] {msg}")

    # Load resources (same as generate_new_draft but call write_paper directly
    # so it can pass through the controllable pipeline params)
    cb("Loading resources")
    research_context = ResearchContextGenerator.load_research_context("output/research_context.md")
    experiment_result = ExperimentRunner.load_experiment_result("output/experiments/experiment_result.json")
    papers = LiteratureSearch.load_papers("output/papers.json")

    paper_specification = None
    try:
        paper_specification = PaperSpecification.load("user_files/paper_specification.md")
    except Exception:
        pass

    pipeline = PaperWritingPipeline()
    draft = pipeline.write_paper(
        research_context=research_context,
        experiment_result=experiment_result,
        papers=papers,
        paper_specification=paper_specification,
        status_callback=cb,
        chunks_per_query=PIPELINE_PARAMS["chunks_per_query"],
        max_chunks_per_paper=PIPELINE_PARAMS["max_chunks_per_paper"],
    )

    # Mark end so the last step's duration can be computed
    timed_events.append((time.monotonic(), "_end"))

    log.output_files = [
        _file_meta("output/paper_draft.md"),
        _file_meta("output/section_writing_prompts.json"),
        _file_meta("output/section_rewrite_prompts.json"),
    ]
    log.extra["title"] = draft.title
    log.extra["section_lengths"] = {
        "abstract": len(draft.abstract),
        "introduction": len(draft.introduction),
        "related_work": len(draft.related_work),
        "methods": len(draft.methods),
        "results": len(draft.results),
        "discussion": len(draft.discussion),
        "conclusion": len(draft.conclusion),
        "acknowledgements": len(draft.acknowledgements) if draft.acknowledgements else 0,
    }
    log.extra["writing_timing"] = _build_writing_timing(timed_events)

    # Citation analysis: count keys in draft text and check against papers.json
    import re as _re
    draft_text = "\n".join([
        draft.abstract, draft.introduction, draft.related_work,
        draft.methods, draft.results, draft.discussion, draft.conclusion,
        draft.acknowledgements or "",
    ])
    # Citation keys look like [AuthorYear] or [Key1, Key2] in the markdown draft
    raw_citations = _re.findall(r'\[([A-Za-z][A-Za-z0-9_,\s-]*\d{4}[a-zA-Z0-9_]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_,\s-]*\d{4}[a-zA-Z0-9_]*)*)\]', draft_text)
    citation_keys = set()
    for match in raw_citations:
        for key in match.split(','):
            key = key.strip()
            if key:
                citation_keys.add(key)
    paper_keys = {p.citation_key for p in papers if hasattr(p, "citation_key") and p.citation_key}
    matched = citation_keys & paper_keys
    hallucinated = citation_keys - paper_keys

    log.extra["citations_total"] = len(citation_keys)
    log.extra["citations_matched"] = len(matched)
    log.extra["citations_hallucinated"] = len(hallucinated)
    log.extra["citations_hallucinated_keys"] = sorted(hallucinated)

    # Word counts
    section_texts = {
        "abstract":     draft.abstract,
        "introduction": draft.introduction,
        "related_work": draft.related_work,
        "methods":      draft.methods,
        "results":      draft.results,
        "discussion":   draft.discussion,
        "conclusion":   draft.conclusion,
        "acknowledgements": draft.acknowledgements or "",
    }
    section_word_counts = {sec: len(text.split()) for sec, text in section_texts.items()}
    total_words = sum(section_word_counts.values())
    log.extra["section_word_counts"] = section_word_counts
    log.extra["total_word_count"] = total_words
    log.extra["citation_density_per_1000_words"] = (
        round(len(citation_keys) / total_words * 1000, 2) if total_words > 0 else 0
    )

    # Closed-access citation ratio: how many matched citations came from papers without a PDF
    papers_by_key = {
        p.citation_key: p for p in papers if hasattr(p, "citation_key") and p.citation_key
    }
    closed_access_cited = {k for k in matched if k in papers_by_key and not papers_by_key[k].is_open_access}
    log.extra["citations_from_closed_access"] = len(closed_access_cited)
    log.extra["citations_from_closed_access_keys"] = sorted(closed_access_cited)

    return draft


def phase_6_document_compilation(log: PhaseLog):
    """Phase 6: Convert to LaTeX and compile to PDF."""
    from phases.latex_generation.paper_converter import PaperConverter

    def cb(msg):
        log.status_messages.append(msg)
        print(f"  [{log.phase_name}] {msg}")

    success = PaperConverter.generate_new_pdf(status_callback=cb)

    pdf_path = Path("output/latex/result/paper.pdf")
    log.output_files = [_file_meta(pdf_path)]
    log.extra["compilation_success"] = success
    log.extra["pdf_size_bytes"] = pdf_path.stat().st_size if pdf_path.exists() else 0

    if pdf_path.exists():
        try:
            from pypdf import PdfReader
            log.extra["pdf_page_count"] = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            log.extra["pdf_page_count"] = None

    if not success:
        raise RuntimeError("LaTeX compilation failed — see stdout log for details.")

    return success


# ── Main orchestrator ────────────────────────────────────────────────────────

PHASES = [
    ("Context Analysis",        phase_1_context_analysis),
    ("Literature Search",       phase_2_literature_search),
    ("Hypothesis Generation",   phase_3_hypothesis_generation),
    ("Experimentation",         phase_4_experimentation),
    ("Paper Writing",           phase_5_paper_writing),
    ("Document Compilation",    phase_6_document_compilation),
]


def _clear_literature_folder():
    """Remove all contents of output/literature/ before a fresh run."""
    import shutil
    lit_dir = Path("output/literature")
    if lit_dir.exists():
        shutil.rmtree(lit_dir)
    lit_dir.mkdir(parents=True, exist_ok=True)


def main():
    # Ensure we're in the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    _clear_literature_folder()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = DemoLog(
        run_id=run_id,
        start_time=datetime.now().isoformat(),
        settings_snapshot=_snapshot_settings(),
        pipeline_params=PIPELINE_PARAMS,
    )

    run_dir = Path(f"tests/demonstration/run_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_output_path = run_dir / f"demo_log_{run_id}.json"

    print(f"Demo run {run_id} — run folder: {run_dir}")
    print(f"Project root: {project_root}\n")

    all_passed = True

    with capture_stdout() as stdout_buffer:
        for i, (name, func) in enumerate(PHASES, start=1):
            phase_log = PhaseLog(phase_name=name, phase_number=i)
            run_phase(phase_log, func)
            log.phases.append(asdict(phase_log))

            # Save intermediate log after every phase (crash safety)
            _save_log(log, log_output_path, stdout_buffer)

            if phase_log.status == "failed":
                all_passed = False
                print(f"\n  Stopping: phase {i} ({name}) failed.")
                break

            # Stop early if experiment verdict is not proven
            if name == "Experimentation":
                verdict = phase_log.extra.get("verdict", "")
                if verdict in ("inconclusive", "not supported"):
                    all_passed = False
                    print(f"\n  Stopping: experiment verdict is '{verdict}'. Skipping paper writing and compilation.")
                    break

    log.end_time = datetime.now().isoformat()
    log.total_duration_seconds = round(
        (datetime.fromisoformat(log.end_time) - datetime.fromisoformat(log.start_time)).total_seconds(), 2
    )
    log.overall_status = "passed" if all_passed else "failed"

    # Build and print summary
    summary = _build_summary(log)
    log.summary = summary
    _print_summary(summary)

    _save_log(log, log_output_path, stdout_buffer)

    # Also save summary as separate file for quick access
    summary_path = log_output_path.with_name(log_output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Copy entire output/ folder into the run directory
    import shutil
    output_src = project_root / "output"
    output_dst = run_dir / "output"
    if output_src.exists():
        if output_dst.exists():
            shutil.rmtree(output_dst)
        shutil.copytree(output_src, output_dst)
        print(f"  Output copied to: {output_dst}")

    print(f"  Log saved to:     {log_output_path}")
    print(f"  Summary saved to: {summary_path}\n")


def _build_summary(log: DemoLog) -> dict:
    """Build summary of key metrics from the completed demo log."""
    summary = {}
    phases_by_name = {p["phase_name"]: p for p in log.phases}

    # ── Timing ────────────────────────────────────────────────────────────
    summary["total_duration_seconds"] = log.total_duration_seconds
    summary["phase_durations"] = {
        p["phase_name"]: p["duration_seconds"] for p in log.phases
    }
    total_dur = log.total_duration_seconds
    summary["phase_time_pct"] = {
        p["phase_name"]: round(p["duration_seconds"] / total_dur * 100, 1) if total_dur > 0 else 0
        for p in log.phases
    }

    # ── Literature funnel ─────────────────────────────────────────────────
    lit = phases_by_name.get("Literature Search", {}).get("extra", {})
    pdfs_downloaded = lit.get("pdfs_downloaded", 0) or 0
    citations_matched = phases_by_name.get("Paper Writing", {}).get("extra", {}).get("citations_matched", 0) or 0
    summary["literature"] = {
        "search_queries":         lit.get("search_queries_generated", "?"),
        "papers_raw":             lit.get("papers_raw_total", "?"),
        "papers_after_ranking":   lit.get("papers_after_ranking", "?"),
        "papers_after_filter":    lit.get("papers_after_filter", "?"),
        "foundational_added":     lit.get("foundational_papers_added", "?"),
        "papers_final":           lit.get("papers_final", "?"),
        "open_access":            lit.get("open_access_count", "?"),
        "closed_access":          (lit.get("papers_final", 0) or 0) - (lit.get("open_access_count", 0) or 0) if isinstance(lit.get("papers_final"), int) else "?",
        "pdfs_downloaded":        lit.get("pdfs_downloaded", "?"),
        "literature_utilization_rate": round(citations_matched / pdfs_downloaded, 3) if pdfs_downloaded > 0 else "?",
    }

    # ── Experimentation ───────────────────────────────────────────────────
    exp = phases_by_name.get("Experimentation", {}).get("extra", {})
    summary["experimentation"] = {
        "fix_attempts":           exp.get("fix_attempts", "?"),
        "validation_attempts":    exp.get("validation_attempts", "?"),
        "plot_count":             exp.get("plot_count", "?"),
        "verdict":                exp.get("verdict", "?"),
        "verdict_reasoning":      exp.get("verdict_reasoning", "?"),
        "experiment_code_lines":  exp.get("experiment_code_lines", "?"),
        "experiment_code_non_blank_lines": exp.get("experiment_code_non_blank_lines", "?"),
    }

    # ── Paper writing ─────────────────────────────────────────────────────
    pw = phases_by_name.get("Paper Writing", {}).get("extra", {})
    expected_sections = [
        "abstract", "introduction", "related_work", "methods",
        "results", "discussion", "conclusion",
    ]
    section_lengths = pw.get("section_lengths", {})
    sections_written = [s for s in expected_sections if section_lengths.get(s, 0) > 0]

    writing_timing = pw.get("writing_timing", {})
    section_stats = writing_timing.get("sections", {})
    summary["paper"] = {
        "title":                  pw.get("title", "?"),
        "sections_expected":      len(expected_sections),
        "sections_written":       len(sections_written),
        "all_sections_present":   len(sections_written) == len(expected_sections),
        "section_lengths":        section_lengths,
        "section_word_counts":    pw.get("section_word_counts", {}),
        "total_word_count":       pw.get("total_word_count", "?"),
        "section_evidence_chunks": {s: v.get("evidence_chunks", 0) for s, v in section_stats.items()},
        "section_rewrite_deltas":  {
            s: {"before": v.get("chars_before_rewrite"), "after": v.get("chars_after_rewrite"), "delta": v.get("chars_delta")}
            for s, v in section_stats.items() if "chars_delta" in v
        },
        "citations_total":        pw.get("citations_total", "?"),
        "citations_matched":      pw.get("citations_matched", "?"),
        "citations_hallucinated": pw.get("citations_hallucinated", "?"),
        "hallucinated_keys":      pw.get("citations_hallucinated_keys", []),
        "citation_density_per_1000_words": pw.get("citation_density_per_1000_words", "?"),
        "citations_from_closed_access": pw.get("citations_from_closed_access", "?"),
        "citations_from_closed_access_keys": pw.get("citations_from_closed_access_keys", []),
    }

    # ── Compilation ───────────────────────────────────────────────────────
    comp = phases_by_name.get("Document Compilation", {}).get("extra", {})
    summary["compilation"] = {
        "success":                comp.get("compilation_success", "?"),
        "pdf_size_bytes":         comp.get("pdf_size_bytes", "?"),
        "pdf_page_count":         comp.get("pdf_page_count", "?"),
    }

    return summary


def _print_summary(summary: dict):
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*70}")
    print(f"  DEMONSTRATION SUMMARY")
    print(f"{'='*70}")

    # Timing
    def _fmt_time(s):
        """Format seconds as 'Xs (Y.Zm)'."""
        return f"{s:.1f}s ({s/60:.1f}m)"

    print(f"\n  ── Timing ──")
    print(f"  Total: {_fmt_time(summary['total_duration_seconds'])}")
    pct = summary.get("phase_time_pct", {})
    for name, dur in summary["phase_durations"].items():
        print(f"    {name:.<30s} {_fmt_time(dur):>14s}  ({pct.get(name, '?'):>5}%)")

    # Literature
    lit = summary.get("literature", {})
    print(f"\n  ── Literature Search ──")
    print(f"    Search queries generated:   {lit.get('search_queries')}")
    print(f"    Papers found (raw):         {lit.get('papers_raw')}")
    print(f"    After ranking:              {lit.get('papers_after_ranking')}")
    print(f"    After LLM filter:           {lit.get('papers_after_filter')}")
    print(f"    Foundational papers added:  {lit.get('foundational_added')}")
    print(f"    Final paper count:          {lit.get('papers_final')}")
    print(f"    Open access:                {lit.get('open_access')}")
    print(f"    Closed access:              {lit.get('closed_access')}")
    print(f"    PDFs downloaded:            {lit.get('pdfs_downloaded')}")
    print(f"    Literature utilization:     {lit.get('literature_utilization_rate')}  (citations matched / PDFs downloaded)")

    # Experimentation
    exp = summary.get("experimentation", {})
    print(f"\n  ── Experimentation ──")
    print(f"    Code fix attempts:          {exp.get('fix_attempts')}")
    print(f"    Validation attempts:        {exp.get('validation_attempts')}")
    print(f"    Figures generated:          {exp.get('plot_count')}")
    print(f"    Experiment code lines:      {exp.get('experiment_code_lines')}  ({exp.get('experiment_code_non_blank_lines')} non-blank)")
    print(f"    Verdict:                    {exp.get('verdict')}")
    if exp.get("verdict_reasoning"):
        print(f"    Reasoning: {exp['verdict_reasoning']}")

    # Paper
    paper = summary.get("paper", {})
    print(f"\n  ── Generated Paper ──")
    print(f"    Title: {paper.get('title')}")
    print(f"    Sections: {paper.get('sections_written')}/{paper.get('sections_expected')}"
          f" {'(all present)' if paper.get('all_sections_present') else '(MISSING SECTIONS)'}")
    word_counts = paper.get("section_word_counts", {})
    for sec, length in paper.get("section_lengths", {}).items():
        words = word_counts.get(sec, "?")
        print(f"      {sec:.<24s} {length:>6} chars  {words:>5} words")
    print(f"    Total word count:           {paper.get('total_word_count')}")

    evidence_chunks = paper.get("section_evidence_chunks", {})
    rewrite_deltas = paper.get("section_rewrite_deltas", {})
    if evidence_chunks or rewrite_deltas:
        print(f"    {'Section':<20s}  {'Chunks':>6}  {'Before':>7}  {'After':>7}  {'Delta':>7}")
        all_sections = sorted(set(list(evidence_chunks) + list(rewrite_deltas)))
        for sec in all_sections:
            chunks = evidence_chunks.get(sec, "-")
            delta_info = rewrite_deltas.get(sec, {})
            before = delta_info.get("before", "-")
            after = delta_info.get("after", "-")
            delta = delta_info.get("delta", "-")
            delta_str = f"+{delta}" if isinstance(delta, int) and delta > 0 else str(delta)
            print(f"      {sec:<20s}  {str(chunks):>6}  {str(before):>7}  {str(after):>7}  {delta_str:>7}")

    print(f"    Citations total:            {paper.get('citations_total')}")
    print(f"    Citations matched:          {paper.get('citations_matched')}")
    print(f"    Citations hallucinated:     {paper.get('citations_hallucinated')}")
    print(f"    Citation density:           {paper.get('citation_density_per_1000_words')} per 1,000 words")
    print(f"    Citations from closed-access papers: {paper.get('citations_from_closed_access')}")
    if paper.get("hallucinated_keys"):
        print(f"    Hallucinated keys:")
        for key in paper["hallucinated_keys"]:
            print(f"      - {key}")

    # Compilation
    comp = summary.get("compilation", {})
    print(f"\n  ── Document Compilation ──")
    print(f"    Success:    {comp.get('success')}")
    print(f"    Pages:      {comp.get('pdf_page_count')}")
    print(f"    PDF size:   {comp.get('pdf_size_bytes', 0):,} bytes")
    print(f"{'='*70}\n")


def _save_log(log: DemoLog, path: Path, stdout_buffer: StringIO):
    """Write current log state to disk."""
    log.stdout_log = stdout_buffer.getvalue()
    path.write_text(json.dumps(asdict(log), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
