# Paper Search Phase

Searches, ranks, filters, and downloads academic papers to build a diverse bibliography for writing.

## Components

### `LiteratureSearch` ([literature_search.py](literature_search.py))
Handles automated paper search.
- **Query Generation**: LLM creates 15 queries across 5 categories (Surveys, Foundational, Core Methods, Related Work, Benchmarks).
- **Execution & Dedup**: Searches via Semantic Scholar API and removes duplicates (DOI or Title/Author match).

### `PaperRanker` ([paper_ranking.py](paper_ranking.py))
Ranks papers using a composite score:
- **Semantic Relevance (80%)**: Cosine similarity to research context.
- **Citation Score (10%)**: Age-aware citation velocity
- **Recency Score (10%)**: Exponential decay

### `PaperFilter` ([paper_filter.py](paper_filter.py))
Selects the final set (~40 papers) in 3 steps:
1.  **Autoselect**: Keeps top papers by Semantic Relevance.
2.  **Fill**: Fills remainder with top composite score.
3.  **Verification**: LLM removes irrelevant papers ("false positives").

### `CitationGapFinder` ([citation_gap_finder.py](citation_gap_finder.py))
Ensures bibliography completeness.
- **Identify**: LLM suggests missing foundational works or other important references.
- **Fetch**: Automatically searches for and adds these missing key papers via SemanticScholar.

### `UserPaperLoader` ([user_paper_loader.py](user_paper_loader.py))
- Loads user-provided PDFs.
- Matches to Semantic Scholar or arXiv via ID or LLM-extracted title.


### `SemanticScholarAPI` ([semantic_scholar_api.py](semantic_scholar_api.py))
- Wraps Semantic Scholar API

## Output
- `output/search_queries.json`: Generated queries.
- `output/papers.json`: Metadata and rankings.
- `output/literature/`: Downloaded PDFs.
