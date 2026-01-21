# Paper Search Phase

Searches, ranks, filters, and downloads academic papers.

## Components

### `Paper` ([paper.py](paper.py))

Data class representing an academic paper with metadata, ranking scores, and citation info.

### `SemanticScholarAPI` ([semantic_scholar_api.py](semantic_scholar_api.py))

Wrapper of the Semantic Scholar API for searching and fetching papers.

### `UserPaperLoader` ([user_paper_loader.py](user_paper_loader.py))

Processes user-provided PDFs, copies them to `output/literature/user_{filename}/`, and fetches metadata from Semantic Scholar.

### `LiteratureSearch` ([literature_search.py](literature_search.py))

Handles the paper search process:
- Generates search queries from paper concept using LLM
- Executes searches via Semantic Scholar API
- Removes duplicates and merges user papers with searched papers
- Downloads PDFs

### `PaperRanker` ([paper_ranking.py](paper_ranking.py))

Ranks papers using embedding similarity and composite scoring:
- Semantic relevance (embedding similarity to paper concept)
- Citation score (age-aware citation impact)
- Recency score

### `PaperFilter` ([paper_filter.py](paper_filter.py))

Filters papers for a diverse selection across categories:
- "High Relevance" (top 10% by relevance, always included)
- "Cutting Edge" (recent + highly relevant)
- "Hidden Gems" (high relevance + low citations)
- "Classics" (high citations + moderate relevance)
- "Well-Rounded" (balanced across all metrics)

## Output

- `output/search_queries.json` - Generated search queries
- `output/papers.json` - All papers with metadata and rankings
- `output/literature/` - Downloaded PDFs
