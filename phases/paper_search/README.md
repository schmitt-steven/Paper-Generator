# Paper Search Phase

Searches, ranks, filters, and downloads academic papers.

## Components

- **`LiteratureSearch`**: Generates search queries and searches arXiv
- **`PaperRanker`**: Ranks papers by relevance, citations, and recency
- **`PaperFilter`**: Filters papers for diverse selection
- **`SemanticScholarAPI`**: Interfaces with Semantic Scholar API

## Process

1. Generate search queries from paper concept
2. Search Semantic Scholar for papers
3. Rank papers
4. Filter for diverse paper types
5. Download PDFs and convert to markdown

## Output

- `output/search_queries.json` - Generated queries
- `output/papers.json` - All found papers
- `output/papers_filtered_with_markdown.json` - Filtered papers with markdown
- `output/literature/` - Downloaded PDFs and markdown files

