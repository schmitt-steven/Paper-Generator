from phases.paper_search.arxiv_api import Paper, RankingScores
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime
from utils.lazy_model_loader import LazyEmbeddingMixin


class PaperRanker(LazyEmbeddingMixin):
    """
    Ranks papers by composite score based on:
    - Semantic relevance (embedding similarity to context)
    - Citation count (logarithmic scale)
    - Recency (exponential decay)
    """
    
    def __init__(self, embedding_model_name: str = "text-embedding-embeddinggemma-300m"):
        """
        Initialize PaperRanker with an embedding model.
        
        Args:
            embedding_model_name: Name of the LM Studio embedding model
        """
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None  # Lazy-loaded via LazyEmbeddingMixin
    
    def rank_papers(
        self,
        papers: List[Paper],
        context: str,
        weights: Dict[str, float] = None
    ) -> List[Paper]:
        """
        Rank papers by composite score and populate their ranking field.
        
        Args:
            papers: List of Paper objects to rank
            context: Research context string to compare against
            weights: Dict with keys 'relevance', 'citations', 'recency' (default: 0.6, 0.2, 0.2)
        
        Returns:
            List of Paper objects with ranking field populated, sorted by final_score descending (best first)
        """
        if weights is None:
            weights = {
                'relevance': 0.8,
                'citations': 0.1,
                'recency': 0.1
            }
        
        if not papers:
            return []
        
        print(f"Ranking {len(papers)} papers...")
        
        print("Embedding context...")
        context_emb = self.embedding_model.embed(context)
        
        # 2. Embed all papers (title + abstract)
        print("Embedding papers...")
        paper_texts = [f"{p.title} {p.summary}" for p in papers]
        paper_embs = [self.embedding_model.embed(text) for text in paper_texts]
        
        # 3. Calculate scores for each paper
        for paper, paper_emb in zip(papers, paper_embs):
            # Relevance score (0-1): cosine similarity
            relevance = self._cosine_similarity(context_emb, paper_emb)
            
            # Citation score (0-1): age-aware logarithmic scale
            citation_score = self._normalize_citations_age_aware(
                paper.citation_count, 
                paper.published
            )
            
            # Recency score (0-1): exponential decay
            recency_score = self._calculate_recency(paper.published)
            
            # Composite score
            final_score = (
                weights['relevance'] * relevance +
                weights['citations'] * citation_score +
                weights['recency'] * recency_score
            )
            
            # Set ranking scores on the paper object
            paper.ranking = RankingScores(
                relevance_score=relevance,
                citation_score=citation_score,
                recency_score=recency_score,
                final_score=final_score
            )
        
        # Sort by final score descending (highest score first)
        papers.sort(key=lambda x: x.ranking.final_score, reverse=True)
        
        print(f"Ranking complete (score range: {papers[-1].ranking.final_score:.3f} to {papers[0].ranking.final_score:.3f})")
        
        return papers
    
    def _cosine_similarity(self, emb1, emb2) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Returns: Similarity score (0-1 range after normalization)
        """
        # Convert to numpy arrays if needed
        if not isinstance(emb1, np.ndarray):
            emb1 = np.array(emb1)
        if not isinstance(emb2, np.ndarray):
            emb2 = np.array(emb2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def _normalize_citations_age_aware(
        self, 
        citations: int, 
        published: str
    ) -> float:
        """
        Age-aware citation normalization: rewards impact velocity.
        RELAXED version: more papers reach 0.6+ range.
        """
        pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
        current_date = datetime.now(pub_date.tzinfo) if pub_date.tzinfo else datetime.now()
        age_years = max((current_date - pub_date).days / 365.25, 0.1)
        
        if citations is None:
            return 0.5 if age_years < 2.0 else 0.2
        
        cites_per_year = citations / age_years
        
        # Target: ~30 cites/year = 0.9, ~5 cites/year = 0.6, ~1 cite/year = 0.4
        if cites_per_year >= 30:
            velocity_score = 0.9
        elif cites_per_year >= 10:
            velocity_score = 0.7 + 0.2 * (cites_per_year - 10) / 20
        elif cites_per_year >= 3:
            velocity_score = 0.5 + 0.2 * (cites_per_year - 3) / 7     
        elif cites_per_year >= 0.5:
            velocity_score = 0.3 + 0.2 * (cites_per_year - 0.5) / 2.5
        else:
            velocity_score = 0.1 + 0.2 * (cites_per_year / 0.5)
        
        if age_years < 0.5 and citations >= 3:
            recency_bonus = 0.03
        elif age_years < 1.0 and citations >= 5: 
            recency_bonus = 0.02
        else:
            recency_bonus = 0.0
        
        # Maturity penalty for dead papers
        if age_years > 4.0 and cites_per_year < 0.5:
            maturity_penalty = 0.2
        else:
            maturity_penalty = 0.0
        
        score = velocity_score + recency_bonus - maturity_penalty
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_recency(self, published: str) -> float:
        """
        Calculate recency score with 4-year half-life.
        
        Scale:
        - 0 years old → 1.0
        - 1 year old → 0.84
        - 2 years old → 0.71
        - 4 years old → 0.5
        - 8 years old → 0.25
        
        Returns:
            Recency score (0-1)
        """
        pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
        current_date = datetime.now(pub_date.tzinfo) if pub_date.tzinfo else datetime.now()
        age_years = (current_date - pub_date).days / 365.25
        
        recency = 2 ** (-age_years / 4)
        
        return recency
    
    @staticmethod
    def print_ranked_papers(papers: List[Paper], n: int = 10):
        """Print top N ranked papers by relevance and by final score."""
        if not papers:
            print("No papers to display")
            return
        
        # Filter papers that have ranking scores
        ranked_papers = [p for p in papers if p.ranking is not None]
        
        if not ranked_papers:
            print("No ranked papers to display")
            return
        
        # Section 1: Top by relevance
        by_relevance = sorted(ranked_papers, key=lambda p: p.ranking.relevance_score, reverse=True)
        print(f"\n{'='*30}")
        print(f"Top {min(n, len(by_relevance))} papers by relevance:")
        print(f"{'='*30}")
        
        for i, paper in enumerate(by_relevance[:n], 1):
            print(f"{i}. {paper.title}")
            print(f"   score: {paper.ranking.final_score:.3f} (rel: {paper.ranking.relevance_score:.2f}, cit: {paper.ranking.citation_score:.2f}, rec: {paper.ranking.recency_score:.2f})")
            print(f"   published: {paper.published[:10]}, citations: {paper.citation_count or 0}")
            print()
        
        # Section 2: Top by final score
        by_final_score = sorted(ranked_papers, key=lambda p: p.ranking.final_score, reverse=True)
        print(f"{'='*30}")
        print(f"Top {min(n, len(by_final_score))} papers by final score:")
        print(f"{'='*30}")
        
        for i, paper in enumerate(by_final_score[:n], 1):
            print(f"{i}. {paper.title}")
            print(f"   score: {paper.ranking.final_score:.3f} (rel: {paper.ranking.relevance_score:.2f}, cit: {paper.ranking.citation_score:.2f}, rec: {paper.ranking.recency_score:.2f})")
            print(f"   published: {paper.published[:10]}, citations: {paper.citation_count or 0}")
            print()

