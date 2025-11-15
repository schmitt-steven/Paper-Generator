from phases.paper_search.arxiv_api import Paper
from typing import List
import numpy as np
from datetime import datetime


class PaperFilter:
    """
    Filters papers based on various strategies after ranking.
    All methods are static - no state needed.
    """
    
    @staticmethod
    def filter_by_score(
        papers: List[Paper],
        min_score: float = 0.3,
        top_k: int = None
    ) -> List[Paper]:
        """Filter papers by minimum score threshold or keep top-K."""
        if not papers:
            return []
        
        # Filter to only ranked papers
        ranked_papers = [p for p in papers if p.ranking is not None]
        
        if top_k is not None:
            filtered = ranked_papers[:top_k]
            print(f"Filtering: Kept top {top_k} papers")
        else:
            filtered = [p for p in ranked_papers if p.ranking.final_score >= min_score]
            print(f"Filtering: Kept {len(filtered)}/{len(ranked_papers)} papers with score >= {min_score}")
        
        return filtered


    @staticmethod
    def filter_diverse(
        papers: List[Paper],
        n_cutting_edge: int = 20,
        n_hidden_gems: int = 15,
        n_classics: int = 15,
        n_well_rounded: int = 10
    ) -> List[Paper]:
        """
        Get diverse mix of papers across different categories.
        
        Priority: Top 10% by relevance score (always included)
        Categories:
        - Cutting Edge: Recent  + highly relevant (relevance > 0.65)
        - Hidden Gems: High relevance (> 0.7) + low citations (< 0.3)
        - Classics: High citations (> 0.6) + moderate relevance (> 0.4)
        - Well-Rounded: High across all metrics
        """
        # Filter to only ranked papers
        ranking_results = [p for p in papers if p.ranking is not None]
        
        if not ranking_results:
            return []
        
        selected_papers = []
        selected_ids = set()
        
        # Priority: Always keep top 10% by relevance score
        # Sort all papers by relevance and take the top 10%
        sorted_by_relevance = sorted(ranking_results, key=lambda x: x.ranking.relevance_score, reverse=True)
        top_10_percent_count = max(1, int(len(sorted_by_relevance) * 0.10))
        high_relevance = sorted_by_relevance[:top_10_percent_count]
        
        # Sort high relevance papers by final score
        high_relevance.sort(key=lambda x: x.ranking.final_score, reverse=True)
        for paper in high_relevance:
            selected_papers.append(paper)
            selected_ids.add(paper.id)
        
        # OLD: Fixed threshold approach
        # high_relevance = [
        #     p for p in ranking_results 
        #     if p.ranking.relevance_score > 0.9
        # ]
        # high_relevance.sort(key=lambda x: x.ranking.final_score, reverse=True)
        # for paper in high_relevance:
        #     selected_papers.append(paper)
        #     selected_ids.add(paper.id)
        
        # Category 1: Cutting Edge
        cutting_edge = [
            p for p in ranking_results 
            if p.ranking.recency_score > 0.6 and p.ranking.relevance_score > 0.8
            and p.id not in selected_ids
        ]
        cutting_edge.sort(key=lambda x: x.ranking.final_score, reverse=True)
        for paper in cutting_edge[:n_cutting_edge]:
            selected_papers.append(paper)
            selected_ids.add(paper.id)
        
        # Category 2: Hidden Gems
        hidden_gems = [
            p for p in ranking_results 
            if p.ranking.relevance_score > 0.8 and p.ranking.citation_score < 0.3
            and p.id not in selected_ids
        ]
        hidden_gems.sort(key=lambda x: x.ranking.relevance_score, reverse=True)
        for paper in hidden_gems[:n_hidden_gems]:
            selected_papers.append(paper)
            selected_ids.add(paper.id)
        
        # Category 3: Classics
        classics = [
            p for p in ranking_results 
            if p.ranking.citation_score > 0.7 and p.ranking.relevance_score > 0.6
            and p.id not in selected_ids
        ]
        classics.sort(key=lambda x: x.ranking.citation_score, reverse=True)
        for paper in classics[:n_classics]:
            selected_papers.append(paper)
            selected_ids.add(paper.id)
        
        # Category 4: Well-Rounded
        well_rounded = [
            p for p in ranking_results 
            if (p.ranking.relevance_score > 0.7 and
                p.ranking.citation_score > 0.4 and
                p.ranking.recency_score > 0.5)
            and p.id not in selected_ids
        ]
        well_rounded.sort(key=lambda x: x.ranking.final_score, reverse=True)
        for paper in well_rounded[:n_well_rounded]:
            selected_papers.append(paper)
            selected_ids.add(paper.id)

        # Sort: top 10% by relevance first, then rest by final_score
        min_high_relevance_score = min([p.ranking.relevance_score for p in high_relevance]) if high_relevance else 1.0
        selected_papers.sort(key=lambda p: (p.ranking.relevance_score < min_high_relevance_score, -p.ranking.final_score))
        
        # Print summary
        actual_high_relevance = len(high_relevance)
        actual_cutting_edge = min(len([p for p in cutting_edge if p.id in selected_ids]), n_cutting_edge)
        actual_hidden_gems = min(len([p for p in hidden_gems if p.id in selected_ids]), n_hidden_gems)
        actual_classics = min(len([p for p in classics if p.id in selected_ids]), n_classics)
        actual_well_rounded = min(len([p for p in well_rounded if p.id in selected_ids]), n_well_rounded)
        
        print(f"\nFiltering (diverse): Selected {len(selected_papers)} papers")
        print(f"  High Relevance: {actual_high_relevance} (top 10% by relevance, always included)")
        print(f"  Cutting Edge: {actual_cutting_edge}/{n_cutting_edge}")
        print(f"  Hidden Gems:  {actual_hidden_gems}/{n_hidden_gems}")
        print(f"  Classics:     {actual_classics}/{n_classics}")
        print(f"  Well-Rounded: {actual_well_rounded}/{n_well_rounded}\n")
        
        return selected_papers

