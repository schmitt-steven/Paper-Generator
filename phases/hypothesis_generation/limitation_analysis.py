import numpy as np
import json
import os
from typing import List, Tuple
from collections import Counter
from phases.context_analysis.paper_conception import PaperConcept
from phases.hypothesis_generation.hypothesis_models import PaperFindings
from utils.lazy_model_loader import LazyEmbeddingMixin


class LimitationAnalyzer(LazyEmbeddingMixin):
    """Analyzes literature to identify research limitations and opportunities"""
    
    def __init__(self, embedding_model_name: str = None, similarity_threshold: float = 0.8):
        from settings import Settings
        self.embedding_model_name = embedding_model_name or Settings.LIMITATION_ANALYSIS_EMBEDDING_MODEL
        self._embedding_model = None  # Lazy-loaded via LazyEmbeddingMixin
        self.similarity_threshold = similarity_threshold  # Threshold for clustering similar limitations
    
    @staticmethod
    def build_from_findings(findings: List[PaperFindings], paper_concept: PaperConcept, similarity_threshold: float = 0.8) -> 'LimitationAnalyzer':
        """Build a LimitationAnalyzer instance with aggregated findings from papers"""
        analyzer = LimitationAnalyzer(similarity_threshold=similarity_threshold)
        analyzer.findings = findings
        analyzer.paper_concept = paper_concept
        
        # Aggregate limitations (raw counts)
        print("Identifying limitations from findings...")
        raw_limitations = Counter()
        for finding in findings:
            if finding.main_limitations:
                raw_limitations[finding.main_limitations] += 1
        
        # Cluster similar limitations together
        print(f"Filtering {len(raw_limitations)} limitations...")
        analyzer.common_limitations = analyzer._cluster_similar_limitations(raw_limitations)
        
        return analyzer
    
    def _cluster_similar_limitations(self, raw_limitations: Counter) -> Counter:
        """
        Cluster similar limitations using embedding similarity.
        
        Groups limitations with similarity > threshold together, keeping the most frequent one.
        """
        if not raw_limitations:
            return Counter()
        
        # Convert to list for processing
        limitations_list = list(raw_limitations.keys())
        counts = dict(raw_limitations)
        
        # Embed all limitations
        embeddings = {}
        for limitation in limitations_list:
            embeddings[limitation] = np.array(self.embedding_model.embed(limitation))
        
        # Track which limitations have been assigned to clusters
        clustered = set()
        clustered_limitations = Counter()
        
        # Greedy clustering: for each limitation, find similar ones
        for limitation in limitations_list:
            if limitation in clustered:
                continue
            
            # Start a new cluster with this limitation
            cluster = [limitation]
            cluster_count = counts[limitation]
            clustered.add(limitation)
            
            # Find similar limitations
            for other_lim in limitations_list:
                if other_lim in clustered or other_lim == limitation:
                    continue
                
                # Compute similarity
                emb1 = embeddings[limitation]
                emb2 = embeddings[other_lim]
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                if similarity >= self.similarity_threshold:
                    cluster.append(other_lim)
                    cluster_count += counts[other_lim]
                    clustered.add(other_lim)
            
            # Use the most frequent limitation in cluster as representative
            # (or first one if tied)
            cluster.sort(key=lambda x: counts[x], reverse=True)
            representative = cluster[0]
            clustered_limitations[representative] = cluster_count
        
        return clustered_limitations
    
    def score_limitation(self, limitation: str) -> float:
        """Score a research limitation based on frequency and relevance."""
        
        # Frequency score (inverse - fewer mentions = true limitation)
        max_count = max(self.common_limitations.values()) if self.common_limitations else 1
        limitation_count = self.common_limitations.get(limitation, 0)
        frequency_score = 1.0 - (limitation_count / max_count)
        
        # Relevance score (cosine similarity to paper concept)
        limitation_embedding = self.embedding_model.embed(limitation)
        concept_text = f"{self.paper_concept.description}"
        concept_embedding = self.embedding_model.embed(concept_text)
        
        # Cosine similarity
        limitation_emb = np.array(limitation_embedding)
        concept_emb = np.array(concept_embedding)
        similarity = np.dot(limitation_emb, concept_emb) / (np.linalg.norm(limitation_emb) * np.linalg.norm(concept_emb))
        relevance_score = (similarity + 1) / 2  # Normalize to [0, 1]
        
        # Weighted combination - relevance is highest priority to align limitations with paper concept
        final_score = 0.3 * frequency_score + 0.7 * relevance_score
        
        return final_score
    
    def find_top_limitations(self, n: int = 10) -> List[Tuple[str, float]]:
        """Find top N research limitations with scores"""
        if not self.common_limitations:
            return []
        
        scored_limitations = [(limitation, self.score_limitation(limitation)) for limitation in self.common_limitations.keys()]
        scored_limitations.sort(key=lambda x: x[1], reverse=True)
        top_limitations = scored_limitations[:n]
        
        # Automatically save
        if top_limitations:
            self.save_limitations(
                top_limitations,
                filepath="output/limitations.json",
                paper_concept_file="output/paper_concept.md",
                num_papers_analyzed=len(self.findings)
            )
        
        return top_limitations
    
    def print_limitations(self, n: int = 10, show_scores: bool = True, top_limitations: List[Tuple[str, float]] = None):
        """
        Print top N research limitations in a formatted way.
        
        Args:
            n: Number of top limitations to print (default: 10)
            show_scores: Whether to display scores alongside limitations (default: True)
            top_limitations: Optional pre-computed top limitations (to avoid duplicate computation)
        """
        if top_limitations is None:
            top_limitations = self.find_top_limitations(n=n)
        
        if not top_limitations:
            print("\nNo limitations found.")
            return
        
        print(f"\nTop {len(top_limitations)} research limitations:")
        print("-" * 80)
        
        for i, (limitation, score) in enumerate(top_limitations, 1):
            if show_scores:
                print(f"{i}. {limitation} (score: {score:.3f})")
            else:
                print(f"{i}. {limitation}")
        
        print("-" * 80)
        print(f"Analyzed {len(self.findings)} papers")
        print(f"Total unique limitations after clustering: {len(self.common_limitations)}")
    
    def save_limitations(self, top_limitations: List[Tuple[str, float]], filepath: str = "output/limitations.json", paper_concept_file: str = "output/paper_concept.md", num_papers_analyzed: int = 0):
        """Save limitations to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        limitations_data = {
            "paper_concept_file": paper_concept_file,
            "num_papers_analyzed": num_papers_analyzed,
            "limitations": [
                {
                    "limitation": limitation,
                    "score": score
                }
                for limitation, score in top_limitations
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(limitations_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(top_limitations)} limitations to {filepath}")
    
    def load_limitations(self, filepath: str = "output/limitations.json") -> List[Tuple[str, float]]:
        """Load limitations from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            limitations = [
                (item["limitation"], item["score"])
                for item in data.get("limitations", [])
            ]
            
            print(f"Loaded {len(limitations)} limitations from {filepath}")
            return limitations
        
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {filepath}: {e}")
            return []
        except Exception as e:
            print(f"Error loading limitations: {e}")
            return []

