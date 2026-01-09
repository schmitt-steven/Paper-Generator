import numpy as np
from typing import List, Dict
from phases.paper_search.paper import Paper

class PaperFilter:

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    @staticmethod
    def _calibrate_threshold(papers: List[Paper], sample_size: int = 50) -> float:
        """
        Dynamically determine what counts as 'Too Similar' for this specific model.
        
        Process:
        1. Take top N papers.
        2. Calculate pairwise similarities.
        3. The 'duplicate threshold' should be very high relative to the average.
        """
        if len(papers) < 2:
            return 0.9  # Default fallback

        # Use a subset
        sample = papers[:sample_size]
        embeddings = [np.array(p.title_abstract_embedding) for p in sample]
        
        # Calculate similarity distribution (just a random sample of pairs to save time)
        similarities = []
        for i in range(len(embeddings)):
            # Compare with just next neighbor, to get the "local" density
            if i + 1 < len(embeddings):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
                similarities.append(sim)
        
        if not similarities:
            return 0.9

        # Calculate statistics
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Idea: A "duplicate" is usually 1 Standard Deviation above the average neighbor
        # Safety net: absolute minimum 0.50, absolute max 0.98
        dynamic_threshold = avg_sim + (1.0 * std_sim)
        dynamic_threshold = max(0.50, min(dynamic_threshold, 0.98))
        
        #print(f"  > Calibration: Avg Sim={avg_sim:.3f}, Std={std_sim:.3f} -> Dynamic Threshold={dynamic_threshold:.3f}")
        return dynamic_threshold

    @staticmethod
    def _cluster_papers(papers: List[Paper], threshold: float) -> Dict[str, List[Paper]]:
        """Groups papers using the Leader Algorithm with the dynamic threshold."""
        # Sort by final score so the best paper leads the cluster its in
        sorted_papers = sorted(
            papers, 
            key=lambda p: p.ranking.final_score if p.ranking else 0, 
            reverse=True
        )

        clusters: Dict[str, List[Paper]] = {}
        
        for paper in sorted_papers:
            assigned = False
            for leader_id, cluster_papers in clusters.items():
                leader = cluster_papers[0]
                sim = PaperFilter._cosine_similarity(paper.title_abstract_embedding, leader.title_abstract_embedding)
                
                if sim >= threshold:
                    clusters[leader_id].append(paper)
                    assigned = True
                    break
            
            if not assigned:
                clusters[paper.id] = [paper]
                
        return clusters

    @staticmethod
    def run(
        papers: List[Paper],
        target_count: int = 50,
        max_per_cluster: int = 5,
        min_relevance: float = 0.4
    ) -> List[Paper]:
        
        if not papers: return []
        
        # Remove duplicates and quality filter
        unique_map = {}
        for p in papers:
            current_score = p.ranking.final_score if p.ranking else 0
            if current_score < min_relevance: continue
            
            if p.id not in unique_map:
                unique_map[p.id] = p
            else:
                existing_score = unique_map[p.id].ranking.final_score if unique_map[p.id].ranking else 0
                if current_score > existing_score:
                    unique_map[p.id] = p
        
        qualified_papers = list(unique_map.values())
        # Sort for calibration step
        qualified_papers.sort(key=lambda p: p.ranking.final_score if p.ranking else 0, reverse=True)
        
        print(f"Filtering: {len(qualified_papers)} papers qualified for clustering.")
        
        # Auto-calibration (get threshold based on this specific batches embeddings)
        similarity_threshold = PaperFilter._calibrate_threshold(qualified_papers)

        # Clustering
        clusters = PaperFilter._cluster_papers(qualified_papers, similarity_threshold)
        print(f"Clustering: Grouped into {len(clusters)} topics using threshold {similarity_threshold:.3f}")
        
        # Ticket System (Weighted Selection)
        # Give high-scoring clusters more tickets/slots, pool all papers, then sort all papers by quality
        candidates = []
        quota_stats = {3: 0, 2: 0, 1: 0}
        
        for cid, cluster_papers in clusters.items():
            if not cluster_papers: 
                continue
            
            # Determine quota based on leader's score
            leader_score = cluster_papers[0].ranking.final_score if cluster_papers[0].ranking else 0
            
            if leader_score >= 0.78:
                quota = 3
            elif leader_score >= 0.72:
                quota = 2
            else:
                quota = 1
            
            quota_stats[quota] += 1
            
            # Take top n papers from this cluster
            candidates.extend(cluster_papers[:quota])

        print(f"  > Quotas: {quota_stats[3]} clusters@3, {quota_stats[2]} clusters@2, {quota_stats[1]} clusters@1")
        
        # Global sort by quality, so lower ranked papers from great clusters beat leaders of weak ones
        candidates.sort(key=lambda p: p.ranking.final_score if p.ranking else 0, reverse=True)
        
        # Cut to target
        selected_papers = candidates[:target_count]
        
        print(f"Selection: Kept top {len(selected_papers)} from {len(candidates)} candidates.")
        return selected_papers
    
    @staticmethod
    def verify_with_llm(
        papers: List[Paper],
        research_context: str,
        model_name: str,
        batch_size: int = 10
    ) -> List[Paper]:
        """Use LLM to filter obvious false positives (papers in completely wrong fields)."""
        import lmstudio as lms
        from pydantic import BaseModel
        
        # Pydantic schema for structured response
        class VerificationResult(BaseModel):
            """Papers to keep from the batch."""
            keep: List[int]  # Paper numbers to keep
        
        if not papers:
            return []
        
        print(f"\nLLM Verification: Checking {len(papers)} papers in batches of {batch_size}...")
        
        model = lms.llm(model_name)
        verified_papers = []
        total_removed = 0
        
        # Process in batches
        for batch_start in range(0, len(papers), batch_size):
            batch_end = min(batch_start + batch_size, len(papers))
            batch = papers[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size
            
            papers_text = ""
            for i, paper in enumerate(batch, 1):
                title = paper.title or "Untitled"
                abstract = paper.summary or "No abstract available"
                papers_text += f"<paper_{i}>\n{title}\n{abstract}\n</paper_{i}>\n\n"
            
            prompt = f"""TASK:
                You are selecting papers for a research literature review. DEFAULT: KEEP EVERYTHING unless it's obviously wrong.

                RESEARCH TOPIC:
                {research_context}

                PAPERS TO EVALUATE:
                {papers_text}

                YOUR DEFAULT DECISION: KEEP the paper. Only reject if it's OBVIOUSLY from a completely unrelated field.

                KEEP these (give them a spot):
                - ANY paper directly addressing the research topic or its core methods
                - ANY survey or review paper covering related areas
                - ANY benchmark or evaluation paper for methods in this domain
                - ANY theoretical/foundational paper that informs the research topic
                - ANY paper that could possibly be cited in a Related Work section
                - When in doubt, KEEP IT

                REMOVE ONLY these (truly obvious garbage):
                - Papers from completely unrelated application domains with no methodological connection
                - Pure domain-specific papers (e.g., clinical trials, material properties, ecological surveys) that don't use any computational or analytical methods relevant to the research topic
                - Papers where the field overlap is purely superficial (e.g., uses a keyword but in a completely different context)

                If a paper's methods, theory, or contributions could reasonably support the research topic above - KEEP IT.

                Return the paper numbers (1-{len(batch)}) that EARN a spot (should be MOST of them)."""

            try:
                response = model.respond(
                    prompt, 
                    response_format=VerificationResult,
                    config={"temperature": 0.0}
                )
                result = response.parsed
                numbers_to_keep = set(result["keep"])
                
                # Filter the batch - keep what's selected
                batch_kept = 0
                for i, paper in enumerate(batch, 1):
                    if i in numbers_to_keep:
                        verified_papers.append(paper)
                        batch_kept += 1
                    else:
                        total_removed += 1
                        print(f"  [REMOVED] {paper.title[:60]}...")
                
                print(f"  Batch {batch_num}/{total_batches}: Kept {batch_kept}/{len(batch)}")
                
            except Exception as e:
                print(f"  Batch {batch_num}/{total_batches}: LLM error, keeping all papers - {e}")
                verified_papers.extend(batch)
        
        print(f"LLM Verification: Removed {total_removed} papers, kept {len(verified_papers)}")
        return verified_papers