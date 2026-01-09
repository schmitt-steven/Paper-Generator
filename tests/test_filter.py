"""Test script for the clustering-based paper filter."""
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper_filter import PaperFilter
from phases.paper_search.paper_ranking import PaperRanker
from phases.context_analysis.paper_conception import PaperConception
from settings import Settings

def main():
    # Load unfiltered papers
    print("=" * 60)
    print("Loading papers_unfiltered.json...")
    papers = LiteratureSearch.load_papers("output/papers_unfiltered.json")
    print(f"Loaded {len(papers)} papers\n")
    
    # Check if embeddings exist
    has_embeddings = all(getattr(p, 'title_abstract_embedding', None) is not None for p in papers)
    
    if not has_embeddings:
        print("Papers don't have embeddings stored. Re-computing...")
        paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
        ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
        ranking_context = f"{paper_concept.description}\nOpen Research Questions:\n{paper_concept.open_questions}"
        papers = ranker.rank_papers(
            papers=papers,
            context=ranking_context,
            weights={'relevance': 0.7, 'citations': 0.2, 'recency': 0.1}
        )
        # Save with embeddings for future runs
        LiteratureSearch.save_papers(papers, filename="papers_unfiltered.json", output_dir="output")
        print("Saved papers with embeddings.\n")
    
    # Run the new filter with verbose output
    print("=" * 60)
    print("Running clustering filter...")
    print("=" * 60)
    
    # Step 1: Filter and calibrate (print happens inside)
    unique_map = {}
    min_relevance = 0.4
    for p in papers:
        current_score = p.ranking.final_score if p.ranking else 0
        if current_score < min_relevance:
            continue
        if p.id not in unique_map:
            unique_map[p.id] = p
        else:
            existing_score = unique_map[p.id].ranking.final_score if unique_map[p.id].ranking else 0
            if current_score > existing_score:
                unique_map[p.id] = p
    
    qualified_papers = list(unique_map.values())
    qualified_papers.sort(key=lambda p: p.ranking.final_score if p.ranking else 0, reverse=True)
    
    print(f"\nQualified papers (score >= {min_relevance}): {len(qualified_papers)}")
    
    # Step 2: Calibrate threshold
    print("\n--- Calibration ---")
    similarity_threshold = PaperFilter._calibrate_threshold(qualified_papers)
    
    # Step 3: Cluster
    print("\n--- Clustering ---")
    clusters = PaperFilter._cluster_papers(qualified_papers, similarity_threshold)
    print(f"Found {len(clusters)} clusters\n")
    
    # Step 4: Show cluster details
    print("=" * 60)
    print("CLUSTER DETAILS")
    print("=" * 60)
    
    # Sort clusters by leader score
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: x[1][0].ranking.final_score if x[1][0].ranking else 0,
        reverse=True
    )
    
    for i, (leader_id, cluster_papers) in enumerate(sorted_clusters, 1):
        leader = cluster_papers[0]
        leader_score = leader.ranking.final_score if leader.ranking else 0
        print(f"\nCluster {i} ({len(cluster_papers)} papers) - Score: {leader_score:.3f}")
        print(f"  Leader: {leader.title[:70]}...")
        
        if len(cluster_papers) > 1:
            print(f"  Other papers:")
            for p in cluster_papers[1:]:
                p_score = p.ranking.final_score if p.ranking else 0
                print(f"    - [{p_score:.3f}] {p.title[:60]}...")
    
    # =============================================
    # Run New Cluster Filter + LLM Verification
    # =============================================
    print("\n" + "=" * 60)
    print("CLUSTER FILTER SELECTION")
    print("=" * 60)
    
    selected = PaperFilter.run(papers, target_count=50, max_per_cluster=5, min_relevance=0.4)
    
    print(f"\nSelected {len(selected)} papers:")
    for i, p in enumerate(selected, 1):
        score = p.ranking.final_score if p.ranking else 0
        print(f"  {i:2d}. [{score:.3f}] {p.title[:60]}...")
    
    # LLM Verification
    print("\n" + "=" * 60)
    print("LLM VERIFICATION (removing false positives)")
    print("=" * 60)
    
    paper_concept = PaperConception.load_paper_concept("output/paper_concept.md")
    research_context = f"{paper_concept.description}\n\nOpen Research Questions:\n{paper_concept.open_questions}"
    
    verified = PaperFilter.verify_with_llm(
        papers=selected,
        research_context=research_context,
        model_name=Settings.LITERATURE_SEARCH_MODEL,
        batch_size=10
    )
    
    print(f"\nFinal verified papers: {len(verified)}")
    for i, p in enumerate(verified, 1):
        score = p.ranking.final_score if p.ranking else 0
        print(f"  {i:2d}. [{score:.3f}] {p.title[:60]}...")
    
    # Save
    print("\n" + "=" * 60)
    output_file = "papers_filtered_test.json"
    LiteratureSearch.save_papers(verified, filename=output_file, output_dir="output")
    print(f"Saved {len(verified)} papers to output/{output_file}")

if __name__ == "__main__":
    main()
