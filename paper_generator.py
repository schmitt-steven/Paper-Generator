import traceback
from typing import List
from phases.context_analysis.paper_conception import PaperConcept, PaperConception
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from phases.context_analysis.user_notes_analysis import NotesAnalyzer
from phases.literature_review.arxiv_api import Paper
from phases.literature_review.literature_search import LiteratureSearch
from phases.literature_review.paper_ranking import PaperRanker
from phases.literature_review.paper_filtering import PaperFilter
from utils.pdf_converter_pymupdf_marker import PDFConverter
from phases.hypothesis_generation.paper_analysis import PaperAnalyzer
from phases.hypothesis_generation.limitation_analysis import LimitationAnalyzer
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from settings import (
    EVIDENCE_GATHERING_MODEL,
    PAPER_INDEXING_EMBEDDING_MODEL,
    PAPER_WRITING_MODEL,
)


class PaperGenerator: 
    
    def generate_paper(self):

        ###########################
        # Step 1/: Analyze context #
        ###########################

        # # Analyze all code files
        # code_analyzer = CodeAnalyzer(model_name="qwen/qwen3-coder-30b")
        # code_files = code_analyzer.load_code_files("user_files")
        # analyzed_files = code_analyzer.analyze_all_files(code_files)
        # #print(code_analyzer.get_analysis_report(analyzed_files))

        # # Analyze all notes (.md, .txt etc.)
        # notes_analyzer = NotesAnalyzer(model_name="qwen/qwen3-coder-30b")
        # notes = notes_analyzer.load_user_notes("user_files")
        # analyzed_notes = notes_analyzer.analyze_all_user_notes(notes)
        # #print(notes_analyzer.get_analysis_report(analyzed_notes))

        # # Generate paper outline and save to markdown file
        # concept_builder = PaperConception(
        #     model_name="qwen/qwen3-coder-30b",
        #     user_code=analyzed_files,
        #     user_notes=analyzed_notes
        # )
        # paper_concept = concept_builder.build_paper_concept()
        # concept_builder.save_paper_concept(paper_concept, filename="paper_concept.md", output_dir="output")
        

        # ALTERNATIVE: Load existing paper concept
        paper_concept: PaperConcept = PaperConception.load_paper_concept("output/paper_concept.md")

        ################################
        # Step 2: Build search queries #       
        ################################

        literature_search = LiteratureSearch(model_name="qwen/qwen3-coder-30b")
        
        # Build search queries based on paper concept and open questions
        #search_queries = literature_search.build_search_queries([paper_concept.description, paper_concept.open_questions])
        #literature_search.save_search_queries(search_queries, filename="search_queries.json")
        
        # ALTERNATIVE: Load existing search queries
        # search_queries = LiteratureSearch.load_search_queries("output/search_queries.json")

        ############################
        # Step 3: Execute searches #       
        ############################

        # Execute searches via arXiv API (with rate limiting built-in)
        # all_papers = literature_search.search_papers(search_queries, max_results_per_query=30)
        # all_papers = literature_search.get_citation_counts(all_papers)
        #all_papers = literature_search.get_bibtex_for_papers(all_papers)
        #literature_search.save_papers(all_papers, filename="papers.json")
        
        # ALTERNATIVE:Load existing papers
        # all_papers: List[Paper] = LiteratureSearch.load_papers("output/papers.json")

        ############################################
        # Step 4: Rank, filter and download papers #       
        ############################################

        # Rank papers based on embedding similarity, citation counts, and publication date
        # The embeddings of the papers are based on title and abstract.
        # ranker = PaperRanker(embedding_model_name="text-embedding-embeddinggemma-300m")
        # ranking_context = f"{paper_concept.description}\nOpen Research Questions:\n{paper_concept.open_questions}"
        # ranked_papers: List[Paper] = ranker.rank_papers(
        #     papers=all_papers,
        #     context=ranking_context,
        #     weights={
        #         'relevance': 0.7,
        #         'citations': 0.2,
        #         'recency': 0.1
        #     }
        # )
        
        # # Filter the ranked papers
        # filtered_papers: List[Paper] = PaperFilter.filter_diverse(
        #     papers=ranked_papers,
        #     n_cutting_edge=15,
        #     n_hidden_gems=15,
        #     n_classics=15,
        #     n_well_rounded=20
        # )
        # PaperRanker.print_ranked_papers(filtered_papers, n=10)

        # Download papers
        # literature_search.download_papers(filtered_papers, base_folder="literature/")

        ###############################################
        # Step 5: Convert papers from PDF to Markdown #       
        ###############################################

        # Convert papers to markdown and update markdown_text field (if needed)
        # converter = PDFConverter(fix_math=False, extract_media=True)
        # papers_with_markdown: List[Paper] = converter.convert_all_papers(filtered_papers, base_folder="literature/")
        # literature_search.save_papers(papers_with_markdown, filename="papers_filtered_with_markdown.json")

        # ALTERNATIVE: Load set of papers that are already ranked, filtered and have markdown
        loaded_papers: List[Paper] = LiteratureSearch.load_papers("output/papers_filtered_with_markdown.json")
        papers_with_markdown: List[Paper] = [p for p in loaded_papers if getattr(p, "markdown_text", None) and p.markdown_text.strip()]

        ###############################
        # Step 6: Generate hypotheses #       
        ###############################
        
        # Filter again to get a good mix of paper types for hypothesis generation
        # print(f"\nFiltering {len(papers_with_markdown)} papers for hypothesis generation...")
        # top_papers: List[Paper] = PaperFilter.filter_diverse(
        #     papers=papers_with_markdown,
        #     n_cutting_edge=5,
        #     n_hidden_gems=5,
        #     n_classics=5,
        #     n_well_rounded=5
        # )
        # print(f"Selected {len(top_papers)} papers for hypothesis generation")
        
        # Extract findings from papers
        #analyzer = PaperAnalyzer(model_name="qwen/qwen3-coder-30b")
        #findings = analyzer.extract_findings(top_papers)
        
        # Save findings
        #PaperAnalyzer.save_findings(findings, "output/paper_findings.json")
        
        # ALTERNATIVE: Load previously saved findings
        findings = PaperAnalyzer.load_findings("output/paper_findings.json")
        
        # Analyze literature limitations
        # limitation_analyzer = LimitationAnalyzer.build_from_findings(findings, paper_concept)
        # limitation_analyzer.print_limitations(n=10, show_scores=True)
        # top_limitations = limitation_analyzer.find_top_limitations(n=10)
        # limitation_analyzer.save_limitations(top_limitations, "output/limitations.json", "output/paper_concept.md", len(findings))
        
        # ALTERNATIVE: Load existing limitations
        limitation_analyzer = LimitationAnalyzer()
        top_limitations = limitation_analyzer.load_limitations("output/limitations.json")
        
        # Generate and validate hypotheses
        hypothesis_builder = HypothesisBuilder(
            model_name="qwen/qwen3-coder-30b",
            embedding_model_name="text-embedding-embeddinggemma-300m",
            paper_concept=paper_concept,
            top_limitations=top_limitations,
            num_papers_analyzed=len(findings)
        )
        
        # hypotheses = hypothesis_builder.generate_hypotheses(n_hypotheses=5)
        # Save hypotheses
        # hypothesis_builder.save_hypotheses(hypotheses, "output/hypotheses.json")


        # ALTERNATIVE: Load existing hypotheses
        hypotheses = hypothesis_builder.load_hypotheses("output/hypotheses.json")

        best_hypothesis = hypothesis_builder.select_best_hypotheses(hypotheses, max_n=1)
        
        ################################
        # Step 7: Generate experiments #       
        ###############################
        
        experiment_runner = ExperimentRunner(model_name="qwen/qwen3-coder-30b")
        
        experiment_result = None
        print(f"\nTesting hypothesis {best_hypothesis.id}: {best_hypothesis.description}")
        try:
            result = experiment_runner.run_experiment(
                best_hypothesis, 
                paper_concept, 
                load_existing_plan=False,
                load_existing_code=False
            )
            print(f"  Experiment completed")
            print(f"  Verdict: {result.hypothesis_evaluation.verdict}")
            print(f"  Reasoning: {result.hypothesis_evaluation.reasoning}")
            experiment_result = result
        except Exception as e:
            print(f"\nError running experiment: {e}")
            traceback.print_exc()
       
        # TODO: Rework the paper concept content
        # IDEA: have a fallback if context window is full (e.g. only paste code if context window is full OR only paste func signatures instead of full code if context window is full)

        ##############################
        # Step 8: Write paper        #
        ##############################

        # TODO: Test specialized embedding model and writing model (e.g. from CycleResearcher paper)

        if experiment_result and papers_with_markdown:
            paper_writing_pipeline = PaperWritingPipeline(
                writer_model_name=PAPER_WRITING_MODEL,
                embedding_model_name=PAPER_INDEXING_EMBEDDING_MODEL,
                evidence_model_name=EVIDENCE_GATHERING_MODEL,
            )

            paper_sections = paper_writing_pipeline.generate_paper(
                context=paper_concept,
                experiment=experiment_result,
                papers=papers_with_markdown,
            )
        else:
            print("Can't write paper: missing experiment results or indexed papers.")

        ##############################
        # Step 9: Convert paper      #       
        ##############################
        
if __name__ == "__main__":
    generator = PaperGenerator()
    generator.generate_paper()