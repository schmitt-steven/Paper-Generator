import traceback
from pathlib import Path
from typing import List
from phases.context_analysis.paper_conception import PaperConcept, PaperConception
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from phases.context_analysis.user_notes_analysis import NotesAnalyzer
from phases.paper_search.arxiv_api import Paper
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.paper_ranking import PaperRanker
from phases.paper_search.paper_filtering import PaperFilter
from phases.paper_writing.data_models import PaperDraft
from utils.pdf_converter_pymupdf_marker import PDFConverter
from phases.hypothesis_generation.paper_analysis import PaperAnalyzer
from phases.hypothesis_generation.limitation_analysis import LimitationAnalyzer
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.latex_generation.paper_converter import PaperConverter
from phases.latex_generation.metadata import LaTeXMetadata
from settings import Settings


class PaperGenerator: 
    
    def generate_paper(self):

        # TODO: use remove_thinking_blocks from llm_utils.py - or shit will break
        # TODO: Test paper concept until LLM gets it as MODEL BASED

        ###############################
        # Step 1/11: Paper Concept  #
        ###############################
        # TODO: Rework the paper concept content
        if Settings.LOAD_PAPER_CONCEPT:
            # Load existing paper concept
            paper_concept: PaperConcept = PaperConception.load_paper_concept("output/paper_concept.md")
        else:
            # Analyze all code files
            code_analyzer = CodeAnalyzer(model_name=Settings.CODE_ANALYSIS_MODEL)
            code_files = code_analyzer.load_code_files("user_files")
            analyzed_files = code_analyzer.analyze_all_files(code_files)
            
            # Analyze all notes (.md, .txt etc.)
            notes_analyzer = NotesAnalyzer(model_name=Settings.NOTES_ANALYSIS_MODEL)
            notes = notes_analyzer.load_user_notes("user_files")
            analyzed_notes = notes_analyzer.analyze_all_user_notes(notes)
            
            # Generate paper outline (auto-saved to output/paper_concept.md)
            concept_builder = PaperConception(
                model_name=Settings.PAPER_CONCEPTION_MODEL,
                user_code=analyzed_files,
                user_notes=analyzed_notes
            )
            paper_concept = concept_builder.build_paper_concept()

        #######################################
        # Step 2/11: Build search queries     #
        #######################################
        literature_search = LiteratureSearch(model_name=Settings.LITERATURE_SEARCH_MODEL)
        
        if Settings.LOAD_SEARCH_QUERIES:
            search_queries = LiteratureSearch.load_search_queries("output/search_queries.json")
        else:
            search_queries = literature_search.build_search_queries(paper_concept)


        #######################################
        # Step 3/11: Execute searches         #
        #######################################
        if Settings.LOAD_PAPERS:
            all_papers: List[Paper] = LiteratureSearch.load_papers("output/papers.json")
        else:
            all_papers = literature_search.search_papers(search_queries, max_results_per_query=30)

            all_papers = literature_search.get_citation_counts(all_papers)
            all_papers = literature_search.get_bibtex_for_papers(all_papers)

        #######################################################
        # Step 4/11: Rank, filter and download papers       #
        #######################################################
        if Settings.LOAD_PAPER_RANKING:
            # Load set of papers that are already ranked, filtered and have markdown
            loaded_papers: List[Paper] = LiteratureSearch.load_papers("output/papers_filtered_with_markdown.json")
            papers_with_markdown: List[Paper] = [p for p in loaded_papers if getattr(p, "markdown_text", None) and p.markdown_text.strip()]
        else:
            # Rank papers based on embedding similarity, citation counts, and publication date
            # The embeddings of the papers are based on title and abstract.
            ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
            ranking_context = f"{paper_concept.description}\nOpen Research Questions:\n{paper_concept.open_questions}"
            ranked_papers: List[Paper] = ranker.rank_papers(
                papers=all_papers,
                context=ranking_context,
                weights={
                    'relevance': 0.7,
                    'citations': 0.2,
                    'recency': 0.1
                }
            )
            
            # Filter the ranked papers
            filtered_papers: List[Paper] = PaperFilter.filter_diverse(
                papers=ranked_papers,
                n_cutting_edge=15,
                n_hidden_gems=15,
                n_classics=15,
                n_well_rounded=15
            )
            PaperRanker.print_ranked_papers(filtered_papers, n=10)

            # Download papers
            literature_search.download_papers_as_pdfs(filtered_papers, base_folder="literature/")
            
            # Convert papers to markdown and update markdown_text field
            converter = PDFConverter(fix_math=False, extract_media=True)
            papers_with_markdown: List[Paper] = converter.convert_all_papers(filtered_papers, base_folder="literature/")
            literature_search.save_papers(papers_with_markdown, filename="papers_filtered_with_markdown.json")


        #######################################
        # Step 5/11: Extract Findings         #
        #######################################
        if Settings.LOAD_FINDINGS:
            findings = PaperAnalyzer.load_findings("output/paper_findings.json")
        else:
            analyzer = PaperAnalyzer(model_name=Settings.PAPER_ANALYSIS_MODEL)
            findings = analyzer.extract_findings(papers_with_markdown)
        
        #######################################
        # Step 6/11: Analyze Limitations      #
        #######################################
        if Settings.LOAD_LIMITATIONS:
            limitation_analyzer = LimitationAnalyzer()
            top_limitations = limitation_analyzer.load_limitations("output/limitations.json")
        else:
            limitation_analyzer = LimitationAnalyzer.build_from_findings(findings, paper_concept)
            top_limitations = limitation_analyzer.find_top_limitations(n=10)
            limitation_analyzer.print_limitations(n=10, show_scores=True, top_limitations=top_limitations)

        # TODO: Make thresholds based on top scores, not hardcoded
                
        #######################################
        # Step 7/11: Generate Hypotheses      #
        #######################################
        hypothesis_builder = HypothesisBuilder(
            model_name=Settings.HYPOTHESIS_BUILDER_MODEL,
            embedding_model_name=Settings.HYPOTHESIS_BUILDER_EMBEDDING_MODEL,
            paper_concept=paper_concept,
            top_limitations=top_limitations,
            num_papers_analyzed=len(findings)
        )

        if Settings.LOAD_HYPOTHESES:
            hypotheses = hypothesis_builder.load_hypotheses("output/hypotheses.json")
        else:
            hypotheses = hypothesis_builder.generate_hypotheses(n_hypotheses=5)

        print(f"Selecting best hypothesis from {len(hypotheses)} hypotheses...")
        best_hypotheses = hypothesis_builder.select_best_hypotheses(hypotheses, max_n=1)
        best_hypothesis = best_hypotheses[0] if best_hypotheses else None
        if not best_hypothesis:
            print("No hypothesis selected. Exiting.")
            return
        print(f"Selected hypothesis {best_hypothesis.id}: {best_hypothesis.description}")

        # TODO: Implememt testing multiple hypotheses
        
        #######################################
        # Step 8/11: Run Experiment           #
        #######################################

        # Q: Smart to use experiment plan for writing? - plan might not fit after code changes
        experiment_runner = ExperimentRunner()
        
        experiment_result = None
        
        if Settings.LOAD_EXPERIMENT_RESULT:
            experiment_result_file = Path("output/experiments") / f"experiment_result_{best_hypothesis.id}.json"
            if not experiment_result_file.exists():
                raise FileNotFoundError(
                    f"Experiment result not found at {experiment_result_file}. "
                    f"Set LOAD_EXPERIMENT_RESULT = False to generate it."
                )
            
            print(f"\n[PaperGenerator] Loading existing experiment result...")
            experiment_result = ExperimentRunner.load_experiment_result(str(experiment_result_file))
            print(f"  Experiment result loaded")
            print(f"  Verdict: {experiment_result.hypothesis_evaluation.verdict}")
            print(f"  Reasoning: {experiment_result.hypothesis_evaluation.reasoning}")
            
            # Exit if hypothesis was disproven or inconclusive
            verdict = experiment_result.hypothesis_evaluation.verdict.lower()
            if verdict in ["disproven", "inconclusive"]:
                print(f"\n[PaperGenerator] Hypothesis was {verdict}. Exiting program.")
                return


        # Check experiment can be run (generate plan/code or use existing)
        elif Settings.LOAD_EXPERIMENT_PLAN and Settings.LOAD_EXPERIMENT_CODE:
            # Try to run existing experiment code if available
            experiment_code_file = Path("output/experiments") / f"experiment_{best_hypothesis.id}.py"
            
            if experiment_code_file.exists():
                try:
                    result = experiment_runner.run_experiment(
                        best_hypothesis,
                        paper_concept,
                        load_existing_plan=True,
                        load_existing_code=True
                    )
                    print(f"  Experiment completed")
                    print(f"  Verdict: {result.hypothesis_evaluation.verdict}")
                    print(f"  Reasoning: {result.hypothesis_evaluation.reasoning}")
                    experiment_result = result
                except Exception as e:
                    print(f"\n[PaperGenerator] Error running existing experiment code: {e}")
                    traceback.print_exc()
                    experiment_result = None
            else:
                print(f"\n[PaperGenerator] No experiment code found for hypothesis {best_hypothesis.id}")
                print(f"[PaperGenerator] Continuing without experiment result")
                experiment_result = None
        else:
            print(f"\nTesting hypothesis {best_hypothesis.id}: {best_hypothesis.description}")
            try:
                result = experiment_runner.run_experiment(
                    best_hypothesis, 
                    paper_concept, 
                    load_existing_plan=Settings.LOAD_EXPERIMENT_PLAN,
                    load_existing_code=Settings.LOAD_EXPERIMENT_CODE
                )
                print(f"  Experiment completed")
                print(f"  Verdict: {result.hypothesis_evaluation.verdict}")
                print(f"  Reasoning: {result.hypothesis_evaluation.reasoning}")
                
                # Exit if hypothesis was disproven or inconclusive
                verdict = result.hypothesis_evaluation.verdict.lower()
                if verdict in ["disproven", "inconclusive"]:
                    print(f"\n[PaperGenerator] Hypothesis was {verdict}. Exiting program.")
                    return
                
                experiment_result = result
            except Exception as e:
                print(f"\nError running experiment: {e}")
                traceback.print_exc()
       
        # IDEA: have a fallback if context window is full (e.g. only paste code if context window is full OR only paste func signatures instead of full code if context window is full)
        # TODO: Improve prompts, ensure fairness, check for bottlenecks etc
        
        #######################################
        # Step 9/11: Write Paper              #
        #######################################
        # TODO: Test specialized embedding and writing model (Specter; LLMs from CycleResearcher paper)
        # Make evidence sizes smaller, less chunks
        # Adjust what evidence is gathered for: NOT the abstract
        # Save not just the evidence prompts, but also the prompts for writing
        # Please for the love of god add a way to save and load evidence

        # Add prints, what section is being written
        # what section is converted to LaTeX, when is latex compiled etc

        if Settings.LOAD_PAPER_DRAFT:
            print(f"\n[PaperGenerator] Loading existing paper draft...")
            try:
                paper_draft = PaperWritingPipeline.load_paper_draft("output/paper_draft.md")
                print(f"  Paper draft loaded successfully")
                print(f"  Title: {paper_draft.title}")
            except FileNotFoundError:
                print(f"  Paper draft file not found.")
                return
            except Exception as e:
                print(f"  Error loading paper draft: {e}")
                traceback.print_exc()
                return
        else:
            if experiment_result and papers_with_markdown:
                paper_writing_pipeline = PaperWritingPipeline()

                paper_draft: PaperDraft = paper_writing_pipeline.write_paper(
                    paper_concept=paper_concept,
                    experiment_result=experiment_result,
                    papers=papers_with_markdown,
                )
            else:
                print("Can't write paper: missing experiment results or indexed papers.")
                return

        #######################################
        # Step 10/11: Convert to LaTeX        #
        #######################################

        converter = PaperConverter()
        
        if Settings.LOAD_LATEX:
            latex_dir = PaperConverter.load_latex("output/latex")
            print(f"\n[PaperGenerator] Loaded existing LaTeX project from: {latex_dir}")
        else:
            # Create metadata (can be customized via settings or parameters)
            metadata = LaTeXMetadata.from_settings(
                generated_title=paper_draft.title,
            )
            
            # Convert to LaTeX
            latex_dir = converter.convert_to_latex(
                paper_draft=paper_draft,
                metadata=metadata,
                indexed_papers=papers_with_markdown,
                experiment_result=experiment_result,
            )
            
            print(f"\n[PaperGenerator] LaTeX project generated at: {latex_dir}")

        #######################################
        # Step 11/11: Compile LaTeX           #
        #######################################
        
        if converter.compile_latex(latex_dir):
            pdf_path = Path("output/result/paper.pdf")
            print(f"[PaperGenerator] PDF compiled successfully at: {pdf_path}")
        else:
            print(f"[PaperGenerator] LaTeX compilation failed. Check logs for details.")


if __name__ == "__main__":
    generator = PaperGenerator()
    generator.generate_paper()
