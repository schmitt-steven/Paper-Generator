import traceback
from typing import List
from pathlib import Path
from phases.context_analysis.user_code_analysis import CodeAnalyzer
from phases.context_analysis.paper_conception import PaperConception, PaperConcept
from phases.context_analysis.user_requirements import load_user_requirements
from phases.paper_search.literature_search import LiteratureSearch
from phases.paper_search.user_paper_loader import UserPaperLoader
from phases.paper_search.paper import Paper
from phases.paper_search.paper_ranking import PaperRanker
from phases.paper_search.paper_filtering import PaperFilter
from phases.hypothesis_generation.paper_analysis import PaperAnalyzer
from phases.hypothesis_generation.limitation_analysis import LimitationAnalyzer
from phases.hypothesis_generation.hypothesis_builder import HypothesisBuilder
from phases.experimentation.experiment_runner import ExperimentRunner
from phases.experimentation.experiment_state import ExperimentResult
from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from phases.paper_writing.data_models import PaperDraft
from phases.latex_generation.paper_converter import PaperConverter
from phases.latex_generation.metadata import LaTeXMetadata
from settings import Settings


class PaperGenerator: 

    # X TODO: Switch from Arxiv to Semantic Scholar
    # X TODO: Add own papers
    
    # TODO: Add setting for automatically generating hypothesis
    # TODO: Build User Requirements frame

    # TODO: Add Tkinter interface with human-in-the-loop feature
        # Screen: settings (Phases (including models, batch sizes etc.), LaTeX data, starting point)
        # Screen: check paper concept
        # Screen: add papers, checkbox: search for additional papers (if checked: slider for number of papers)
        # Screen: check hypothesis
        # Screen: check experiment plan
    # Tkinter GUI features:
        # Tabs: to preview and edit Markdown
        # Drag and drop field to add own papers, left click to select from file browser
        # Dropdown with available models from LM Studio
        # Dropdown for starting point
        # Bottom section/bar,on each screen, with button to continue/start a step
    
    # TODO: Remove table of contents from tex
    # TODO: Move results into folders of each step
    # Check: Paper Concept still needed?
    # TODO: Add review/improvement loop to the paper writing process

    def generate_paper(self):

        # Load user requirements (always loaded as it's needed for hypothesis check)
        user_requirements = load_user_requirements("user_files/user_requirements.md")
        user_provided_hypothesis = bool(user_requirements.hypothesis and user_requirements.hypothesis.strip())

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
            
            # Generate paper outline (auto-saved to output/paper_concept.md)
            concept_builder = PaperConception(
                model_name=Settings.PAPER_CONCEPTION_MODEL,
                user_code=analyzed_files,
                user_requirements=user_requirements
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
        
        # Load user-provided papers and merge with searched papers
        user_paper_loader = UserPaperLoader(model_name=Settings.LITERATURE_SEARCH_MODEL)
        user_papers = user_paper_loader.load_user_papers(
            folder_path="user_files/papers/",
            s2_api=literature_search.s2_api
        )
        
        if user_papers:
            print(f"\nMerging {len(user_papers)} user-provided paper(s) with {len(all_papers)} searched paper(s)...")
            # Detect and merge duplicates (prefer user papers)
            all_papers = literature_search.detect_and_merge_duplicates(user_papers, all_papers)
            print(f"Total papers after merging: {len(all_papers)}\n")

        #######################################################
        # Step 4/11: Rank, filter and download papers       #
        #######################################################
        if Settings.LOAD_PAPER_RANKING:
            # Load set of papers that are already ranked, filtered and have markdown
            loaded_papers: List[Paper] = LiteratureSearch.load_papers("output/papers_filtered_with_markdown.json")
            # Filter: include papers that have markdown text (both user-provided and searched need markdown)
            papers_with_markdown: List[Paper] = []
            for p in loaded_papers:
                if getattr(p, "markdown_text", None) is not None and isinstance(p.markdown_text, str) and p.markdown_text.strip():
                    papers_with_markdown.append(p)
            
            # Count user-provided papers in final list
            user_with_markdown = [p for p in papers_with_markdown if p.user_provided]
            
            print(f"\n[PaperGenerator] Loaded {len(papers_with_markdown)} papers from saved ranking")
        else:
            # Separate user-provided papers (skip ranking for these)
            user_provided_papers = [p for p in all_papers if p.user_provided]
            searched_papers = [p for p in all_papers if not p.user_provided]
            
            print(f"\n[PaperGenerator] Ranking {len(searched_papers)} searched papers (skipping {len(user_provided_papers)} user-provided papers)...")
            
            # Rank only automatically searched papers
            ranker = PaperRanker(embedding_model_name=Settings.PAPER_RANKING_EMBEDDING_MODEL)
            ranking_context = f"{paper_concept.description}\nOpen Research Questions:\n{paper_concept.open_questions}"
            ranked_searched_papers: List[Paper] = ranker.rank_papers(
                papers=searched_papers,
                context=ranking_context,
                weights={
                    'relevance': 0.7,
                    'citations': 0.2,
                    'recency': 0.1
                }
            ) if searched_papers else []
            
            # Merge user-provided papers back into final list
            ranked_papers: List[Paper] = user_provided_papers + ranked_searched_papers

            # Filter papers: include papers that have markdown text
            # User-provided papers bypass ranking but still need markdown to be usable
            papers_with_markdown: List[Paper] = []
            for p in ranked_papers:
                if getattr(p, "markdown_text", None) is not None and isinstance(p.markdown_text, str) and p.markdown_text.strip():
                    papers_with_markdown.append(p)
            
            # Count user-provided papers in final list
            user_with_markdown = [p for p in papers_with_markdown if p.user_provided]
            
            print(f"\n[PaperGenerator] Final paper list: {len(papers_with_markdown)} papers with markdown")           
            if len(user_provided_papers) > len(user_with_markdown):
                missing = len(user_provided_papers) - len(user_with_markdown)
                print(f"Warning: {missing} user-provided paper(s) missing markdown (will not be usable)")

        findings = []
        top_limitations = []

        if user_provided_hypothesis:
            print(f"\n[PaperGenerator] User hypothesis provided. Skipping Findings and Limitations analysis.")
        else:
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
                top_limitations = LimitationAnalyzer.load_limitations("output/limitations.json")
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
            hypotheses = HypothesisBuilder.load_hypotheses("output/hypotheses.json")
        elif user_provided_hypothesis:
            print(f"Using user-provided hypothesis...")
            hypotheses = hypothesis_builder.create_hypothesis_from_user_input(user_requirements)
        else:
            hypotheses = hypothesis_builder.generate_hypotheses(n_hypotheses=5)

        if user_provided_hypothesis:
            selected_hypothesis = hypotheses[0] if hypotheses else None
            if selected_hypothesis:
                 print(f"Auto-selected user hypothesis {selected_hypothesis.id}")
        else:
            print(f"Selecting best hypothesis from {len(hypotheses)} hypotheses...")
            selected_hypothesis = hypothesis_builder.select_best_hypotheses(hypotheses, max_n=1)[0]
            
        if not selected_hypothesis:
            print("No hypothesis selected. Exiting.")
            return
        print(f"Using hypothesis: {selected_hypothesis.description}")
        
        #######################################
        # Step 8/11: Run Experiment           #
        #######################################

        # Q: Smart to use experiment plan for writing? - plan might not fit after code changes
        experiment_runner = ExperimentRunner()
        
        experiment_result = None
        
        if Settings.LOAD_EXPERIMENT_RESULT:
            experiment_result_file = Path("output/experiments") / f"experiment_result_{selected_hypothesis.id}.json"
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
            experiment_code_file = Path("output/experiments") / f"experiment_{selected_hypothesis.id}.py"
            
            if experiment_code_file.exists():
                try:
                    result = experiment_runner.run_experiment(
                        selected_hypothesis,
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
                print(f"\n[PaperGenerator] No experiment code found for hypothesis {selected_hypothesis.id}")
                print(f"[PaperGenerator] Continuing without experiment result")
                experiment_result = None
        else:
            print(f"\nTesting hypothesis {selected_hypothesis.id}: {selected_hypothesis.description}")
            try:
                result = experiment_runner.run_experiment(
                    selected_hypothesis, 
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
        # Exclude used chunks, not seen chunks
        
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
        
        print(f"\n{'='*80}")
        print(f"CONVERTING TO LATEX")
        print(f"{'='*80}\n")

        converter = PaperConverter()
        
        if Settings.LOAD_LATEX:
            latex_dir = PaperConverter.load_latex("output/latex")
            print(f"[PaperGenerator] Loaded existing LaTeX project from: {latex_dir}")
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
            
            print(f"[PaperGenerator] LaTeX project generated at: {latex_dir}")

        #######################################
        # Step 11/11: Compile LaTeX           #
        #######################################
        
        print(f"\n{'='*80}")
        print(f"COMPILING LATEX")
        print(f"{'='*80}\n")
        
        if converter.compile_latex(latex_dir):
            pdf_path = Path("output/result/paper.pdf")
            print(f"[PaperGenerator] PDF compiled successfully at: {pdf_path}")
        else:
            print(f"[PaperGenerator] LaTeX compilation failed. Check logs for details.")


if __name__ == "__main__":
    generator = PaperGenerator()
    generator.generate_paper()
