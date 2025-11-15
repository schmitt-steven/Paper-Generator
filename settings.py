class Settings:
    """Configuration settings for the paper generator pipeline."""
    
    # Context Analysis Phase
    CODE_ANALYSIS_MODEL =    "qwen/qwen3-next-80b"  
    NOTES_ANALYSIS_MODEL =   "qwen/qwen3-next-80b"  
    PAPER_CONCEPTION_MODEL = "qwen/qwen3-next-80b"  
    
    # Paper Search Phase
    LITERATURE_SEARCH_MODEL =             "qwen/qwen3-next-80b"  
    PAPER_ANALYSIS_MODEL =                "qwen/qwen3-next-80b"
    PAPER_RANKING_EMBEDDING_MODEL =       "qwen3-embedding-4b-dwq"  # Must be an embedding model!
    LIMITATION_ANALYSIS_EMBEDDING_MODEL = "qwen3-embedding-4b-dwq"  # Must be an embedding model!
    
    # Hypothesis Generation Phase
    HYPOTHESIS_BUILDER_MODEL =           "qwen3-next-80b-a3b-thinking-mlx"
    HYPOTHESIS_BUILDER_EMBEDDING_MODEL = "qwen3-embedding-4b-dwq"  # Must be an embedding model!
    
    # Experimentation Phase
    EXPERIMENT_PLAN_MODEL =          "qwen3-next-80b-a3b-thinking-mlx"
    EXPERIMENT_CODE_WRITE_MODEL =    "qwen/qwen3-next-80b"  
    EXPERIMENT_CODE_FIX_MODEL =      "qwen/qwen3-next-80b"  
    EXPERIMENT_CODE_IMPROVE_MODEL =  "qwen/qwen3-next-80b" 
    EXPERIMENT_VALIDATION_MODEL =    "qwen/qwen3-next-80b" 
    EXPERIMENT_VERDICT_MODEL =       "qwen3-next-80b-a3b-thinking-mlx"
    EXPERIMENT_PLOT_CAPTION_MODEL =  "qwen/qwen3-next-80b"  # Must be a VISION model!
    PAPER_INDEXING_EMBEDDING_MODEL = "qwen3-embedding-4b-dwq"  # Must be an embedding model!
    
    # Paper Writing Phase
    EVIDENCE_GATHERING_MODEL = "qwen/qwen3-next-80b"  
    PAPER_WRITING_MODEL =      "qwen/qwen3-next-80b"
    
    # LaTeX Conversion Phase
    LATEX_CONVERSION_MODEL = "qwen/qwen3-next-80b"
    
    # LaTeX Metadata
    LATEX_TITLE =         ""  # If empty string, title that LLM generated will be used.
    LATEX_AUTHOR =        "Qwen3, Qwen3 and Qwen3"  # Example: "John Doe" or "John Doe, Jane Smith" or "John Doe, Jane Smith, and Bob Johnson" for multiple authors
    LATEX_SUPERVISOR =    ""  # Example: "Prof. Max Mustermann" or "Prof. Max Mustermann, Dr. Steven Smith" for multiple supervisors
    LATEX_TYPE_OF_WORK =  ""  # Example: "Master's Thesis"
    LATEX_PROGRAM =       ""  # Example: "Computer Science"
    LATEX_FACULTY =       ""  # Optional: Faculty name
    LATEX_STUDY_PROGRAM = ""  # Optional: Study program name
    LATEX_COMPANY =       ""  # Optional: Company name
    LATEX_DIGITAL_SUBMISSION = True  # True for digital, False for paper
    
    # Set to True to load existing files from output/ and skip the respective step
    LOAD_PAPER_CONCEPT =     True
    LOAD_SEARCH_QUERIES =    True
    LOAD_PAPERS =            True
    LOAD_PAPER_RANKING =     True
    LOAD_FINDINGS =          True
    LOAD_LIMITATIONS =       True
    LOAD_HYPOTHESES =        True
    LOAD_EXPERIMENT_RESULT = False  # if true, ignores LOAD_EXPERIMENT_PLAN/CODE
    LOAD_EXPERIMENT_PLAN =   True
    LOAD_EXPERIMENT_CODE =   True
    LOAD_PAPER_DRAFT =       False
    LOAD_LATEX =             False
