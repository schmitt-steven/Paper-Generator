import pytest
import sys
from pathlib import Path

# Add project root to sys.path so we can import modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from settings import Settings


@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """
    Globally isolate and mock the Settings class for all verification tests.
    
    This ensures that tests:
    1. NEVER overwrite the user's real settings.py via save_to_file().
    2. Always use a consistent set of models for E2E integration, rather 
       than whatever the user happened to select in the UI last.
    """
    
    # 1. Override the save function to do absolutely nothing during tests
    def _mock_save_to_file():
        pass
    
    Settings.save_to_file = classmethod(lambda cls: _mock_save_to_file())
    
    # 2. Hardcode the baseline models used for E2E verification
    # Using the standard 80B model to ensure consistent, testable capability
    test_llm = "qwen/qwen3-next-80b"
    test_embedding = "text-embedding-qwen3-embedding-4b@q5_0"
    
    # Apply to all phases
    Settings.CODE_ANALYSIS_MODEL = test_llm
    Settings.CONTEXT_GENERATOR_MODEL = test_llm
    Settings.LITERATURE_SEARCH_MODEL = test_llm
    Settings.HYPOTHESIS_BUILDER_MODEL = test_llm
    Settings.EXPERIMENT_PLAN_MODEL = test_llm
    Settings.EXPERIMENT_CODE_WRITE_MODEL = test_llm
    Settings.EXPERIMENT_VALIDATION_MODEL = test_llm
    Settings.EXPERIMENT_VERDICT_MODEL = test_llm
    Settings.PAPER_WRITING_MODEL = test_llm
    Settings.LATEX_GENERATION_MODEL = test_llm
    
    # Must explicitly be embedding models
    Settings.PAPER_RANKING_EMBEDDING_MODEL = test_embedding
    Settings.PAPER_INDEXING_EMBEDDING_MODEL = test_embedding
    
    # Vision models
    Settings.EXPERIMENT_PLOT_CAPTION_MODEL = "qwen3-vl-32b-instruct-mlx"
    
    # Let tests run with the user's actual template
    # Settings.LATEX_TEMPLATE is left unmodified
    
    # Use standard test credentials rather than whatever is in user's UI
    Settings.UNPAYWALL_EMAIL = "test@example.com"
    Settings.SEMANTIC_SCHOLAR_API_KEY = ""

    yield  # Run tests
    # Mocking is torn down after the session (though Python exits anyway)
