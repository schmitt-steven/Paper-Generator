import unittest
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to path so we can import phases
sys.path.insert(0, str(Path(__file__).parent.parent))

from phases.context_analysis.user_code_analysis import CodeAnalyzer, UserCode
from phases.experimentation.experiment_runner import ExperimentRunner

class TestExperimentUserCode(unittest.TestCase):
    def setUp(self):
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.user_files_dir = os.path.join(self.test_dir, "user_files")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.user_files_dir)
        os.makedirs(self.output_dir)
        
        # Create a dummy user code file
        self.code_content = """
import math

CONST_VAL = 42

def calculate_metric(data: list[float], limit: int = 10) -> float:
    \"\"\"Calculates a specialized metric.\"\"\"
    return sum(data[:limit]) / len(data)

class DataProcessor:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def normalize(self, x: float) -> float:
        return x / 100.0
"""
        self.code_file_path = os.path.join(self.user_files_dir, "my_utils.py")
        with open(self.code_file_path, 'w') as f:
            f.write(self.code_content)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ast_extraction(self):
        """Test that AST extraction correctly identifies signatures."""
        analyzer = CodeAnalyzer()
        # Manually create UserCode object to skip LLM analysis part
        user_code = UserCode(
            file_path=self.code_file_path,
            file_name="my_utils.py",
            file_content=self.code_content
        )
        
        analyzer.extract_signatures(user_code)
        
        print("\nExtracted Signatures:")
        for sig in user_code.signatures:
            print(f"  {sig}")
            
        self.assertTrue(any("def calculate_metric(data, limit)" in s for s in user_code.signatures))
        self.assertTrue(any("Class: DataProcessor" in s for s in user_code.signatures))
        self.assertTrue(any("method: normalize(self, x)" in s for s in user_code.signatures))

    def test_runner_integration(self):
        """Test file copying and prompt formatting in ExperimentRunner."""
        runner = ExperimentRunner(base_output_dir=self.output_dir)
        
        # Mock user code list with signatures already extracted
        user_code_obj = UserCode(
            file_path=self.code_file_path,
            file_name="my_utils.py",
            file_content=self.code_content
        )
        user_code_obj.signatures = [
            "Function: def calculate_metric(data, limit) -> float",
            "Class: DataProcessor"
        ]
        user_code_list = [user_code_obj]
        
        # 1. Test Prompt Formatting (Signatures)
        formatted = runner._format_user_code_files(user_code_list, use_signatures_only=True)
        print("\nFormatted Prompt (Signatures):")
        print(formatted)
        
        self.assertIn("[AVAILABLE LOCAL MODULES]", formatted)
        self.assertIn("Module: my_utils.py", formatted)
        self.assertIn("def calculate_metric", formatted)
        self.assertNotIn("import math", formatted) # Should NOT contain full content
        
        # 2. Test Prompt Formatting (Full Content)
        formatted_full = runner._format_user_code_files(user_code_list, use_signatures_only=False)
        self.assertIn("[USER CODE FILES]", formatted_full)
        self.assertIn("import math", formatted_full) # Should contain full content
        
        # 3. Test File Copying Logic (Simulate logic from run_experiment)
        src_path = user_code_obj.file_path
        dest_path = os.path.join(runner.base_output_dir, user_code_obj.file_name)
        shutil.copy2(src_path, dest_path)
        
        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, self.code_content)

if __name__ == "__main__":
    unittest.main()
