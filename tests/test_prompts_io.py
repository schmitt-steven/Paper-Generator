from phases.paper_writing.paper_writing_pipeline import PaperWritingPipeline
from pathlib import Path
import json

def test_json_prompts():
    pipeline = PaperWritingPipeline()
    output_dir = "tests/temp_output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Test Data: Prompts with tricky Markdown content
    prompts = {
        "Introduction": "Write an intro.",
        "Methods": "Use code:\n```python\n# This is a comment\n```\n# Header inside prompt",
        "Results": "Another # header here."
    }
    
    filename = "test_prompts.json"
    
    # 1. Save
    print("Saving prompts...")
    pipeline._save_prompts(prompts, filename, output_dir)
    
    saved_path = Path(output_dir) / filename
    if not saved_path.exists():
        print("FAIL: File not created.")
        return

    # 2. Check file content manually
    print("Checking JSON content...")
    with open(saved_path, 'r') as f:
        data = json.load(f)
        if data != prompts:
            print("FAIL: JSON content mismatch.")
            print("Expected:", prompts)
            print("Got:", data)
            return
            
    # 3. Load using pipeline logic
    print("Loading prompts...")
    loaded = pipeline.load_section_writing_prompts(str(saved_path))
    
    if loaded == prompts:
        print("PASS: Saved and loaded successfully.")
    else:
        print("FAIL: Pipeline load mismatch.")

    # Cleanup
    import shutil
    shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_json_prompts()
