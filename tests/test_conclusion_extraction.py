
import os
import glob
import random
import sys

# Add project root to path
sys.path.append(os.getcwd())

from utils.file_utils import preprocess_markdown, remove_references_section, extract_conclusion

def test_extraction():
    literature_dir = "output/literature"
    md_files = glob.glob(os.path.join(literature_dir, "**/*.md"), recursive=True)
    
    if not md_files:
        print("No markdown files found in output/literature/")
        return

    # Pick 5 random files (or all if < 5)
    sample_files = random.sample(md_files, min(5, len(md_files)))

    print(f"Testing conclusion extraction on {len(sample_files)} random papers...\n")

    for md_file in sample_files:
        print(f"File: {os.path.basename(md_file)}")
        print(f"Path: {md_file}")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simulate the pipeline
            print("  1. Preprocessing...")
            text = preprocess_markdown(text)
            
            print("  2. Stripping references...")
            text = remove_references_section(text)
            
            print("  3. Extracting conclusion...")
            conclusion = extract_conclusion(text)
            
            if conclusion:
                print(f"  [SUCCESS] Extracted {len(conclusion)} chars")
                print(f"  Preview: {conclusion[:200].replace(chr(10), ' ')}...")
            else:
                # Check if the text actually contains the word "Conclusion" or "Concluding Remarks" to see if it SHOULD have succeeded
                if "CONCLUSION" in text.upper() or "CONCLUDING REMARKS" in text.upper():
                    print("  [FAILURE] 'Conclusion' detected in text but extraction failed.")
                else:
                    print("  [N/A] No conclusion section found in text.")
                
        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    test_extraction()
