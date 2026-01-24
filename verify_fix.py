import re
import markdown

text = """
**Required Mathematical Formulas/Technical Details:**  
- **Bellman Optimality Update (RBQL):**  
  Upon reaching a terminal state...
"""

# Logic from MarkdownLabel.set_markdown
markdown_text = re.sub(r'([^\n])\n(\s*[-*+] )', r'\1\n\n\2', text)
markdown_text = re.sub(r'([^\n])\n(\s*\d+\. )', r'\1\n\n\2', markdown_text)

print(f"--- Transformed Text ---\n{markdown_text}\n------------------------")

html = markdown.markdown(markdown_text, extensions=['fenced_code', 'tables', 'sane_lists'])
print(f"--- Generated HTML ---\n{html}\n----------------------")

if "<ul>" in html or "<ol>" in html:
    print("SUCCESS: List detected.")
else:
    print("FAILURE: No list detected.")
