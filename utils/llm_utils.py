import re


def remove_thinking_blocks(content: str) -> str:
    """Remove thinking blocks marked with `<think>`...`</think>` tags."""
    
    # Remove everything between `<think>` and `</think>` (including the tags)
    # Using DOTALL flag to match across newlines
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Clean up any extra whitespace that might be left (3+ newlines → 2)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    return content.strip()

