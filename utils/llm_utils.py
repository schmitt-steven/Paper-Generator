import re


def remove_thinking_blocks(content: str) -> str:
    """Remove thinking blocks from LLM responses.

    Handles multiple formats:
    - Standard: <think>...</think>
    - Missing opening tag: everything before </think> is thinking
    """
    # Standard case: remove <think>...</think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # Handle missing opening tag: strip everything before </think>
    if '</think>' in content:
        content = content.split('</think>')[-1]

    # Clean up any extra whitespace that might be left (3+ newlines → 2)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

    return content.strip()

