import lmstudio as lms


def is_lm_studio_running() -> bool:
    """Check if LM Studio is running and accessible.
    
    Returns:
        True if LM Studio is running, False otherwise.
    """
    try:
        # Try to list models - this will fail if LM Studio isn't running
        lms.list_downloaded_models("llm")
        return True
    except Exception:
        return False


def get_model_names(model_type="llm", vision_only=False):
    """Fetch model keys from LM Studio.
    
    Args:
        model_type: "llm" or "embedding"
        vision_only: If True, only return LLM models with vision support
    
    Returns:
        List of model keys (strings)
    """
    models = lms.list_downloaded_models(model_type)
    
    if vision_only:
        # Filter for vision models, only works for LLM models
        if model_type != "llm":
            return []
        # Access vision attribute via the info property
        vision_models = [model.model_key for model in models if hasattr(model, 'info') and hasattr(model.info, 'vision') and model.info.vision]
        return vision_models
    
    # Return all model keys
    return [model.model_key for model in models]

