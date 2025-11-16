"""Lazy model loading to conserve RAM and work with LM Studio's single-model mode."""

import lmstudio as lms
from typing import Optional


class LazyModelMixin:
    """
    Mixin class for lazy loading of LLM models.
    
    Models are only loaded when first accessed, not during __init__.
    This works well with LM Studio's "Only Keep Last JIT Loaded Model" server setting,
    which conserves RAM.
    """
    
    @property
    def model(self):
        """Lazy-load the LLM model on first access."""
        if not hasattr(self, '_model') or self._model is None:
            if not hasattr(self, 'model_name'):
                raise AttributeError("model_name must be set before accessing model")
            self._model = lms.llm(self.model_name)
        return self._model


class LazyEmbeddingMixin:
    """
    Mixin class for lazy loading of embedding models.
    
    Similar to LazyModelMixin but for embedding models.
    """
    
    @property
    def embedding_model(self):
        """Lazy-load the embedding model on first access."""
        if not hasattr(self, '_embedding_model') or self._embedding_model is None:
            if not hasattr(self, 'embedding_model_name'):
                raise AttributeError("embedding_model_name must be set before accessing embedding_model")
            self._embedding_model = lms.embedding_model(self.embedding_model_name)
        return self._embedding_model

