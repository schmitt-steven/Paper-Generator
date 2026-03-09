import subprocess
import shutil
from typing import Optional
import lmstudio as lms
from settings import Settings


class LMSJITSettings:
    """
    Context manager that explicitly loads both the LLM and embedding model
    via the `lms` CLI, bypassing LM Studio's JIT auto-unload behavior.

    CLI-loaded models are treated as explicitly loaded and are not subject
    to the 'Only Keep Last JIT Loaded Model' setting.

    On exit, both models are unloaded to free memory.
    """

    def __init__(self):
        self._lms_path = shutil.which("lms")

    def _run_lms(self, *args) -> bool:
        """Run an lms CLI command. Returns True on success."""
        if not self._lms_path:
            print("[LMSJITSettings] 'lms' CLI not found on PATH")
            return False
        try:
            result = subprocess.run(
                [self._lms_path, *args],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"[LMSJITSettings] lms {' '.join(args)} failed: {result.stderr.strip()}")
                return False
            return True
        except Exception as e:
            print(f"[LMSJITSettings] Error running lms: {e}")
            return False

    def __enter__(self):
        llm_id = Settings.PAPER_WRITING_MODEL
        emb_id = Settings.PAPER_INDEXING_EMBEDDING_MODEL

        # Check what's already loaded
        loaded_ids = {m.identifier for m in lms.list_loaded_models()}

        # Explicitly load LLM via CLI (bypasses JIT auto-unload)
        if llm_id not in loaded_ids:
            print(f"[LMSJITSettings] Loading LLM: {llm_id}")
            self._run_lms("load", llm_id, "--yes")
        else:
            print(f"[LMSJITSettings] LLM already loaded: {llm_id}")

        # Explicitly load embedding model via CLI
        if emb_id not in loaded_ids:
            print(f"[LMSJITSettings] Loading embedding model: {emb_id}")
            self._run_lms("load", emb_id, "--yes")
        else:
            print(f"[LMSJITSettings] Embedding model already loaded: {emb_id}")

        # Verify
        loaded_after = [m.identifier for m in lms.list_loaded_models()]
        print(f"[LMSJITSettings] Models in memory: {loaded_after}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("[LMSJITSettings] Unloading models...")
        try:
            for model in lms.list_loaded_models():
                try:
                    model.unload()
                    print(f"[LMSJITSettings] Unloaded {model.identifier}")
                except Exception as e:
                    print(f"[LMSJITSettings] Failed to unload {model.identifier}: {e}")
        except Exception as e:
            print(f"[LMSJITSettings] Error while unloading models: {e}")
