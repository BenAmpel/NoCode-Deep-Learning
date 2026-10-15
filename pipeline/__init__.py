"""The training pipeline behind the Train and model-sweep buttons.

``runner.run`` executes the stages (data → model → train → evaluate →
export) and raises on failure; the UI wraps it to show progress and errors.
"""
from pipeline.config import RUN_CONFIG_FIELDS, RunConfig

__all__ = ["RUN_CONFIG_FIELDS", "RunConfig"]
