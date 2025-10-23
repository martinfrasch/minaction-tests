# src/__init__.py
"""
Source modules for minAction.net LLM physics testing framework
"""

from .model_interface import get_model_interface, BaseModelInterface
from .evaluation import evaluate_response, aggregate_results

__all__ = [
    'get_model_interface',
    'BaseModelInterface',
    'evaluate_response',
    'aggregate_results'
]
