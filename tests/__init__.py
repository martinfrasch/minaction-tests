# tests/__init__.py
"""
Test modules for minAction.net LLM physics understanding tests
"""

from . import (
    test_forward_derivation,
    test_inverse_problems,
    test_physical_constraints,
    test_cross_domain,
    test_symmetry
)

__all__ = [
    'test_forward_derivation',
    'test_inverse_problems',
    'test_physical_constraints',
    'test_cross_domain',
    'test_symmetry'
]
