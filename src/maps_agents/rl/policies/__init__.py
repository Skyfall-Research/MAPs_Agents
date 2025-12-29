"""
Hierarchical Policy Package for Theme Park Tycoon.

Provides custom SB3 policies for efficient learning with sparse action spaces.
"""

from .hierarchical_multidiscrete_policy import HierarchicalMultiDiscretePolicy
from .simple_hierarchical_policy import SimpleHierarchicalPolicy

__all__ = ['HierarchicalMultiDiscretePolicy', 'SimpleHierarchicalPolicy']
