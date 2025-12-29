"""
MAPs Agents: Agent implementations for the Mini Amusement Park Simulator.

This package provides various agent implementations (LLM, RL, VLM) that can
interact with the MAPs environment.
"""

__version__ = "0.1.0"

# Core evaluation framework
from .eval import AbstractAgent, evaluate_agent, ActionGenerationError
from .eval.resource_interface import Resource, ResourceCost
from .eval.utils import EvalConfig
from .eval.state_interface import (
    GameResponse,
    MapPydanticGameResponse,
    MapRawGameResponse,
)

# Agent implementations
from .rl import RLAgent
from .llm import ReactAgent
from .vlm import ReactVLMAgent

__all__ = [
    # Version
    '__version__',

    # Core framework
    'AbstractAgent',
    'evaluate_agent',
    'ActionGenerationError',
    'Resource',
    'ResourceCost',
    'EvalConfig',
    'GameResponse',
    'OBSERVATION_TYPE_TO_GAME_RESPONSE',

    # RL agents and policies
    'RLAgent',
    'HierarchicalMultiDiscretePolicy',
    'SimpleHierarchicalPolicy',

    # LLM agents
    'ReactAgent',
    'ReactVLMAgent',
]
