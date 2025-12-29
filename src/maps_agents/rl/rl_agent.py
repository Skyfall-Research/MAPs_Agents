"""
Stable-Baselines3 Agent for MAPs Environment

This module provides an agent that loads pre-trained SB3 models
and performs inference through the AbstractAgent interface.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Union
from stable_baselines3 import PPO
import yaml

from maps_agents.eval import AbstractAgent, ActionGenerationError
from maps_agents.eval.resource_interface import ResourceCost
from maps_agents.eval.utils import EvalConfig
from maps_agents.eval.state_interface import MapGameResponse


class RLAgent(AbstractAgent):
    """
    Reinforcement Learning agent for MAPs environment.

    Loads pre-trained PPO models and performs inference. Supports both
    "simple" mode (5 actions, vectors only) and "full" mode (11 actions,
    grid + vectors).

    Attributes:
        model: Loaded Stable-Baselines3 PPO model
        mode: "simple" or "full" observation/action mode
        difficulty: Game difficulty level
        model_path: Path to the loaded model file
        state_history: Dictionary tracking state per run_id
    """

    def __init__(
        self,
        agent_config_path: Union[str | os.PathLike],
        difficulty: str,
        model_path: Optional[str | os.PathLike] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Initialize RLAgent.

        Args:
            model_path: Path to .zip model file
            mode: "simple" or "full" mode
            difficulty: "easy", "medium", or "hard"
            training_layouts: "all", "ribs", "the_islands", "zig_zag"
            name: Optional agent name

        Raises:
            ValueError: If parameters are invalid or model cannot be loaded
            FileNotFoundError: If model file doesn't exist
        """
        super().__init__(name)

        # Load config first
        with open(agent_config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.mode = config['mode']
        self.training_layouts = config['training_layouts']
        self.difficulty = difficulty

        # Validate parameters
        if self.mode not in ["simple", "full"]:
            raise ValueError(f"mode must be 'simple' or 'full', got {self.mode}")
        if self.difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"difficulty must be 'easy', 'medium', or 'hard', got {self.difficulty}")
        if self.training_layouts not in ["all", "ribs", "the_islands", "zig_zag"]:
            raise ValueError(f"training_layouts must be 'all', 'ribs', 'the_islands', or 'zig_zag', got {self.training_layouts}")

        # Load model
        if model_path is None:
            model_path = os.path.join("./trained_models", f"{self.mode}_{self.difficulty}_{self.training_layouts}", "final_model.zip")

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print(f"Training new model with mode={self.mode}, difficulty={self.difficulty}, training_layouts={self.training_layouts}")
            model_path = self._train_model(self.training_layouts, self.mode, self.difficulty, "./trained_models")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = PPO.load(model_path)
        self.model_path = model_path

        # Initialize state tracking
        self.state_history: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def get_agent(agent_config_path: Optional[str | os.PathLike], eval_config: EvalConfig) -> 'RLAgent':
        return RLAgent(
            agent_config_path=agent_config_path,
            difficulty=eval_config.difficulty,
        )


    @property
    def observation_type(self) -> str:
        if self.mode == "simple":
            return 'gym_simple'
        else:
            return 'gym'

    @staticmethod
    def _train_model(
        training_layouts: str,
        mode: str,
        difficulty: str,
        base_path: str,
    ) -> str:
        """
        Invoke training script programmatically.

        Args:
            training_layouts: Training variant to use
            mode: "simple" or "full"
            difficulty: Game difficulty
            base_path: Base path for saving models
            agent_type: Agent type (currently only "ppo" supported)
            train_kwargs: Additional kwargs for train_agent()
        """
        # Import here to avoid circular dependency
        from maps_agents.rl.train_agent import train_agent

        # Invoke training
        final_model_path = train_agent(
            difficulty=difficulty,
            mode=mode,
            total_timesteps=500000,
            save_path=base_path,
            training_layouts=training_layouts
        )
        return final_model_path

    def act(
        self,
        game_response: MapGameResponse,
        run_id: int,
        logging_id: Optional[str] = None
    ) -> str:
        """
        Generate action for current game state.

        Args:
            game_inputs: Current game state
            run_id: Unique trajectory ID
            logging_id: Optional logging identifier

        Returns:
            Action string in Python function call format

        Raises:
            ActionGenerationError: If action generation fails
        """
        try:
            # Get action from model
            action, _ = self.model.predict(game_response.obs, deterministic=True)
            return action

        except Exception as e:
            raise ActionGenerationError(f"Failed to generate action: {str(e)}")

    def get_action_resource_usage(self, reset: bool = True) -> ResourceCost:
        """
        Get resource usage (empty for RL agents).
        """
        return ResourceCost()


get_agent = RLAgent.get_agent