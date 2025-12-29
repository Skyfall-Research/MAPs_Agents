"""
Evaluation utilities for hyperparameter tuning.
"""

import numpy as np
from typing import List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from maps_agents.rl.train_agent import make_env


def evaluate_on_layouts(
    model: PPO,
    test_layouts: List[str],
    n_episodes: int,
    host: str,
    port: str,
    difficulty: str,
    mode: str
) -> Tuple[float, float]:
    """
    Evaluate model on multiple test layouts.

    Returns:
        Tuple of (mean_reward, std_reward) across all episodes and layouts
    """
    all_rewards = []

    for layout in test_layouts:
        # Create environment for this layout
        env = DummyVecEnv([
            make_env(host, port, difficulty, mode, seed=200, layouts=[layout], env_idx=0)
        ])

        try:
            # Run episodes
            for _ in range(n_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward[0]

                    if done[0]:
                        break

                all_rewards.append(episode_reward)
        finally:
            env.close()

    return float(np.mean(all_rewards)), float(np.std(all_rewards))
