"""
Training function that accepts hyperparameters from Optuna.
"""

from typing import Optional, Tuple
import os
from optuna.trial import Trial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from maps_agents.rl.train_agent import make_env, get_layouts_for_variant
from maps_agents.rl import HierarchicalMultiDiscretePolicy, SimpleHierarchicalPolicy
from maps_agents.rl.tuning.objective import TrialEvalCallback


def train_with_hyperparams(
    agent_type: str,
    host: str,
    port: str,
    difficulty: str,
    mode: str,
    training_layouts: str,
    total_timesteps: int,
    n_envs: int,
    save_path: str,
    # Hyperparameters
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    # Optional Optuna trial
    trial: Optional[Trial] = None,
    # Other parameters
    max_grad_norm: float = 0.5,
    vf_coef: float = 0.5,
) -> Tuple[PPO, EvalCallback]:
    """
    Train PPO agent with specified hyperparameters.

    Returns:
        Tuple of (trained_model, eval_callback)
    """
    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Set random seed
    set_random_seed(42)

    # Get layouts for variant
    train_layouts, test_layouts = get_layouts_for_variant(training_layouts)

    # Select policy
    policy_class = SimpleHierarchicalPolicy if mode == "simple" else HierarchicalMultiDiscretePolicy

    # Create training environments
    if n_envs == 1:
        env = DummyVecEnv([make_env(host, port, difficulty, mode, seed=42, layouts=train_layouts, env_idx=0)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    else:
        env = DummyVecEnv([
            make_env(host, port, difficulty, mode, seed=42 + i, layouts=train_layouts, env_idx=i)
            for i in range(n_envs)
        ])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(host, port, difficulty, mode, seed=100, eval_mode=True, layouts=test_layouts, env_idx=0)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    # Create callback with Optuna trial for pruning
    if trial is not None:
        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=5,
            eval_freq=max(10000 // n_envs, 1),
            deterministic=True,
            verbose=0
        )
    else:
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=f"{save_path}/best_model",
            log_path=f"{save_path}/logs",
            eval_freq=max(10000 // n_envs, 1),
            deterministic=True,
            n_eval_episodes=5
        )

    # Initialize PPO model with hyperparameters
    model = PPO(
        policy_class,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        verbose=0,  # Quiet during tuning
        tensorboard_log=f"{save_path}/tensorboard",
        policy_kwargs={"difficulty": difficulty}
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )

    # Save final model
    model.save(f"{save_path}/final_model")

    return model, eval_callback
