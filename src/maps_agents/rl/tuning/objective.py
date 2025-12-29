"""
Objective function for Optuna hyperparameter tuning.
"""

import optuna
from optuna.trial import Trial
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class TrialEvalCallback(EvalCallback):
    """
    Callback for Optuna that reports intermediate values and prunes trials.
    """
    def __init__(
        self,
        eval_env,
        trial: Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Called on every environment step.
        Reports intermediate values to Optuna and handles pruning.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Trigger evaluation
            super()._on_step()

            # Report mean reward to Optuna
            if len(self.evaluations_results) > 0:
                mean_reward = float(np.mean(self.evaluations_results[-1]))
                self.trial.report(mean_reward, self.eval_idx)
                self.eval_idx += 1

                # Check if trial should be pruned
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False  # Stop training

        return True


def objective(
    trial: Trial,
    training_layouts: str = "all",
    mode: str = "full",
    difficulty: str = "easy",
    n_eval_episodes: int = 10,
    n_timesteps: int = 500000,  # Shorter for tuning
    n_envs: int = 4,  # Parallel envs for faster training
    host: str = "localhost",
    port: str = "3000",
    save_path: str = "./tuning_runs"
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object
        training_layouts: Training variant to use
        mode: "simple" or "full"
        difficulty: Game difficulty
        n_eval_episodes: Episodes for final evaluation
        n_timesteps: Total training timesteps (reduced for tuning)
        n_envs: Number of parallel environments
        host: Game server host
        port: Game server port
        save_path: Directory to save trial models

    Returns:
        Mean episode reward on test layouts (metric to maximize)
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)

    # Optional: Sample advanced parameters
    # max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    # vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)

    # Constraint: batch_size should be <= n_steps
    if batch_size > n_steps:
        # Return poor score for invalid configurations
        return -float('inf')

    try:
        # Create trial-specific save path
        trial_save_path = f"{save_path}/trial_{trial.number}"

        # Train model with sampled hyperparameters
        from maps_agents.rl.tuning.train_with_params import train_with_hyperparams

        model, eval_callback = train_with_hyperparams(
            agent_type="PPO",
            host=host,
            port=port,
            difficulty=difficulty,
            mode=mode,
            training_layouts=training_layouts,
            total_timesteps=n_timesteps,
            n_envs=n_envs,
            save_path=trial_save_path,
            # Hyperparameters from Optuna
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            # Optuna trial for pruning
            trial=trial
        )

        # Check if pruned
        if eval_callback.is_pruned:
            raise optuna.TrialPruned()

        # Final evaluation on test layouts
        # Get test layouts for the variant
        from maps_agents.rl.train_agent import get_layouts_for_variant
        _, test_layouts = get_layouts_for_variant(training_layouts)

        # Evaluate on test layouts
        from maps_agents.rl.tuning.evaluate import evaluate_on_layouts
        mean_reward, std_reward = evaluate_on_layouts(
            model=model,
            test_layouts=test_layouts,
            n_episodes=n_eval_episodes,
            host=host,
            port=port,
            difficulty=difficulty,
            mode=mode
        )

        # Store additional metrics as user attributes
        trial.set_user_attr("std_reward", std_reward)
        trial.set_user_attr("n_envs", n_envs)
        trial.set_user_attr("training_layouts", training_layouts)

        return float(mean_reward)

    except optuna.TrialPruned:
        # Re-raise pruned trials as-is
        raise
    except Exception as e:
        # Log full error details for debugging
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to mark as failed, not pruned
