"""
Objective function for Optuna hyperparameter tuning.
"""

import optuna
from optuna.trial import Trial
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from maps_agents.rl.tuning.train_with_params import train_with_hyperparams


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
        deterministic: bool = False,
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
    trial: "Trial",
    training_layouts: str = "all",
    mode: str = "full",
    difficulty: str = "easy",
    n_eval_episodes: int = 10,
    n_timesteps: int = 500_000,  # shorter for tuning
    # NOTE: we now tune n_envs inside the objective; keep this arg only as a fallback/default
    n_envs: int = 25,
    host: str = "localhost",
    port: str = "3000",
    save_path: str = "./tuning_runs",
) -> float:
    """
    Optuna objective for PPO tuning with:
      - tuned n_envs (CPU-heavy stepping)
      - conditional n_steps based on n_envs
      - derived batch_size from num_minibatches and total rollout size (N = n_steps * n_envs)
      - tightened, more PPO-sane ranges for gamma/epochs/clip/entropy
      - optional stability knobs (vf_coef, max_grad_norm, target_kl)
    """

    # 1) Tune rollout length to keep total samples/update reasonable
    #    N = n_steps * n_envs ~ 8k–32k (rough target)
    # -------------------------
    # horizon=100 -> nice to include multiples of 100, but SB3 supports any int
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])

    # Total rollout samples per PPO update
    rollout_size = n_steps * n_envs

    # -------------------------
    # 2) Derive batch_size via num_minibatches
    # -------------------------
    num_minibatches = trial.suggest_categorical("num_minibatches", [8, 16, 32])
    batch_size = rollout_size // num_minibatches

    # Enforce constraints:
    # - batch_size must be >= 64 (avoid super noisy minibatches)
    # - need at least 4 minibatches
    # - batch_size must be > 0 and divide nicely (it will, by construction)
    if batch_size < 64 or (rollout_size // batch_size) < 4:
        return -float("inf")

    # (Optional) keep batch_size in a reasonable cap if you find huge batches slow learning
    if batch_size > 4096:
        return -float("inf")

    # -------------------------
    # 4) Sample PPO hyperparameters (tighter, better-behaved ranges)
    # -------------------------
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])

    # Epochs: avoid very large values
    n_epochs = trial.suggest_categorical("n_epochs", [2, 3, 5, 10])

    # horizon=100: gamma near 0.99+ is usually appropriate
    gamma = 0.995
    gae_lambda = trial.suggest_float("gae_lambda", 0.92, 0.98)

    # clip range: avoid very aggressive clipping most of the time
    clip_range = 0.2

    # entropy: avoid absurd extremes
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 3e-2, log=True)

    # -------------------------
    # 5) Add a couple high-leverage stability knobs
    # -------------------------
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Optional: target_kl can prevent catastrophic updates (works well with pruning)
    target_kl = trial.suggest_categorical("target_kl", [None, 0.01, 0.02, 0.05])

    # Optional: clip_range_vf for value function clipping
    clip_range_vf = trial.suggest_categorical("clip_range_vf", [None, 0.5, 1.0, 2.0])

    try:
        # Trial-specific save path
        trial_save_path = f"{save_path}/trial_{trial.number}"
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
            # Hyperparams
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            clip_range_vf=clip_range_vf,
            n_eval_episodes=n_eval_episodes,
            # Trial for pruning
            trial=trial,
        )

        # Pruning check (kept from your original flow)
        if getattr(eval_callback, "is_pruned", False):
            raise optuna.TrialPruned()

        # Use final evaluation results from callback
        # (already evaluated on test_layouts with n_eval_episodes episodes)
        if len(eval_callback.evaluations_results) > 0:
            mean_reward = float(np.mean(eval_callback.evaluations_results[-1]))
            std_reward = float(np.std(eval_callback.evaluations_results[-1]))
        else:
            # Fallback if no evaluations were performed
            mean_reward = -float("inf")
            std_reward = 0.0

        # Log attrs for analysis
        trial.set_user_attr("std_reward", float(std_reward))
        trial.set_user_attr("training_layouts", training_layouts)
        trial.set_user_attr("rollout_size", int(rollout_size))
        trial.set_user_attr("num_minibatches", int(num_minibatches))
        trial.set_user_attr("batch_size", int(batch_size))
        trial.set_user_attr("n_envs", int(n_envs))
        trial.set_user_attr("n_steps", int(n_steps))

        return float(mean_reward)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
